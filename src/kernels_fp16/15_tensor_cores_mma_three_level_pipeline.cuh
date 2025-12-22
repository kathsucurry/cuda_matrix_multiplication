#pragma once

#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <mma.h>

#include "14_tensor_cores_mma_swizzled.cuh"


namespace tc_mma_three_level_pipeline {

template <uint const BN, uint const BK,
      uint const MMA_M, uint const MMA_N, uint const MMA_K,
      uint const NUM_MMA_M, uint const NUM_MMA_N, uint const NUM_MMA_K,
      uint const SW_B_YYY_MASK, uint const SW_B_SHIFT>
__device__ __forceinline__ void load_smem_to_regs(
    __nv_bfloat16 *__restrict__ As,
    __nv_bfloat16 *__restrict__ Bs,
    uint32_t A_register[NUM_MMA_M][NUM_MMA_K][4],
    uint32_t B_register[NUM_MMA_K][NUM_MMA_N][2],
    uint const warp_row_offset, uint const warp_col_offset,
    uint const lane_idx
) {
    for (uint mma_k_idx{0}; mma_k_idx < NUM_MMA_K; ++mma_k_idx) {
        uint const mma_k_offset{mma_k_idx * MMA_K};
        // Load A elements from the shared memory to register.
        for (int mma_row_idx{0}; mma_row_idx < NUM_MMA_M; ++mma_row_idx) {
            uint32_t shared_A_pointer = static_cast<uint32_t>(
                __cvta_generic_to_shared(&As[
                    (warp_row_offset + mma_row_idx * MMA_M + (lane_idx % MMA_M)) * BK +
                    mma_k_offset + (lane_idx / MMA_M) * 8
                ]));
            asm volatile (
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                "{%0, %1, %2, %3}, [%4];"
                : "=r"(A_register[mma_row_idx][mma_k_idx][0]), "=r"(A_register[mma_row_idx][mma_k_idx][1]),
                  "=r"(A_register[mma_row_idx][mma_k_idx][2]), "=r"(A_register[mma_row_idx][mma_k_idx][3])
                : "r"(shared_A_pointer)
            );
        }

        // Load B elements from the shared memory to register.
        for (int mma_col_idx{0}; mma_col_idx < NUM_MMA_N; ++mma_col_idx) {
            uint32_t const offset{
                (mma_k_offset + (lane_idx % MMA_K)) * BN +
                (warp_col_offset + mma_col_idx * MMA_N)};
            uint32_t shared_B_pointer = static_cast<uint32_t>(
                __cvta_generic_to_shared(&Bs[offset ^ ((offset & SW_B_YYY_MASK) >> SW_B_SHIFT)]));
            asm volatile (
                "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
                "{%0, %1}, [%2];"
                : "=r"(B_register[mma_k_idx][mma_col_idx][0]), "=r"(B_register[mma_k_idx][mma_col_idx][1])
                : "r"(shared_B_pointer)
            );
        }
    }
}


template <uint const BN, uint const BK,
      uint const MMA_M, uint const MMA_N, uint const MMA_K,
      uint const NUM_MMA_M, uint const NUM_MMA_N, uint const NUM_MMA_K>
__device__ __forceinline__ void compute_dot_products(
    uint32_t A_register[NUM_MMA_M][NUM_MMA_K][4],
    uint32_t B_register[NUM_MMA_K][NUM_MMA_N][2],
    float acc_register[NUM_MMA_M][NUM_MMA_N][4]
) {
    for (uint mma_k_idx{0}; mma_k_idx < NUM_MMA_K; ++mma_k_idx) {
        for (int mma_row_idx{0}; mma_row_idx < NUM_MMA_M; ++mma_row_idx) {
            for (int mma_col_idx{0}; mma_col_idx < NUM_MMA_N; ++mma_col_idx) {
                // Perform MMA.
                asm volatile (
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%10, %11, %12, %13};"
                    : "=f"(acc_register[mma_row_idx][mma_col_idx][0]), "=f"(acc_register[mma_row_idx][mma_col_idx][1]),
                      "=f"(acc_register[mma_row_idx][mma_col_idx][2]), "=f"(acc_register[mma_row_idx][mma_col_idx][3])
                    : "r"(A_register[mma_row_idx][mma_k_idx][0]), "r"(A_register[mma_row_idx][mma_k_idx][1]), "r"(A_register[mma_row_idx][mma_k_idx][2]), "r"(A_register[mma_row_idx][mma_k_idx][3]),
                      "r"(B_register[mma_k_idx][mma_col_idx][0]), "r"(B_register[mma_k_idx][mma_col_idx][1]),
                      "f"(acc_register[mma_row_idx][mma_col_idx][0]), "f"(acc_register[mma_row_idx][mma_col_idx][1]),
                      "f"(acc_register[mma_row_idx][mma_col_idx][2]), "f"(acc_register[mma_row_idx][mma_col_idx][3])
                );                 
            }
        }
    }
}
};


template <uint const NUM_THREADS,
          uint const BM, uint const BN, uint const BK,
          uint const WM, uint const WN>
__global__ void __launch_bounds__(NUM_THREADS) tensor_cores_mma_three_level_pipeline_gemm(
    int M, int N, int K, float alpha,
    __nv_bfloat16 *__restrict__ A,
    __nv_bfloat16 *__restrict__ B,
    float beta,
    float         *__restrict__ C
) {
    __shared__ __nv_bfloat16 As[2][BM * BK];
    __shared__ __nv_bfloat16 Bs[2][BK * BN];

    uint const lane_idx{threadIdx.x % 32};
    uint const warp_idx{threadIdx.x / 32};
    uint const warp_row_offset{(warp_idx / (BN / WN)) * WM};
    uint const warp_col_offset{(warp_idx % (BN / WN)) * WN};

    {
        uint const block_row_offset{blockIdx.y * BM};
        uint const block_col_offset{blockIdx.x * BN};

        A += block_row_offset * K;
        B += block_col_offset;
        C += block_row_offset * N + block_col_offset;
    }

    // Define the matrix fragment size here since the implementation (e.g., fragment
    // layout) depends on the size.
    constexpr uint MMA_M{16};
    constexpr uint MMA_N{8};
    constexpr uint MMA_K{16};

    constexpr uint SW_B_SHIFT{6 - ilog2<MMA_N>()};
    constexpr uint SW_B_YYY_MASK{((1 << (ilog2<64 / MMA_N>())) - 1) << (ilog2<MMA_N>() + SW_B_SHIFT)};

    constexpr uint NUM_MMA_M{WM / MMA_M};
    constexpr uint NUM_MMA_N{WN / MMA_N};
    constexpr uint NUM_MMA_K{BK / MMA_K};

    uint32_t A_register[NUM_MMA_M][NUM_MMA_K][4];
    uint32_t B_register[NUM_MMA_K][NUM_MMA_N][2];
    // Accumulator: a vector expression of 4 .f32 registers; initialize with 0s.
    float acc_register[NUM_MMA_M][NUM_MMA_N][4] = {0.0f};

    tc_swizzled::load_gmem_to_smem<BM, BN, BK, NUM_THREADS, 3, SW_B_YYY_MASK, SW_B_SHIFT>(
        A, B, N, K,
        As[0], Bs[0],
        threadIdx.x
    );

    A += BK;
    B += BK * N;

    tc_swizzled::load_gmem_to_smem<BM, BN, BK, NUM_THREADS, 3, SW_B_YYY_MASK, SW_B_SHIFT>(
        A, B, N, K,
        As[1], Bs[1],
        threadIdx.x
    );

    // Load from smem to registers.
    __pipeline_wait_prior(1);
    __syncthreads();

    tc_mma_three_level_pipeline::load_smem_to_regs<
        BN, BK, MMA_M, MMA_N, MMA_K, NUM_MMA_M, NUM_MMA_N, NUM_MMA_K, SW_B_YYY_MASK, SW_B_SHIFT>(
            As[0], Bs[0], A_register, B_register, warp_row_offset, warp_col_offset, lane_idx
        );
    
    int current{0};

    for (int k_offset{0}; k_offset < K - 2 * BK; k_offset += BK) {
        A += BK;
        B += BK * N;

        // All previous loads from smem should finish first.
        __syncthreads();

        // Load from gmem to smem, replacing the current smem.
        tc_swizzled::load_gmem_to_smem<BM, BN, BK, NUM_THREADS, 3, SW_B_YYY_MASK, SW_B_SHIFT>(
            A, B, N, K,
            As[current], Bs[current],
            threadIdx.x
        );

        // Execute the dot product.
        tc_mma_three_level_pipeline::compute_dot_products<
            BN, BK, MMA_M, MMA_N, MMA_K, NUM_MMA_M, NUM_MMA_N, NUM_MMA_K>(
                A_register, B_register, acc_register
            );

        // Load from smem to registers.
        __pipeline_wait_prior(1);
        __syncthreads();

        tc_mma_three_level_pipeline::load_smem_to_regs<
            BN, BK, MMA_M, MMA_N, MMA_K, NUM_MMA_M, NUM_MMA_N, NUM_MMA_K, SW_B_YYY_MASK, SW_B_SHIFT>(
                As[current ^ 1], Bs[current ^ 1], A_register, B_register, warp_row_offset, warp_col_offset, lane_idx
            );

        current ^= 1;
    }
    __pipeline_wait_prior(0);
    __syncthreads();

    tc_mma_three_level_pipeline::compute_dot_products<
        BN, BK, MMA_M, MMA_N, MMA_K, NUM_MMA_M, NUM_MMA_N, NUM_MMA_K>(
            A_register, B_register, acc_register
        );
    
    tc_mma_three_level_pipeline::load_smem_to_regs<
        BN, BK, MMA_M, MMA_N, MMA_K, NUM_MMA_M, NUM_MMA_N, NUM_MMA_K, SW_B_YYY_MASK, SW_B_SHIFT>(
            As[current ^ 1], Bs[current ^ 1], A_register, B_register, warp_row_offset, warp_col_offset, lane_idx
        );

    __syncthreads();
    
    tc_mma_three_level_pipeline::compute_dot_products<
        BN, BK, MMA_M, MMA_N, MMA_K, NUM_MMA_M, NUM_MMA_N, NUM_MMA_K>(
            A_register, B_register, acc_register
        );

    for (int mma_row_idx{0}; mma_row_idx < NUM_MMA_M; ++mma_row_idx) {
        for (int mma_col_idx{0}; mma_col_idx < NUM_MMA_N; ++mma_col_idx) {
            float C_register[4];

            uint const C_row_offset{warp_row_offset + mma_row_idx * MMA_M};
            uint const C_col_offset{warp_col_offset + mma_col_idx * MMA_N};

            float *C_pointer = &C[C_row_offset * N + C_col_offset];
            // Follow the fragment layout described in PTX ISA.
            uint const fragment_row_offset{lane_idx / 4};
            uint const fragment_col_offset{(lane_idx % 4) * 2};

            reinterpret_cast<float2 *>(&C_register[0])[0] =
                reinterpret_cast<float2 *>(&C_pointer[fragment_row_offset * N + fragment_col_offset])[0];
            reinterpret_cast<float2 *>(&C_register[2])[0] =
                reinterpret_cast<float2 *>(&C_pointer[(fragment_row_offset + 8) * N + fragment_col_offset])[0];

            // Compute alpha * (AB) + beta * C.
            C_register[0] = acc_register[mma_row_idx][mma_col_idx][0] * alpha + C_register[0] * beta;
            C_register[1] = acc_register[mma_row_idx][mma_col_idx][1] * alpha + C_register[1] * beta;
            C_register[2] = acc_register[mma_row_idx][mma_col_idx][2] * alpha + C_register[2] * beta;
            C_register[3] = acc_register[mma_row_idx][mma_col_idx][3] * alpha + C_register[3] * beta;
        
            reinterpret_cast<float2 *>(&C_pointer[fragment_row_offset * N + fragment_col_offset])[0] =
                reinterpret_cast<float2 *>(&C_register[0])[0];
            reinterpret_cast<float2 *>(&C_pointer[(fragment_row_offset + 8) * N + fragment_col_offset])[0] =
                reinterpret_cast<float2 *>(&C_register[2])[0];
        }
    }
}
