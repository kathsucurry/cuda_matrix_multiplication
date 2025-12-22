#pragma once

#include <cuda_bf16.h>
#include <mma.h>


namespace tc_swizzled {
    
template <uint const BM, uint const BN, uint const BK,
          uint const NUM_THREADS, uint const FACTOR,
          uint const SW_B_YYY_MASK, uint const SW_B_SHIFT>
__device__ __forceinline__ void load_gmem_to_smem(
    __nv_bfloat16 *__restrict__ A, __nv_bfloat16 *__restrict__ B, int N, int K,
    __nv_bfloat16 *__restrict__ As, __nv_bfloat16 *__restrict__ Bs,
    uint const thread_idx
) {
    constexpr uint stride_A{(NUM_THREADS << FACTOR) / BK};
    constexpr uint stride_B{(NUM_THREADS << FACTOR) / BN};

    uint const A_block_row_idx{thread_idx / (BK >> FACTOR)};
    uint const A_block_col_idx{(thread_idx % (BK >> FACTOR)) << FACTOR};
    uint const B_block_row_idx{thread_idx / (BN >> FACTOR)};
    uint const B_block_col_idx{(thread_idx % (BN >> FACTOR)) << FACTOR};

    for (int A_load_offset{0}; A_load_offset < BM; A_load_offset += stride_A) {
        uint const offset{(A_block_row_idx + A_load_offset) * BK + A_block_col_idx};
        __pipeline_memcpy_async(
            &As[offset],
            &A[(A_block_row_idx + A_load_offset) * K + A_block_col_idx],
            8 * sizeof(__nv_bfloat16)
        );
    }
        
    for (int B_load_offset{0}; B_load_offset < BK; B_load_offset += stride_B) {
        uint const offset{(B_block_row_idx + B_load_offset) * BN + B_block_col_idx};
        __pipeline_memcpy_async(
            &Bs[offset ^ ((offset & SW_B_YYY_MASK) >> SW_B_SHIFT)],
            &B[(B_block_row_idx + B_load_offset) * N + B_block_col_idx],
            8 * sizeof(__nv_bfloat16)
        );
    }
    __pipeline_commit();
}

};


template <uint const NUM_THREADS,
          uint const BM, uint const BN, uint const BK,
          uint const WM, uint const WN>
__global__ void __launch_bounds__(NUM_THREADS) tensor_cores_mma_swizzled_gemm(
    int M, int N, int K, float alpha,
    __nv_bfloat16 *__restrict__ A,
    __nv_bfloat16 *__restrict__ B,
    float beta,
    float         *__restrict__ C
) {
    __shared__ __nv_bfloat16 As[BM * BK];
    __shared__ __nv_bfloat16 Bs[BK * BN];

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

    constexpr uint SW_B_SHIFT{6 - ilog2<MMA_N>()}; // log2(64) - base
    constexpr uint SW_B_YYY_MASK{((1 << (ilog2<64 / MMA_N>())) - 1) << (ilog2<MMA_N>() + SW_B_SHIFT)};

    constexpr uint NUM_MMA_M{WM / MMA_M};
    constexpr uint NUM_MMA_N{WN / MMA_N};

    // Define register storages; pack in uint32_t to be reinterpreted as .bf16 later.
    // Multiplicand A: a vector expression containing four .f16x2 registers, with each register
    // containing two .bf16 elements from the matrix A.
    uint32_t A_register[4];
    // Multiplicand B: a vector expression containing two .f16x2 registers, with each register
    // containing two .bf16 elements from the matrix B.
    uint32_t B_register[2];
    // Accumulator: a vector expression of 4 .f32 registers; initialize with 0s.
    float acc_register[NUM_MMA_M][NUM_MMA_N][4] = {0.0f};

    for (int k_offset{0}; k_offset < K; k_offset += BK) {
        // Stage 1: shared-memory stores.
        tc_swizzled::load_gmem_to_smem<
            BM, BN, BK, NUM_THREADS, 3, SW_B_YYY_MASK, SW_B_SHIFT
        >(
            A, B, N, K,
            As, Bs,
            threadIdx.x
        );
        __pipeline_wait_prior(0);
        __syncthreads();

        A += BK;
        B += BK * N;

        // Execute the dot product.
        for (uint mma_k_offset{0u}; mma_k_offset < BK; mma_k_offset += MMA_K) {
            // Load A elements from the shared memory to register.
            for (int mma_row_idx{0}; mma_row_idx < NUM_MMA_M; ++mma_row_idx) {
                {
                    uint32_t shared_A_pointer = static_cast<uint32_t>(
                        __cvta_generic_to_shared(&As[
                            (warp_row_offset + mma_row_idx * MMA_M + (lane_idx % MMA_M)) * BK +
                            mma_k_offset + (lane_idx / MMA_M) * 8
                        ]));
                    asm volatile (
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                        "{%0, %1, %2, %3}, [%4];"
                        : "=r"(A_register[0]), "=r"(A_register[1]), "=r"(A_register[2]), "=r"(A_register[3])
                        : "r"(shared_A_pointer)
                    );
                }

                // Load B elements from the shared memory to register.
                for (int mma_col_idx{0}; mma_col_idx < NUM_MMA_N; ++mma_col_idx) {
                    {
                        uint const offset{
                            (mma_k_offset + (lane_idx % MMA_K)) * BN +
                            (warp_col_offset + mma_col_idx * MMA_N)};
                        uint32_t shared_B_pointer = static_cast<uint32_t>(
                            __cvta_generic_to_shared(&Bs[offset ^ ((offset & SW_B_YYY_MASK) >> SW_B_SHIFT)]));
                        asm volatile (
                            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
                            "{%0, %1}, [%2];"
                            : "=r"(B_register[0]), "=r"(B_register[1])
                            : "r"(shared_B_pointer)
                        );
                    }

                    // Perform MMA.
                    asm volatile (
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%10, %11, %12, %13};"
                        : "=f"(acc_register[mma_row_idx][mma_col_idx][0]), "=f"(acc_register[mma_row_idx][mma_col_idx][1]),
                          "=f"(acc_register[mma_row_idx][mma_col_idx][2]), "=f"(acc_register[mma_row_idx][mma_col_idx][3])
                        : "r"(A_register[0]), "r"(A_register[1]), "r"(A_register[2]), "r"(A_register[3]),
                          "r"(B_register[0]), "r"(B_register[1]),
                          "f"(acc_register[mma_row_idx][mma_col_idx][0]), "f"(acc_register[mma_row_idx][mma_col_idx][1]),
                          "f"(acc_register[mma_row_idx][mma_col_idx][2]), "f"(acc_register[mma_row_idx][mma_col_idx][3])
                    );                 
                }
            }
        }
        __syncthreads();
    }

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
