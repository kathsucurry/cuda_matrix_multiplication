#pragma once

#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <mma.h>

#include "10_tensor_cores_memcpy_async.cuh"


namespace wt_tc_three_level_pipeline {

template <uint const BN, uint const BK,
      uint const WMMA_M, uint const WMMA_N, uint const WMMA_K,
      uint const NUM_WMMA_M, uint const NUM_WMMA_N, uint const NUM_WMMA_K>
__device__ __forceinline__ void load_smem_to_regs(
    __nv_bfloat16 *__restrict__ As,
    __nv_bfloat16 *__restrict__ Bs,
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, nvcuda::wmma::row_major> a_frags[NUM_WMMA_M][NUM_WMMA_K],
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, nvcuda::wmma::row_major> b_frags[NUM_WMMA_K][NUM_WMMA_N],
    uint const warp_row_offset, uint const warp_col_offset
) {
    for (uint wmma_k_idx{0}; wmma_k_idx < NUM_WMMA_K; ++wmma_k_idx) {
        uint const wmma_k_offset{wmma_k_idx * WMMA_K};

        // Load A elements from smem to registers.
        for (int wmma_row_idx{0}; wmma_row_idx < NUM_WMMA_M; ++wmma_row_idx) {
            nvcuda::wmma::load_matrix_sync(
                a_frags[wmma_row_idx][wmma_k_idx],
                &As[(warp_row_offset + wmma_row_idx * WMMA_M) * BK + wmma_k_offset],
                BK
            );
        }

        // Load B elements from smem to registers.
        for (int wmma_col_idx{0}; wmma_col_idx < NUM_WMMA_N; ++wmma_col_idx) {
            nvcuda::wmma::load_matrix_sync(
                b_frags[wmma_k_idx][wmma_col_idx],
                &Bs[wmma_k_offset * BN + (warp_col_offset + wmma_col_idx * WMMA_N)],
                BN
            );
        }
    }
}


template <uint const BN, uint const BK,
      uint const WMMA_M, uint const WMMA_N, uint const WMMA_K,
      uint const NUM_WMMA_M, uint const NUM_WMMA_N, uint const NUM_WMMA_K>
__device__ __forceinline__ void compute_dot_products(
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, nvcuda::wmma::row_major> a_frags[NUM_WMMA_M][NUM_WMMA_K],
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, nvcuda::wmma::row_major> b_frags[NUM_WMMA_K][NUM_WMMA_N],
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frags[NUM_WMMA_M][NUM_WMMA_N]
) {
    for (uint wmma_k_idx{0u}; wmma_k_idx < NUM_WMMA_K; ++wmma_k_idx) {
        for (int wmma_row_idx{0}; wmma_row_idx < NUM_WMMA_M; ++wmma_row_idx) {
            for (int wmma_col_idx{0}; wmma_col_idx < NUM_WMMA_N; ++wmma_col_idx) {
                nvcuda::wmma::mma_sync(
                    acc_frags[wmma_row_idx][wmma_col_idx],
                    a_frags[wmma_row_idx][wmma_k_idx],
                    b_frags[wmma_k_idx][wmma_col_idx],
                    acc_frags[wmma_row_idx][wmma_col_idx]
                );
            }
        }
    }
}
};


template <uint const NUM_THREADS, uint const BM, uint const BN, uint const BK,
          uint const WM, uint const WN, uint const WMMA_M, uint const WMMA_N, uint const WMMA_K>
__global__ void __launch_bounds__(NUM_THREADS) tensor_cores_three_level_pipeline_gemm(
    int M, int N, int K, float alpha,
    __nv_bfloat16 *__restrict__ A,
    __nv_bfloat16 *__restrict__ B,
    float beta,
    float         *__restrict__ C
) {
    __shared__ __nv_bfloat16 As[2][BM * BK];
    __shared__ __nv_bfloat16 Bs[2][BK * BN];

    constexpr uint NUM_WMMA_M{WM / WMMA_M};
    constexpr uint NUM_WMMA_N{WN / WMMA_N};
    constexpr uint NUM_WMMA_K{BK / WMMA_K};

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
    
    // Create fragments.
    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, nvcuda::wmma::row_major> a_frags[NUM_WMMA_M][NUM_WMMA_K];
    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, nvcuda::wmma::row_major> b_frags[NUM_WMMA_K][NUM_WMMA_N];
    nvcuda::wmma::fragment<
        nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frags[NUM_WMMA_M][NUM_WMMA_N];
    nvcuda::wmma::fragment<
        nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize the accumulator fragments.
    for (int wmma_row_idx{0}; wmma_row_idx < NUM_WMMA_M; ++wmma_row_idx)
        for (int wmma_col_idx{0}; wmma_col_idx < NUM_WMMA_N; ++wmma_col_idx)
            nvcuda::wmma::fill_fragment(acc_frags[wmma_row_idx][wmma_col_idx], 0.0f);

    // Stage 1: shared-memory stores.
    wt_tc_memcpy_async::load_gmem_to_smem<BM, BN, BK, NUM_THREADS, 3>(
        A, B, N, K,
        As[0], Bs[0],
        threadIdx.x
    );

    A += BK;
    B += BK * N;

    wt_tc_memcpy_async::load_gmem_to_smem<BM, BN, BK, NUM_THREADS, 3>(
        A, B, N, K,
        As[1], Bs[1],
        threadIdx.x
    );

    // Load from smem to registers.
    __pipeline_wait_prior(1);
    __syncthreads();

    wt_tc_three_level_pipeline::load_smem_to_regs<
        BN, BK, WMMA_M, WMMA_N, WMMA_K, NUM_WMMA_M, NUM_WMMA_N, NUM_WMMA_K
    >(
        As[0], Bs[0], a_frags, b_frags, warp_row_offset, warp_col_offset
    );

    int current{0};

    for (int k_offset{0}; k_offset < K - 2 * BK; k_offset += BK) {
        A += BK;
        B += BK * N;

        // All previous loads from smem should finish first.
        __syncthreads();

        // Load from gmem to smem, replacing the current smem.
        wt_tc_memcpy_async::load_gmem_to_smem<BM, BN, BK, NUM_THREADS, 3>(
            A, B, N, K,
            As[current], Bs[current],
            threadIdx.x
        );

        // Execute the dot product.
        wt_tc_three_level_pipeline::compute_dot_products<BN, BK, WMMA_M, WMMA_N, WMMA_K, NUM_WMMA_M, NUM_WMMA_N, NUM_WMMA_K>(
            a_frags, b_frags, acc_frags);

        // Load from smem to registers.
        __pipeline_wait_prior(1);
        __syncthreads();

        wt_tc_three_level_pipeline::load_smem_to_regs<
            BN, BK, WMMA_M, WMMA_N, WMMA_K, NUM_WMMA_M, NUM_WMMA_N, NUM_WMMA_K
        >(
            As[current ^ 1], Bs[current ^ 1],
            a_frags, b_frags,
            warp_row_offset, warp_col_offset
        );

        current ^= 1;
    }
    __pipeline_wait_prior(0);
    __syncthreads();

    wt_tc_three_level_pipeline::compute_dot_products<BN, BK, WMMA_M, WMMA_N, WMMA_K, NUM_WMMA_M, NUM_WMMA_N, NUM_WMMA_K>(
        a_frags, b_frags, acc_frags);

    wt_tc_three_level_pipeline::load_smem_to_regs<
        BN, BK, WMMA_M, WMMA_N, WMMA_K, NUM_WMMA_M, NUM_WMMA_N, NUM_WMMA_K
    >(
        As[current ^ 1], Bs[current ^ 1],
        a_frags, b_frags,
        warp_row_offset, warp_col_offset
    );

    __syncthreads();

    wt_tc_three_level_pipeline::compute_dot_products<BN, BK, WMMA_M, WMMA_N, WMMA_K, NUM_WMMA_M, NUM_WMMA_N, NUM_WMMA_K>(
        a_frags, b_frags, acc_frags);

    // Stage 3: epilogue + output stores.
    for (int wmma_row_idx{0}; wmma_row_idx < NUM_WMMA_M; ++wmma_row_idx) {
        for (int wmma_col_idx{0}; wmma_col_idx < NUM_WMMA_N; ++wmma_col_idx) {
            uint const C_row_offset{warp_row_offset + wmma_row_idx * WMMA_M};
            uint const C_col_offset{warp_col_offset + wmma_col_idx * WMMA_N};

            nvcuda::wmma::load_matrix_sync(
                c_frag,
                &C[C_row_offset * N + C_col_offset],
                N,
                nvcuda::wmma::mem_row_major
            );

            for (int i{0}; i < c_frag.num_elements; ++i) {
                c_frag.x[i] = alpha * acc_frags[wmma_row_idx][wmma_col_idx].x[i] + beta * c_frag.x[i];
            }

            nvcuda::wmma::store_matrix_sync(
                &C[C_row_offset * N + C_col_offset], c_frag, N, nvcuda::wmma::mem_row_major
            );
        }
    }
}
