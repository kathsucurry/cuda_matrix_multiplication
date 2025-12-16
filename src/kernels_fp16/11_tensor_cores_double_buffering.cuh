#pragma once

#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <mma.h>

#include "9_tensor_cores.cuh"
#include "10_tensor_cores_memcpy_async.cuh"


template <uint const NUM_THREADS, uint const BM, uint const BN, uint const BK,
          uint const WM, uint const WN, uint const WMMA_M, uint const WMMA_N, uint const WMMA_K>
__global__ void __launch_bounds__(NUM_THREADS) tensor_cores_double_buffering_gemm(
    int M, int N, int K, float alpha,
    __nv_bfloat16 *__restrict__ A,
    __nv_bfloat16 *__restrict__ B,
    float beta,
    float         *__restrict__ C
) {
    __shared__ __nv_bfloat16 As[2][BM * BK];
    __shared__ __nv_bfloat16 Bs[2][BK * BN];

    uint As_offset{}, Bs_offset{};

    constexpr uint NUM_WMMA_M{WM / WMMA_M};
    constexpr uint NUM_WMMA_N{WN / WMMA_N};

    {
        uint const warp_idx{threadIdx.x / 32};
        uint const warp_row_offset{(warp_idx / (BN / WN)) * WM};
        uint const warp_col_offset{(warp_idx % (BN / WN)) * WN};

        uint const block_row_offset{blockIdx.y * BM};
        uint const block_col_offset{blockIdx.x * BN};

        A += block_row_offset * K;
        B += block_col_offset;
        C += (block_row_offset + warp_row_offset) * N + (block_col_offset + warp_col_offset);

        As_offset = warp_row_offset * BK;
        Bs_offset = warp_col_offset;
    }

    // Declare fragments.
    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<
        nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frags[NUM_WMMA_M][NUM_WMMA_N];

    // Initialize the accumulator fragments.
    for (int wmma_row_idx{0}; wmma_row_idx < NUM_WMMA_M; ++wmma_row_idx)
        for (int wmma_col_idx{0}; wmma_col_idx < NUM_WMMA_N; ++wmma_col_idx)
            nvcuda::wmma::fill_fragment(acc_frags[wmma_row_idx][wmma_col_idx], 0.0f);

    int8_t current{0};

    // Stage 1: shared-memory stores.
    wt_tc_memcpy_async::load_gmem_to_smem<BM, BN, BK, NUM_THREADS, 3>(
        A, B, N, K,
        As[current], Bs[current],
        threadIdx.x
    );

    for (int k_offset{0}; k_offset < K - BK; k_offset += BK) {
        A += BK;
        B += BK * N;

        wt_tc_memcpy_async::load_gmem_to_smem<BM, BN, BK, NUM_THREADS, 3>(
            A, B, N, K,
            As[current ^ 1], Bs[current ^ 1],
            threadIdx.x
        );
        __pipeline_wait_prior(1);
        __syncthreads();

        // Stage 2: dot-product computation.
        wt_tc::compute_dot_products<BN, BK, WMMA_M, WMMA_N, WMMA_K, NUM_WMMA_M, NUM_WMMA_N>(
            &As[current][As_offset], &Bs[current][Bs_offset], a_frag, b_frag, acc_frags);
        __syncthreads();

        current ^= 1;
    }
    __pipeline_wait_prior(0);
    __syncthreads();

    wt_tc::compute_dot_products<BN, BK, WMMA_M, WMMA_N, WMMA_K, NUM_WMMA_M, NUM_WMMA_N>(
        &As[current][As_offset], &Bs[current][Bs_offset], a_frag, b_frag, acc_frags);

    // Stage 3: epilogue + output stores.
    wt_tc::run_epilogue<WMMA_M, WMMA_N, WMMA_K, NUM_WMMA_M, NUM_WMMA_N>(
        C, acc_frags, N, alpha, beta);
}
