#pragma once

#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <mma.h>

#include "9_tensor_cores.cuh"


namespace wt_tc_memcpy_async {
    
template <uint const BM, uint const BN, uint const BK, const uint NUM_THREADS, const uint FACTOR>
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
        __pipeline_memcpy_async(
            &As[(A_block_row_idx + A_load_offset) * BK + A_block_col_idx],
            &A[(A_block_row_idx + A_load_offset) * K + A_block_col_idx],
            8 * sizeof(__nv_bfloat16)
        );
    }
        
    for (int B_load_offset{0}; B_load_offset < BK; B_load_offset += stride_B) {
        __pipeline_memcpy_async(
            &Bs[(B_block_row_idx + B_load_offset) * BN + B_block_col_idx],
            &B[(B_block_row_idx + B_load_offset) * N + B_block_col_idx],
            8 * sizeof(__nv_bfloat16)
        );
    }
    __pipeline_commit();
}

};


template <uint const NUM_THREADS, uint const BM, uint const BN, uint const BK,
          uint const WM, uint const WN, uint const WMMA_M, uint const WMMA_N, uint const WMMA_K>
__global__ void __launch_bounds__(NUM_THREADS) tensor_cores_memcpy_async_gemm(
    int M, int N, int K, float alpha,
    __nv_bfloat16 *__restrict__ A,
    __nv_bfloat16 *__restrict__ B,
    float beta,
    float         *__restrict__ C
) {
    __shared__ __nv_bfloat16 As[BM * BK];
    __shared__ __nv_bfloat16 Bs[BK * BN];

    __nv_bfloat16 *As_warp{nullptr}, *Bs_warp{nullptr};

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

        As_warp = &As[warp_row_offset * BK];
        Bs_warp = &Bs[warp_col_offset];
    }

    // Create fragments.
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

    for (int k_offset{0}; k_offset < K; k_offset += BK) {
        // Stage 1: shared-memory stores.
        wt_tc_memcpy_async::load_gmem_to_smem<BM, BN, BK, NUM_THREADS, 3>(
            A, B, N, K,
            As, Bs,
            threadIdx.x
        );
        __pipeline_wait_prior(0);

        A += BK;
        B += BK * N;

        __syncthreads();

        // Stage 2: dot-product computation.
        wt_tc::compute_dot_products<BN, BK, WMMA_M, WMMA_N, WMMA_K, NUM_WMMA_M, NUM_WMMA_N>(
            As_warp, Bs_warp, a_frag, b_frag, acc_frags);
        __syncthreads();
    }

    // Stage 3: epilogue + output stores.
    wt_tc::run_epilogue<WMMA_M, WMMA_N, WMMA_K, NUM_WMMA_M, NUM_WMMA_N>(
        C, acc_frags, N, alpha, beta);
}
