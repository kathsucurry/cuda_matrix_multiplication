#pragma once

#include <cuda_bf16.h>

#include "../kernels_fp32/04_vectorize.cuh"
#include "../kernels_fp32/05_warptiling.cuh"


/**
 * Corresponds to kernel 10: warptiling (no subdivision/usage of WMITER/WNITER).
 */
template <uint const NUM_THREADS, uint const BM, uint const BN, uint const BK, uint
    const WM, uint const WN, uint const TM, uint const TN>
__global__ void __launch_bounds__(NUM_THREADS) warptiling_gemm(
    int M, int N, int K, float alpha,
    __nv_bfloat16 *__restrict__ A,
    __nv_bfloat16 *__restrict__ B,
    float beta,
    float         *__restrict__ C
) {
    __shared__ __nv_bfloat16 As[BM * BK];
    __shared__ __nv_bfloat16 Bs[BK * BN];

    float out_values[TM * TN] = {0.0f};
    float reg_M[TM] = {0.0f};
    float reg_N[TN] = {0.0f};

    __nv_bfloat16 *As_warp{nullptr}, *Bs_warp{nullptr};

    {
        uint const lane_idx{threadIdx.x % 32};
        uint const warp_idx{threadIdx.x / 32};
        uint const warp_row_offset{(warp_idx / (BN / WN)) * WM};
        uint const warp_col_offset{(warp_idx % (BN / WN)) * WN};
        
        uint const thread_row_offset{(lane_idx / (WN / TN)) * TM};
        uint const thread_col_offset{(lane_idx % (WN / TN)) * TN};

        uint const block_row_offset{blockIdx.y * BM};
        uint const block_col_offset{blockIdx.x * BN};

        A += block_row_offset * K;
        B += block_col_offset;
        // We only need C during the epilogue, which is warp and thread-specific.
        C += (block_row_offset + warp_row_offset + thread_row_offset) * N +
             (block_col_offset + warp_col_offset + thread_col_offset);

        As_warp = &As[(warp_row_offset + thread_row_offset) * BK];
        Bs_warp = &Bs[(warp_col_offset + thread_col_offset)];
    }

    for (int k_offset{0}; k_offset < K; k_offset += BK) {
        // Stage 1: shared-memory stores.
        vectorize::load_from_gmem<__nv_bfloat16, BM, BN, BK, NUM_THREADS, 3>(
            A, B, N, K,
            As, Bs,
            threadIdx.x
        );
        __syncthreads();

        A += BK;
        B += BK * N;

        // Stage 2: dot-product computation.
        wt::compute_gemm<__nv_bfloat16, BN, BK, WM, WN, TM, TN>(
            As_warp, Bs_warp, reg_M, reg_N, out_values
        );
        __syncthreads();
    }

    // Stage 3: epilogue; output stores.
    wt::run_epilogue<TM, TN>(C, out_values, N, alpha, beta);
}
