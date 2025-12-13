#pragma once

#include "../kernels_templated/04_vectorize.cuh"


/**
 * Corresponds to kernel 10: warptiling (no subdivision/usage of WMITER/WNITER).
 */
template <typename T, uint const NUM_THREADS, uint const BM, uint const BN, uint const BK, uint
    const WM, uint const WN, uint const TM, uint const TN>
__global__ void __launch_bounds__(NUM_THREADS) warptiling_gemm(
    int M, int N, int K, 
    float   alpha,
    T       *__restrict__ A,
    T       *__restrict__ B,
    float   beta,
    float   *__restrict__ C
) {
    __shared__ T As[BM * BK];
    __shared__ T Bs[BK * BN];

    float out_values[TM * TN] = {0.0f};
    float reg_M[TM] = {0.0f};
    float reg_N[TN] = {0.0f};

    T *As_warp{nullptr}, *Bs_warp{nullptr};

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
        vectorize::load_gmem_to_smem<T, BM, BN, BK, NUM_THREADS>(
            A, B, N, K,
            As, Bs,
            threadIdx.x
        );
        __syncthreads();

        A += BK;
        B += BK * N;

        // Stage 2: dot-product computation.
        vectorize::compute_dot_products<T, BN, BK, TM, TN>(
            As_warp, Bs_warp, reg_M, reg_N, out_values
        );
        __syncthreads();
    }

    // Stage 3: epilogue + output stores.
    vectorize::run_epilogue<TM, TN>(C, out_values, N, alpha, beta);
}
