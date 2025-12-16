#pragma once

#include <cuda_pipeline.h>

#include "../kernels_templated/06_warptiling_subdivided.cuh"


namespace wt_memcpy_async {
    
template <uint const BM, uint const BN, uint const BK>
__device__ void load_gmem_to_smem(
    float *__restrict__ A, float *__restrict__ B, int N, int K,
    float *__restrict__ As, float *__restrict__ Bs,
    uint A_block_row_idx, uint A_block_col_idx,
    uint B_block_row_idx, uint B_block_col_idx,
    uint stride_A, uint stride_B
) {
    for (int A_load_offset{0}; A_load_offset < BM; A_load_offset += stride_A) {
        __pipeline_memcpy_async(
            &As[(A_block_row_idx + A_load_offset) * BK + A_block_col_idx],
            &A[(A_block_row_idx + A_load_offset) * K + A_block_col_idx],
            4 * sizeof(float)
        );
    }
        
    for (int B_load_offset{0}; B_load_offset < BK; B_load_offset += stride_B) {
        __pipeline_memcpy_async(
            &Bs[(B_block_row_idx + B_load_offset) * BN + B_block_col_idx],
            &B[(B_block_row_idx + B_load_offset) * N + B_block_col_idx],
            4 * sizeof(float)
        );
    }
    __pipeline_commit();
}

};


template <uint const NUM_THREADS, uint const BM, uint const BN, uint const BK,
    uint const WM, uint const WN, uint const WNITER, uint const TM, uint const TN>
__global__ void __launch_bounds__(NUM_THREADS) double_buffering_gemm(
    int M, int N, int K, float alpha,
    float *__restrict__ A, float *__restrict__ B, float beta, float *__restrict__ C
) {
    __shared__ float As[2][BM * BK];
    __shared__ float Bs[2][BK * BN];

    constexpr uint WMITER{WM * WN / (32 * TM * TN * WNITER)};
    constexpr uint WSUBM{WM / WMITER};
    constexpr uint WSUBN{WN / WNITER};

    float out_values[WMITER * TM * WNITER * TN] = {0.0f};
    float reg_M[WMITER * TM] = {0.0f};
    float reg_N[WNITER * TN] = {0.0f};

    // Offset As and Bs given warp and thread indices.
    uint As_offset{}, Bs_offset{};

    {
        uint const lane_idx{threadIdx.x % 32};
        uint const warp_idx{threadIdx.x / 32};
        uint const warp_row_offset{(warp_idx / (BN / WN)) * WM};
        uint const warp_col_offset{(warp_idx % (BN / WN)) * WN};

        uint const thread_col_in_warp{lane_idx % (WSUBN / TN)};
        uint const thread_row_in_warp{lane_idx / (WSUBN / TN)};

        uint const block_row_offset{blockIdx.y * BM};
        uint const block_col_offset{blockIdx.x * BN};

        A += block_row_offset * K;
        B += block_col_offset;
        // We only need C during the epilogue, which is warp and thread-specific.
        C += (block_row_offset + warp_row_offset + thread_row_in_warp * TM) * N +
             (block_col_offset + warp_col_offset + thread_col_in_warp * TN);

        As_offset = (warp_row_offset + thread_row_in_warp * TM) * BK;
        Bs_offset = warp_col_offset + thread_col_in_warp * TN;
    }

    constexpr uint stride_A{(NUM_THREADS << 2) / BK};
    constexpr uint stride_B{(NUM_THREADS << 2) / BN};

    // For storing into shared memory.
    uint const A_block_row_idx{threadIdx.x / (BK >> 2)};
    uint const A_block_col_idx{(threadIdx.x % (BK >> 2)) << 2};
    uint const B_block_row_idx{threadIdx.x / (BN >> 2)};
    uint const B_block_col_idx{(threadIdx.x % (BN >> 2)) << 2};

    int current{0};

    // Stage 1: shared-memory stores.
    wt_memcpy_async::load_gmem_to_smem<BM, BN, BK>(
        A, B, N, K,
        As[current], Bs[current],
        A_block_row_idx, A_block_col_idx,
        B_block_row_idx, B_block_col_idx,
        stride_A, stride_B
    );

    for (int k_offset{0}; k_offset < K - BK; k_offset += BK) {
        A += BK;
        B += BK * N;

        // Stage 1: shared-memory stores.
        wt_memcpy_async::load_gmem_to_smem<BM, BN, BK>(
            A, B, N, K,
            As[current ^ 1], Bs[current ^ 1],
            A_block_row_idx, A_block_col_idx,
            B_block_row_idx, B_block_col_idx,
            stride_A, stride_B
        );
        __pipeline_wait_prior(1);
        __syncthreads();

        // Stage 2: dot-product computation.
        wt_sd::compute_dot_products<float, BN, BK, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
            &As[current][As_offset], &Bs[current][Bs_offset], reg_M, reg_N, out_values
        );
        __syncthreads();

        current ^= 1;
    }
    __pipeline_wait_prior(0);
    __syncthreads();

    wt_sd::compute_dot_products<float, BN, BK, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
        &As[current][As_offset], &Bs[current][Bs_offset], reg_M, reg_N, out_values
    );

    // Stage 3: epilogue + output stores.
    wt_sd::run_epilogue<WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
        C, out_values, N,
        alpha, beta
    );    
}
