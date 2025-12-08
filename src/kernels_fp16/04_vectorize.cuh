#pragma once

#include <cuda_bf16.h>

#include "../kernels_fp32/04_vectorize.cuh"


/**
 * Corresponds to kernel 6: vectorize SMEM and GMEM access.
 */
template <uint const NUM_THREADS, uint const BM, uint const BN, uint const BK, uint const TM, uint const TN>
__global__ void __launch_bounds__(NUM_THREADS) vectorize_gemm(
    int M, int N, int K, float alpha,
    __nv_bfloat16 *__restrict__ A, __nv_bfloat16 *__restrict__ B, float beta, float *__restrict__ C
) {
    __shared__ __nv_bfloat16 As[BM * BK];
    __shared__ __nv_bfloat16 Bs[BK * BN];

    float out_values[TM * TN] = {0.0f};
    float reg_M[TM] = {0.0f};
    float reg_N[TN] = {0.0f};

    uint const threadIdx_x{threadIdx.x % (BN / TN)};
    uint const threadIdx_y{threadIdx.x / (BN / TN)};

    {
        uint const block_row_offset{blockIdx.y * BM};
        uint const block_col_offset{blockIdx.x * BN};

        A += block_row_offset * K;
        B += block_col_offset;
        C += block_row_offset * N + block_col_offset;
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
        for (int k{0}; k < BK; ++k) {
            for (int tile_y_idx{0}; tile_y_idx < TM; ++tile_y_idx)
                reg_M[tile_y_idx] = __bfloat162float(As[(threadIdx_y * TM + tile_y_idx) * BK + k]);
            
            for (int tile_x_idx{0}; tile_x_idx < TN; ++tile_x_idx)
                reg_N[tile_x_idx] = __bfloat162float(Bs[k * BN + (threadIdx_x * TN + tile_x_idx)]);


            for (int tile_y_idx{0}; tile_y_idx < TM; ++tile_y_idx) {
                for (int tile_x_idx{0}; tile_x_idx < TN; ++tile_x_idx) {
                    out_values[tile_y_idx * TN + tile_x_idx] +=
                        reg_M[tile_y_idx] * reg_N[tile_x_idx];
                }
            }
        }
        __syncthreads();
    }

    // Stage 3: epilogue; output stores.
    vectorize::run_epilogue<TM, TN>(C, out_values, N, threadIdx_y, threadIdx_x,alpha, beta);
}
