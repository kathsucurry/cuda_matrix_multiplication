#pragma once

#include "04_vectorize.cuh"
#include "../traits.cuh"


namespace wt {
    
template <typename T, uint const BN, uint const BK, uint const WM, uint const WN, uint const TM, uint const TN>
__device__ void compute_gemm(
    T const *__restrict__ As,
    T const *__restrict__ Bs,
    float   *__restrict__ reg_M,
    float   *__restrict__ reg_N,
    float *out_values
) {
    using traits = float_traits<T>;

    for (int k{0}; k < BK; ++k) {
        for (int tile_y_idx{0}; tile_y_idx < TM; ++tile_y_idx)
            reg_M[tile_y_idx] = traits::to_compute(As[tile_y_idx * BK + k]);
        
        for (int tile_x_idx{0}; tile_x_idx < TN; ++tile_x_idx)
            reg_N[tile_x_idx] = traits::to_compute(Bs[k * BN + tile_x_idx]);

        for (int tile_y_idx{0}; tile_y_idx < TM; ++tile_y_idx) {
            for (int tile_x_idx{0}; tile_x_idx < TN; ++tile_x_idx) {
                out_values[tile_y_idx * TN + tile_x_idx] += reg_M[tile_y_idx] * reg_N[tile_x_idx];
            }
        }
    }
}


template <uint const TM, uint const TN>
__device__ void run_epilogue(
    float *__restrict__ C,
    float const *__restrict__ out_values,
    uint const N,
    float alpha,
    float beta
) {
    for (int tile_y_idx{0}; tile_y_idx < TM; ++tile_y_idx) {
        for (int tile_x_idx{0}; tile_x_idx < TN; tile_x_idx += 4) {
            float4 tmp = reinterpret_cast<float4 *>(
                &C[tile_y_idx * N + tile_x_idx]
            )[0];

            tmp.x = alpha * out_values[tile_y_idx * TN + tile_x_idx + 0] + beta * tmp.x;
            tmp.y = alpha * out_values[tile_y_idx * TN + tile_x_idx + 1] + beta * tmp.y;
            tmp.z = alpha * out_values[tile_y_idx * TN + tile_x_idx + 2] + beta * tmp.z;
            tmp.w = alpha * out_values[tile_y_idx * TN + tile_x_idx + 3] + beta * tmp.w;
            reinterpret_cast<float4 *>(
                &C[tile_y_idx * N + tile_x_idx]
            )[0] = tmp;
        }
    }
}

};


/**
 * Corresponds to kernel 10: warptiling (no subdivision/usage of WMITER/WNITER).
 */
template <uint const NUM_THREADS, uint const BM, uint const BN, uint const BK, uint
    const WM, uint const WN, uint const TM, uint const TN>
__global__ void __launch_bounds__(NUM_THREADS) warptiling_gemm(
    int M, int N, int K, float alpha,
    float *__restrict__ A, float *__restrict__ B, float beta, float *__restrict__ C
) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    float out_values[TM * TN] = {0.0f};
    float reg_M[TM] = {0.0f};
    float reg_N[TN] = {0.0f};

    float *As_warp{nullptr}, *Bs_warp{nullptr};

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
        vectorize::load_from_gmem<float, BM, BN, BK, NUM_THREADS, 2>(
            A, B, N, K,
            As, Bs,
            threadIdx.x
        );
        __syncthreads();

        A += BK;
        B += BK * N;

        // Stage 2: dot-product computation.
        wt::compute_gemm<float, BN, BK, WM, WN, TM, TN>(
            As_warp, Bs_warp, reg_M, reg_N, out_values
        );
        __syncthreads();
    }

    // Stage 3: epilogue; output stores.
    wt::run_epilogue<TM, TN>(C, out_values, N, alpha, beta);
}
