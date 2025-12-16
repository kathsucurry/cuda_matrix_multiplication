#pragma once

#include "../traits.cuh"
#include "../kernels_templated/04_vectorize.cuh"


namespace wt_sd {
    
template <typename T,
          uint const BN, uint const BK,
          uint const WMITER, uint const WNITER,
          uint const WSUBM, uint const WSUBN, 
          uint const TM, uint const TN>
__device__ void compute_dot_products(
    T *__restrict__ As, T *__restrict__ Bs,
    float *__restrict__ reg_M, float *__restrict__ reg_N, float *__restrict__ out_values
) {
    using traits = float_traits<T>;

    // Recall that reg_M has a size of WMITER * TM and
    // reg_N has a size of WNITER * TN.
    for (int k{0}; k < BK; ++k) {
        for (int wmiter_idx{0}; wmiter_idx < WMITER; ++wmiter_idx) {
            for (int tm_idx{0}; tm_idx < TM; ++tm_idx) {
                reg_M[wmiter_idx * TM + tm_idx] = traits::to_compute(
                    As[(wmiter_idx * WSUBM + tm_idx) * BK + k]);
            }
        }

        for (int wniter_idx{0}; wniter_idx < WNITER; ++wniter_idx) {
            for (int tn_idx{0}; tn_idx < TN; ++tn_idx) {
                reg_N[wniter_idx * TN + tn_idx] = traits::to_compute(
                    Bs[k * BN + (wniter_idx * WSUBN + tn_idx)]);
            }
        }

        for (int wmiter_idx{0}; wmiter_idx < WMITER; ++wmiter_idx)
            for (int wniter_idx{0}; wniter_idx < WNITER; ++wniter_idx)
                for (int tm_idx{0}; tm_idx < TM; ++tm_idx)
                    for (int tn_idx{0}; tn_idx < TN; ++tn_idx) {
                        out_values[(wmiter_idx * TM + tm_idx) * (WNITER * TN) + wniter_idx * TN + tn_idx] +=
                            reg_M[wmiter_idx * TM + tm_idx] * reg_N[wniter_idx * TN + tn_idx];
                    }
    }
}


template <uint const WMITER, uint const WNITER, uint const WSUBM,
          uint const WSUBN, uint const TM, uint const TN>
__device__ void run_epilogue(
    float *__restrict__ C,
    float const *__restrict__ out_values,
    uint const N,
    float alpha,
    float beta
) {
    for (int wmiter_idx{0}; wmiter_idx < WMITER; ++wmiter_idx)
        for (int wniter_idx{0}; wniter_idx < WNITER; ++wniter_idx) {
            uint const tile_row_idx{wmiter_idx * WSUBM};
            uint const tile_col_idx{wniter_idx * WSUBN};
            
            for (int tm_idx{0}; tm_idx < TM; ++tm_idx)
                for (int tn_idx{0}; tn_idx < TN; tn_idx += 4) {
                    uint const cell_row_idx{tile_row_idx + tm_idx};
                    uint const cell_col_idx{tile_col_idx + tn_idx};
                    
                    float4 tmp = reinterpret_cast<float4 *>(
                        &C[cell_row_idx * N + cell_col_idx]
                        )[0];
                    uint const first_out_idx = (wmiter_idx * TM + tm_idx) * (WNITER * TN) + wniter_idx * TN + tn_idx;
                    tmp.x = alpha * out_values[first_out_idx + 0] + beta * tmp.x;
                    tmp.y = alpha * out_values[first_out_idx + 1] + beta * tmp.y;
                    tmp.z = alpha * out_values[first_out_idx + 2] + beta * tmp.z;
                    tmp.w = alpha * out_values[first_out_idx + 3] + beta * tmp.w;
                    reinterpret_cast<float4 *>(
                        &C[cell_row_idx * N + cell_col_idx]
                    )[0] = tmp;
                }
        }       
}

};


/**
 * Corresponds to kernel 10: warptiling.
 */
template <typename T, uint const NUM_THREADS, uint const BM, uint const BN, uint const BK,
    uint const WM, uint const WN, uint const WNITER, uint const TM, uint const TN>
__global__ void __launch_bounds__(NUM_THREADS) warptiling_subdivided_gemm(
    int M, int N, int K, float alpha,
    T *__restrict__ A, T *__restrict__ B, float beta, float *__restrict__ C
) {
    __shared__ T As[BM * BK];
    __shared__ T Bs[BK * BN];

    constexpr uint WMITER{WM * WN / (32 * TM * TN * WNITER)};
    constexpr uint WSUBM{WM / WMITER};
    constexpr uint WSUBN{WN / WNITER};

    float out_values[WMITER * TM * WNITER * TN] = {0.0f};
    float reg_M[WMITER * TM] = {0.0f};
    float reg_N[WNITER * TN] = {0.0f};

    T *As_warp{nullptr}, *Bs_warp{nullptr};

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
        
        As_warp = &As[(warp_row_offset + thread_row_in_warp * TM) * BK];
        Bs_warp = &Bs[(warp_col_offset + thread_col_in_warp * TN)];
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
        wt_sd::compute_dot_products<T, BN, BK, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
            As_warp, Bs_warp, reg_M, reg_N, out_values
        );
        __syncthreads();
    }

    // Stage 3: epilogue + output stores.
    wt_sd::run_epilogue<WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
        C, out_values, N, alpha, beta
    );
}
