#pragma once

#include "06_warptiling_subdivided.cuh"


namespace wt_transposed_As {

template <uint const BM, uint const BN, uint const BK, uint const NUM_THREADS, uint const FACTOR>
__device__ void load_gmem_to_smem(
    float *__restrict__ A, float *__restrict__ B, int K, int N,
    float *__restrict__ As, float *__restrict__ Bs,
    uint const thread_idx
) {
    constexpr uint stride_A{(NUM_THREADS << FACTOR) / BK};
    constexpr uint stride_B{(NUM_THREADS << FACTOR) / BN};

    // For storing into shared memory.
    uint const A_block_row_idx{thread_idx / (BK >> FACTOR)};
    uint const A_block_col_idx{(thread_idx % (BK >> FACTOR)) << FACTOR};
    uint const B_block_row_idx{thread_idx / (BN >> FACTOR)};
    uint const B_block_col_idx{(thread_idx % (BN >> FACTOR)) << FACTOR};

    for (int A_load_offset{0}; A_load_offset < BM; A_load_offset += stride_A) {
        // Store in transposed As.
        float4 tmp = reinterpret_cast<float4 *>(&A[(A_block_row_idx + A_load_offset) * K + A_block_col_idx])[0];

        As[(A_block_col_idx + 0) * BM + A_block_row_idx + A_load_offset] = tmp.x;
        As[(A_block_col_idx + 1) * BM + A_block_row_idx + A_load_offset] = tmp.y;
        As[(A_block_col_idx + 2) * BM + A_block_row_idx + A_load_offset] = tmp.z;
        As[(A_block_col_idx + 3) * BM + A_block_row_idx + A_load_offset] = tmp.w;
    }
        
    for (int B_load_offset{0}; B_load_offset < BK; B_load_offset += stride_B) {
        reinterpret_cast<float4 *>(&Bs[(B_block_row_idx + B_load_offset) * BN + B_block_col_idx])[0] =
            reinterpret_cast<float4 *>(&B[(B_block_row_idx + B_load_offset) * N + B_block_col_idx])[0];
    }
}

template <uint const BM, uint const BN, uint const BK,
            uint const WM, uint const WN, 
            uint const WMITER, uint const WNITER,
            uint const WSUBM, uint const WSUBN, 
            uint const TM, uint const TN>
__device__ void compute_dot_products(
    float *__restrict__ As, float *__restrict__ Bs,
    float *__restrict__ reg_M, float *__restrict__ reg_N, float *__restrict__ out_values
) {
    // Recall that reg_M has a size of WMITER * TM and
    // reg_N has a size of WNITER * TN.
    for (int k{0}; k < BK; ++k) {
        for (int wmiter_idx{0}; wmiter_idx < WMITER; ++wmiter_idx) {
            for (int tm_idx{0}; tm_idx < TM; tm_idx += 4) {
                reinterpret_cast<float4 *>(&reg_M[wmiter_idx * TM + tm_idx])[0] =
                    reinterpret_cast<float4 *>(&As[k * BM + (wmiter_idx * WSUBM + tm_idx)])[0];
            }
        }

        for (int wniter_idx{0}; wniter_idx < WNITER; ++wniter_idx) {
            for (int tn_idx{0}; tn_idx < TN; tn_idx += 4) {
                reinterpret_cast<float4 *>(&reg_N[wniter_idx * TN + tn_idx])[0] =
                    reinterpret_cast<float4 *>(&Bs[k * BN + (wniter_idx * WSUBN + tn_idx)])[0];
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

};


template <uint const NUM_THREADS, uint const BM, uint const BN, uint const BK,
    uint const WM, uint const WN, uint const WNITER, uint const TM, uint const TN>
__global__ void __launch_bounds__(NUM_THREADS) warptiling_transposed_a_gemm(
    int M, int N, int K, float alpha,
    float *__restrict__ A, float *__restrict__ B, float beta, float *__restrict__ C
) {
    __shared__ float As[BK * BM];
    __shared__ float Bs[BK * BN];
    
    constexpr uint WMITER{WM * WN / (32 * TM * TN * WNITER)};
    constexpr uint WSUBM{WM / WMITER};
    constexpr uint WSUBN{WN / WNITER};

    float out_values[WMITER * TM * WNITER * TN] = {0.0f};
    float reg_M[WMITER * TM] = {0.0f};
    float reg_N[WNITER * TN] = {0.0f};

    float *As_warp{nullptr}, *Bs_warp{nullptr};
    
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
        
        As_warp = &As[warp_row_offset + thread_row_in_warp * TM];
        Bs_warp = &Bs[warp_col_offset + thread_col_in_warp * TN];
    }

    for (int k_offset{0}; k_offset < K; k_offset += BK) {
        // Stage 1: shared-memory stores.
        wt_transposed_As::load_gmem_to_smem<BM, BN, BK, NUM_THREADS, 2>(
            A, B, N, K,
            As, Bs,
            threadIdx.x
        );
        __syncthreads();

        A += BK;
        B += BK * N;

        // Stage 2: dot-product computation.
        wt_transposed_As::compute_dot_products<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
            As_warp, Bs_warp, reg_M, reg_N, out_values
        );
        __syncthreads();
    }

    // Stage 3: epilogue + output stores.
    wt_sd::run_epilogue<WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
        C, out_values, N, alpha, beta
    );     
}
