#pragma once


namespace wt_sd {
    
template <size_t const BM, size_t const BN, size_t const BK>
__device__ void load_from_gmem(
    float *A, float *B, int K, int N,
    float *As, float *Bs, size_t k_offset,
    size_t block_row_offset, size_t block_col_offset,
    size_t A_block_row_idx, size_t A_block_col_idx,
    size_t B_block_row_idx, size_t B_block_col_idx,
    size_t LOAD_ITER_M, size_t LOAD_ITER_N
) {
    for (size_t iter_m{0}; iter_m < LOAD_ITER_M; ++iter_m) {
        reinterpret_cast<float4 *>(&As[4 * (threadIdx.x + iter_m * blockDim.x)])[0] =
            reinterpret_cast<float4 *>(&A[
                (block_row_offset + A_block_row_idx + (BM / LOAD_ITER_M * iter_m)) * K +
                A_block_col_idx + k_offset
            ])[0];
    }
        
    for (size_t iter_n{0}; iter_n < LOAD_ITER_N; ++iter_n) {
        reinterpret_cast<float4 *>(&Bs[4 * (threadIdx.x + iter_n * blockDim.x)])[0] =
            reinterpret_cast<float4 *>(&B[
                (B_block_row_idx + k_offset + (BK / LOAD_ITER_N) * iter_n) * N +
                block_col_offset + B_block_col_idx
            ])[0];
    }
}

template <size_t const BM, size_t const BN, size_t const BK,
            size_t const WM, size_t const WN, 
            size_t const WMITER, size_t const WNITER,
            size_t const WSUBM, size_t const WSUBN, 
            size_t const TM, size_t const TN>
__device__ void compute_gemm(
    float *As, float *Bs,
    float *reg_M, float *reg_N, float *out_values,
    size_t lane_idx,
    size_t warp_row_offset, size_t warp_col_offset,
    size_t thread_row_in_warp, size_t thread_col_in_warp
) {
    // Recall that reg_M has a size of WMITER * TM and
    // reg_N has a size of WNITER * TN.
    for (size_t k{0}; k < BK; ++k) {
        for (size_t wmiter_idx{0}; wmiter_idx < WMITER; ++wmiter_idx) {
            for (size_t tm_idx{0}; tm_idx < TM; ++tm_idx) {
                reg_M[wmiter_idx * TM + tm_idx] = As[
                    (warp_row_offset + wmiter_idx * WSUBM + thread_row_in_warp * TM + tm_idx) * BK + k
                ];
            }
        }

        for (size_t wniter_idx{0}; wniter_idx < WNITER; ++wniter_idx) {
            for (size_t tn_idx{0}; tn_idx < TN; ++tn_idx) {
                reg_N[wniter_idx * TN + tn_idx] = Bs[
                    k * BN + (warp_col_offset + wniter_idx * WSUBN + thread_col_in_warp * TN + tn_idx)
                ];
            }
        }

        for (size_t wmiter_idx{0}; wmiter_idx < WMITER; ++wmiter_idx)
            for (size_t wniter_idx{0}; wniter_idx < WNITER; ++wniter_idx)
                for (size_t tm_idx{0}; tm_idx < TM; ++tm_idx)
                    for (size_t tn_idx{0}; tn_idx < TN; ++tn_idx) {
                        out_values[(wmiter_idx * TM + tm_idx) * (WNITER * TN) + wniter_idx * TN + tn_idx] +=
                            reg_M[wmiter_idx * TM + tm_idx] * reg_N[wniter_idx * TN + tn_idx];
                    }
    }
}

};


/**
 * Corresponds to kernel 10: warptiling.
 */
template <size_t BM, size_t BN, size_t BK, size_t WM, size_t WN, size_t WNITER, size_t TM, size_t TN>
__global__ void warptiling_subdivided(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    size_t const lane_idx{threadIdx.x % 32};
    size_t const warp_idx{threadIdx.x / 32};
    size_t const warp_row_offset{(warp_idx / (BN / WN)) * WM};
    size_t const warp_col_offset{(warp_idx % (BN / WN)) * WN};

    constexpr size_t WMITER{WM * WN / (32 * TM * TN * WNITER)};
    constexpr size_t WSUBM{WM / WMITER};
    constexpr size_t WSUBN{WN / WNITER};

    size_t const thread_col_in_warp{lane_idx % (WSUBN / TN)};
    size_t const thread_row_in_warp{lane_idx / (WSUBN / TN)};

    float out_values[WMITER * TM * WNITER * TN] = {0.0f};
    float reg_M[WMITER * TM] = {0.0f};
    float reg_N[WNITER * TN] = {0.0f};

    size_t const block_row_offset{blockIdx.y * BM};
    size_t const block_col_offset{blockIdx.x * BN};

    size_t const LOAD_ITER_M{(BK * BM) / (4 * blockDim.x)};
    size_t const LOAD_ITER_N{(BK * BN) / (4 * blockDim.x)};

    // For storing into shared memory.
    size_t const A_block_row_idx{threadIdx.x / (BK / 4)};
    size_t const A_block_col_idx{(threadIdx.x % (BK / 4)) * 4};
    size_t const B_block_row_idx{threadIdx.x / (BN / 4)};
    size_t const B_block_col_idx{(threadIdx.x % (BN / 4)) * 4};

    for (size_t k_offset{0}; k_offset < K; k_offset += BK) {
        wt_sd::load_from_gmem<BM, BN, BK>(
            A, B, K, N,
            As, Bs, k_offset,
            block_row_offset, block_col_offset,
            A_block_row_idx, A_block_col_idx,
            B_block_row_idx, B_block_col_idx,
            LOAD_ITER_M, LOAD_ITER_N
        );
        __syncthreads();

        // Execute the dot product.
        wt_sd::compute_gemm<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
            As, Bs, reg_M, reg_N, out_values, lane_idx, warp_row_offset, warp_col_offset,
            thread_row_in_warp, thread_col_in_warp
        );
        __syncthreads();
    }

    for (size_t wmiter_idx{0}; wmiter_idx < WMITER; ++wmiter_idx)
        for (size_t wniter_idx{0}; wniter_idx < WNITER; ++wniter_idx) {
            size_t const tile_row_idx{block_row_offset + warp_row_offset + wmiter_idx * WSUBM + thread_row_in_warp * TM};
            size_t const tile_col_idx{block_col_offset + warp_col_offset + wniter_idx * WSUBN + thread_col_in_warp * TN};
            
            for (size_t tm_idx{0}; tm_idx < TM; ++tm_idx)
                for (size_t tn_idx{0}; tn_idx < TN; tn_idx += 4) {
                    size_t const cell_row_idx{tile_row_idx + tm_idx};
                    size_t const cell_col_idx{tile_col_idx + tn_idx};
                    
                    float4 tmp = reinterpret_cast<float4 *>(
                        &C[cell_row_idx * N + cell_col_idx]
                        )[0];
                    size_t const first_out_idx = (wmiter_idx * TM + tm_idx) * (WNITER * TN) + wniter_idx * TN + tn_idx;
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
