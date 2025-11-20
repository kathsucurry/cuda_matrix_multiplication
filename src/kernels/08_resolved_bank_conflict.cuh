#pragma once


namespace wt_swizzled {

template <size_t const BM, size_t const BN, size_t const BK>
__device__ void load_from_gmem(
    float *__restrict__ A, float *__restrict__ B, int K, int N,
    float *__restrict__ As, float *__restrict__ Bs,
    size_t A_block_row_idx, size_t A_block_col_idx,
    size_t B_block_row_idx, size_t B_block_col_idx,
    size_t stride_A, size_t stride_B
) {
    for (size_t A_load_offset{0}; A_load_offset < BM; A_load_offset += stride_A) {
        // Store in transposed As.
        float4 tmp = reinterpret_cast<float4 *>(&A[(A_block_row_idx + A_load_offset) * K + A_block_col_idx])[0];
        size_t const col{A_block_row_idx + A_load_offset};

        As[(A_block_col_idx + 0) * BM + (col ^ (A_block_col_idx + 0)) % BM] = tmp.x;
        As[(A_block_col_idx + 1) * BM + (col ^ (A_block_col_idx + 1)) % BM] = tmp.y;
        As[(A_block_col_idx + 2) * BM + (col ^ (A_block_col_idx + 2)) % BM] = tmp.z;
        As[(A_block_col_idx + 3) * BM + (col ^ (A_block_col_idx + 3)) % BM] = tmp.w;
    }

    for (size_t B_load_offset{0}; B_load_offset < BK; B_load_offset += stride_B) {
        reinterpret_cast<float4 *>(&Bs[(B_block_row_idx + B_load_offset) * BN + B_block_col_idx])[0] =
            reinterpret_cast<float4 *>(&B[(B_block_row_idx + B_load_offset) * N + B_block_col_idx])[0];
    }
}

template <size_t const BM, size_t const BN, size_t const BK,
            size_t const WM, size_t const WN, 
            size_t const WMITER, size_t const WNITER,
            size_t const WSUBM, size_t const WSUBN, 
            size_t const TM, size_t const TN>
__device__ void compute_gemm(
    float *__restrict__ As, float *__restrict__ Bs,
    float *__restrict__ reg_M, float *__restrict__ reg_N, float *__restrict__ out_values,
    size_t row_offset, size_t col_offset
) {
    // Recall that reg_M has a size of WMITER * TM and
    // reg_N has a size of WNITER * TN.
    for (size_t k{0}; k < BK; ++k) {
        for (size_t wmiter_idx{0}; wmiter_idx < WMITER; ++wmiter_idx) {
            for (size_t tm_idx{0}; tm_idx < TM; ++tm_idx) {
                reg_M[wmiter_idx * TM + tm_idx] = As[
                    k * BM + ((row_offset + wmiter_idx * WSUBM + tm_idx) ^ k) % BM
                ];
            }
        }

        for (size_t wniter_idx{0}; wniter_idx < WNITER; ++wniter_idx) {
            for (size_t tn_idx{0}; tn_idx < TN; ++tn_idx) {
                reg_N[wniter_idx * TN + tn_idx] = Bs[
                    k * BN + (col_offset + wniter_idx * WSUBN + tn_idx)
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
template <size_t NUM_THREADS, size_t BM, size_t BN, size_t BK, size_t WM, size_t WN, size_t WNITER, size_t TM, size_t TN>
__global__ void __launch_bounds__(NUM_THREADS) warptiling_swizzled_smem_gemm(
    int M, int N, int K, float alpha,
    float *__restrict__ A, float *__restrict__ B, float beta, float *__restrict__ C
) {
    __shared__ float As[BK * BM];
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

    {
        size_t const block_row_offset{blockIdx.y * BM};
        size_t const block_col_offset{blockIdx.x * BN};

        A += block_row_offset * K;
        B += block_col_offset;
        C += block_row_offset * N + block_col_offset;
    }

    constexpr size_t stride_A{(NUM_THREADS << 2) / BK};
    constexpr size_t stride_B{(NUM_THREADS << 2) / BN};

    // For storing into shared memory.
    size_t const A_block_row_idx{threadIdx.x / (BK >> 2)};
    size_t const A_block_col_idx{(threadIdx.x % (BK >> 2)) << 2};
    size_t const B_block_row_idx{threadIdx.x / (BN >> 2)};
    size_t const B_block_col_idx{(threadIdx.x % (BN >> 2)) << 2};

    for (size_t k_offset{0}; k_offset < K; k_offset += BK) {
        wt_swizzled::load_from_gmem<BM, BN, BK>(
            A, B, N, K,
            As, Bs,
            A_block_row_idx, A_block_col_idx,
            B_block_row_idx, B_block_col_idx,
            stride_A, stride_B
        );
        __syncthreads();

        A += BK;
        B += BK * N;

        // Execute the dot product.
        wt_swizzled::compute_gemm<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
            As, Bs, reg_M, reg_N, out_values, 
            warp_row_offset + thread_row_in_warp * TM,
            warp_col_offset + thread_col_in_warp * TN
        );
        __syncthreads();
    }

    for (size_t wmiter_idx{0}; wmiter_idx < WMITER; ++wmiter_idx)
        for (size_t wniter_idx{0}; wniter_idx < WNITER; ++wniter_idx) {
            size_t const tile_row_idx{warp_row_offset + wmiter_idx * WSUBM + thread_row_in_warp * TM};
            size_t const tile_col_idx{warp_col_offset + wniter_idx * WSUBN + thread_col_in_warp * TN};
            
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
