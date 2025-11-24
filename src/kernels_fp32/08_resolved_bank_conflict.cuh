#pragma once


namespace wt_swizzled {

template <uint const BM, uint const BN, uint const BK>
__device__ void load_from_gmem(
    float *__restrict__ A, float *__restrict__ B, int N, int K,
    float *__restrict__ As, float *__restrict__ Bs,
    uint A_block_row_idx, uint A_block_col_idx,
    uint B_block_row_idx, uint B_block_col_idx,
    uint stride_A, uint stride_B
) {
    for (int A_load_offset{0}; A_load_offset < BM; A_load_offset += stride_A) {
        float4 tmp = reinterpret_cast<float4 *>(&As[(A_block_row_idx + A_load_offset) * BK + A_block_col_idx])[0];
        A[(A_block_row_idx + A_load_offset) * K + (A_block_row_idx ^ (A_block_col_idx + 0)) % BK] = tmp.x;
        A[(A_block_row_idx + A_load_offset) * K + (A_block_row_idx ^ (A_block_col_idx + 1)) % BK] = tmp.y;
        A[(A_block_row_idx + A_load_offset) * K + (A_block_row_idx ^ (A_block_col_idx + 2)) % BK] = tmp.z;
        A[(A_block_row_idx + A_load_offset) * K + (A_block_row_idx ^ (A_block_col_idx + 3)) % BK] = tmp.w;
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
__device__ void compute_gemm(
    float *__restrict__ As, float *__restrict__ Bs,
    float *__restrict__ reg_M, float *__restrict__ reg_N, float *__restrict__ out_values,
    uint warp_row_offset, uint warp_col_offset,
    uint thread_row_in_warp, uint thread_col_in_warp
) {
    // Recall that reg_M has a size of WMITER * TM and
    // reg_N has a size of WNITER * TN.
    for (int k{0}; k < BK; ++k) {
        for (int wmiter_idx{0}; wmiter_idx < WMITER; ++wmiter_idx) {
            for (int tm_idx{0}; tm_idx < TM; ++tm_idx) {
                uint const row_idx{warp_row_offset + wmiter_idx * WSUBM + thread_row_in_warp * TM + tm_idx};
                reg_M[wmiter_idx * TM + tm_idx] = As[row_idx * BK + (k ^ row_idx) % BK];
                // reg_M[wmiter_idx * TM + tm_idx] = As[
                //     (warp_row_offset + wmiter_idx * WSUBM + thread_row_in_warp * TM + tm_idx) * BK + k
                // ];
            }
        }

        for (int wniter_idx{0}; wniter_idx < WNITER; ++wniter_idx) {
            for (int tn_idx{0}; tn_idx < TN; ++tn_idx) {
                reg_N[wniter_idx * TN + tn_idx] = Bs[
                    k * BN + (warp_col_offset + wniter_idx * WSUBN + thread_col_in_warp * TN + tn_idx)
                ];
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
__global__ void __launch_bounds__(NUM_THREADS) warptiling_swizzled_smem_gemm(
    int M, int N, int K, float alpha,
    float *__restrict__ A, float *__restrict__ B, float beta, float *__restrict__ C
) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    uint const lane_idx{threadIdx.x % 32};
    uint const warp_idx{threadIdx.x / 32};
    uint const warp_row_offset{(warp_idx / (BN / WN)) * WM};
    uint const warp_col_offset{(warp_idx % (BN / WN)) * WN};

    constexpr uint WMITER{WM * WN / (32 * TM * TN * WNITER)};
    constexpr uint WSUBM{WM / WMITER};
    constexpr uint WSUBN{WN / WNITER};

    uint const thread_col_in_warp{lane_idx % (WSUBN / TN)};
    uint const thread_row_in_warp{lane_idx / (WSUBN / TN)};

    float out_values[WMITER * TM * WNITER * TN] = {0.0f};
    float reg_M[WMITER * TM] = {0.0f};
    float reg_N[WNITER * TN] = {0.0f};

    {
        uint const block_row_offset{blockIdx.y * BM};
        uint const block_col_offset{blockIdx.x * BN};

        A += block_row_offset * K;
        B += block_col_offset;
        C += block_row_offset * N + block_col_offset;
    }

    constexpr uint stride_A{(NUM_THREADS << 2) / BK};
    constexpr uint stride_B{(NUM_THREADS << 2) / BN};

    // For storing into shared memory.
    uint const A_block_row_idx{threadIdx.x / (BK >> 2)};
    uint const A_block_col_idx{(threadIdx.x % (BK >> 2)) << 2};
    uint const B_block_row_idx{threadIdx.x / (BN >> 2)};
    uint const B_block_col_idx{(threadIdx.x % (BN >> 2)) << 2};

    for (int k_offset{0}; k_offset < K; k_offset += BK) {
        wt::load_from_gmem<BM, BN, BK>(
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
        wt_sd::compute_gemm<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
            As, Bs, reg_M, reg_N, out_values, warp_row_offset, warp_col_offset,
            thread_row_in_warp, thread_col_in_warp
        );
        __syncthreads();
    }

    for (int wmiter_idx{0}; wmiter_idx < WMITER; ++wmiter_idx)
        for (int wniter_idx{0}; wniter_idx < WNITER; ++wniter_idx) {
            uint const tile_row_idx{warp_row_offset + wmiter_idx * WSUBM + thread_row_in_warp * TM};
            uint const tile_col_idx{warp_col_offset + wniter_idx * WSUBN + thread_col_in_warp * TN};
            
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
