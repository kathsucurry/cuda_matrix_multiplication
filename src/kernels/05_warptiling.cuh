#pragma once


namespace wt {
    
template <size_t const BM, size_t const BN, size_t const BK>
__device__ void load_from_gmem(
    float *A, float *B, int K,
    float *As, float *Bs, size_t k_offset,
    size_t block_row_offset, size_t block_col_offset,
    size_t A_block_row_idx, size_t A_block_col_idx,
    size_t B_trans_block_row_idx, size_t B_trans_block_col_idx,
    size_t LOAD_ITER_M, size_t LOAD_ITER_N
) {
    for (size_t iter_m{0}; iter_m < LOAD_ITER_M; ++iter_m) {
        // Can also use As[4 * (threadIdx.x + iter_m * blockDim.x)].
        reinterpret_cast<float4 *>(&As[(A_block_row_idx + (BM / LOAD_ITER_M * iter_m)) * BK + A_block_col_idx])[0] =
            reinterpret_cast<float4 *>(&A[
                (block_row_offset + A_block_row_idx + (BM / LOAD_ITER_M * iter_m)) * K +
                A_block_col_idx + k_offset
            ])[0];
    }
    
    for (size_t iter_n{0}; iter_n < LOAD_ITER_N; ++iter_n) {
        float4 tmp{
             reinterpret_cast<float4 *>(
                &B[
                    (block_col_offset + B_trans_block_row_idx + (BN / LOAD_ITER_N * iter_n)) * K +
                    B_trans_block_col_idx + k_offset
                ])[0]
        };
        // Transpose.
        Bs[(B_trans_block_col_idx + 0) * BN + B_trans_block_row_idx + (BN / LOAD_ITER_N * iter_n)] = tmp.x;
        Bs[(B_trans_block_col_idx + 1) * BN + B_trans_block_row_idx + (BN / LOAD_ITER_N * iter_n)] = tmp.y;
        Bs[(B_trans_block_col_idx + 2) * BN + B_trans_block_row_idx + (BN / LOAD_ITER_N * iter_n)] = tmp.z;
        Bs[(B_trans_block_col_idx + 3) * BN + B_trans_block_row_idx + (BN / LOAD_ITER_N * iter_n)] = tmp.w;
    }
}

template <size_t const BN, size_t const BK, size_t const WM, size_t const WN, size_t const TM, size_t const TN>
__device__ void compute_gemm(
    float *As, float *Bs,
    float *reg_M, float *reg_N, float *out_values,
    size_t lane_idx,
    size_t warp_row_offset, size_t warp_col_offset,
    size_t thread_row_offset, size_t thread_col_offset
) {
    for (size_t k{0}; k < BK; ++k) {
        for (size_t tile_y_idx{0}; tile_y_idx < TM; ++tile_y_idx)
            reg_M[tile_y_idx] = As[(warp_row_offset + thread_row_offset + tile_y_idx) * BK + k];
        
        for (size_t tile_x_idx{0}; tile_x_idx < TN; ++tile_x_idx)
            reg_N[tile_x_idx] = Bs[k * BN + (warp_col_offset + thread_col_offset + tile_x_idx)];

#pragma unroll
        for (size_t tile_y_idx{0}; tile_y_idx < TM; ++tile_y_idx) {
#pragma unroll
            for (size_t tile_x_idx{0}; tile_x_idx < TN; ++tile_x_idx) {
                out_values[tile_y_idx * TN + tile_x_idx] += reg_M[tile_y_idx] * reg_N[tile_x_idx];
            }
        }
    }
}

};


/**
 * Corresponds to kernel 10: warptiling.
 */
template <size_t BM, size_t BN, size_t BK, size_t WM, size_t WN, size_t TM, size_t TN>
__global__ void warptiling(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    size_t const lane_idx{threadIdx.x % 32};
    size_t const warp_idx{threadIdx.x / 32};
    size_t const warp_row_offset{(warp_idx / (BN / WN)) * WM};
    size_t const warp_col_offset{(warp_idx % (BN / WN)) * WN};
    size_t const thread_row_offset{(lane_idx / (WN / TN)) * TM};
    size_t const thread_col_offset{(lane_idx % (WN / TN)) * TN};

    float out_values[TM * TN] = {0.0f};
    float reg_M[TM] = {0.0f};
    float reg_N[TN] = {0.0f};

    size_t const block_row_offset{blockIdx.y * BM};
    size_t const block_col_offset{blockIdx.x * BN};

    size_t const LOAD_ITER_M{(BK * BM) / (4 * blockDim.x)};
    size_t const LOAD_ITER_N{(BK * BN) / (4 * blockDim.x)};

    // For storing into shared memory.
    size_t const A_block_row_idx = threadIdx.x / (BK / 4);
    // The transposed row index.
    size_t const B_trans_block_row_idx = threadIdx.x / (BK / 4);

    for (size_t k_offset{0}; k_offset < K; k_offset += BK) {
        size_t const A_block_col_idx{(threadIdx.x % (BK / 4)) * 4};
        size_t const B_trans_block_col_idx{(threadIdx.x % (BK / 4)) * 4};
        wt::load_from_gmem<BM, BN, BK>(
            A, B, K,
            As, Bs, k_offset,
            block_row_offset, block_col_offset,
            A_block_row_idx, A_block_col_idx,
            B_trans_block_row_idx, B_trans_block_col_idx,
            LOAD_ITER_M, LOAD_ITER_N
        );
        __syncthreads();

        // Execute the dot product.
        wt::compute_gemm<BN, BK, WM, WN, TM, TN>(
            As, Bs, reg_M, reg_N, out_values, lane_idx, warp_row_offset, warp_col_offset,
            thread_row_offset, thread_col_offset
        );
        __syncthreads();
    }

    for (size_t tile_y_idx{0}; tile_y_idx < TM; ++tile_y_idx) {
        for (size_t tile_x_idx{0}; tile_x_idx < TN; tile_x_idx += 4) {
            size_t const cell_row_idx{block_row_offset + warp_row_offset + thread_row_offset + tile_y_idx};
            size_t const cell_col_idx{block_col_offset + warp_col_offset + thread_col_offset + tile_x_idx};

            float4 tmp = reinterpret_cast<float4 *>(
                &C[cell_row_idx * N + cell_col_idx]
            )[0];
            tmp.x = alpha * out_values[tile_y_idx * TN + tile_x_idx + 0] + beta * tmp.x;
            tmp.y = alpha * out_values[tile_y_idx * TN + tile_x_idx + 1] + beta * tmp.y;
            tmp.z = alpha * out_values[tile_y_idx * TN + tile_x_idx + 2] + beta * tmp.z;
            tmp.w = alpha * out_values[tile_y_idx * TN + tile_x_idx + 3] + beta * tmp.w;
            reinterpret_cast<float4 *>(
                &C[cell_row_idx * N + cell_col_idx]
            )[0] = tmp;
        }
    }
}
