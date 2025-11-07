#pragma once


namespace wt {
    
template <size_t const BN, size_t const BK>
__device__ void load_from_gmem(
    float *A, float *B, int K,
    float *As, float *Bs, size_t k_offset,
    size_t block_row_offset, size_t block_col_offset,
    size_t A_block_row_idx, size_t A_block_col_idx,
    size_t B_block_trans_row_idx, size_t B_block_trans_col_idx
) {
    reinterpret_cast<float4 *>(&As[A_block_row_idx * BK + A_block_col_idx])[0] =
        reinterpret_cast<float4 *>(&A[(block_row_offset + A_block_row_idx) * K + A_block_col_idx + k_offset])[0];
    
    float4 tmp{
        reinterpret_cast<float4 *>(
            &B[
                (block_col_offset + B_block_trans_row_idx) * K +
                B_block_trans_col_idx + k_offset
            ])[0]
    };

    // Transpose.
    Bs[(B_block_trans_col_idx + 0) * BN + B_block_trans_row_idx] = tmp.x;
    Bs[(B_block_trans_col_idx + 1) * BN + B_block_trans_row_idx] = tmp.y;
    Bs[(B_block_trans_col_idx + 2) * BN + B_block_trans_row_idx] = tmp.z;
    Bs[(B_block_trans_col_idx + 3) * BN + B_block_trans_row_idx] = tmp.w;
}

__device__ void compute_gemm() {

}

};


/**
 * Corresponds to kernel 10: warptiling.
 */
template <size_t BM, size_t BN, size_t BK, size_t WM, size_t WN, size_t TM, size_t TN>
__global__ void warptiling(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const size_t warp_idx = threadIdx.x / 32;
    const size_t warp_col = warp_idx % (BN / WN);
    const size_t warp_row = warp_idx / (BN / WN);

    // constexpr size_t WM_ITER = (WM * WN) / (32 * TM * TN * WN_ITER);
    // constexpr size_t WSUBM = WM / WM_ITER;
    // constexpr size_t WSUBN = WN / WN_ITER;

    


    float out_values[TM * TN] = {0.0f};

    size_t const threadIdx_x{threadIdx.x % (BN / TN)};
    size_t const threadIdx_y{threadIdx.x / (BN / TN)};

    size_t const block_row_offset{blockIdx.y * BM};
    size_t const block_col_offset{blockIdx.x * BN};

    // For storing into shared memory.
    size_t const A_block_row_idx = threadIdx.x / (BK / 4);
    // The transposed row index.
    size_t const B_block_trans_row_idx = threadIdx.x / (BK / 4);

    for (size_t k_offset{0}; k_offset < K; k_offset += BK) {
        size_t const A_block_col_idx{(threadIdx.x % (BK / 4)) * 4};
        size_t const B_block_trans_col_idx{(threadIdx.x % (BK / 4)) * 4};
        wt::load_from_gmem<BN, BK>(
            A, B, K,
            As, Bs, k_offset,
            block_row_offset, block_col_offset,
            A_block_row_idx, A_block_col_idx,
            B_block_trans_row_idx, B_block_trans_col_idx
        );
        __syncthreads();

        // Execute the dot product.
        for (size_t k{0}; k < BK; ++k) {
            for (size_t tile_y_idx{0}; tile_y_idx < TM; ++tile_y_idx) {
                for (size_t tile_x_idx{0}; tile_x_idx < TN; ++tile_x_idx) {
                    out_values[tile_y_idx * TN + tile_x_idx] +=
                        As[(threadIdx_y * TM + tile_y_idx) * BK + k] *
                        Bs[k * BN + (threadIdx_x * TN + tile_x_idx)];
                }
            }
        }
        __syncthreads();
    }

    for (size_t tile_y_idx{0}; tile_y_idx < TM; ++tile_y_idx) {
        for (size_t tile_x_idx{0}; tile_x_idx < TN; tile_x_idx += 4) {
            size_t const cell_row_idx{block_row_offset + threadIdx_y * TM + tile_y_idx};
            size_t const cell_col_idx{block_col_offset + threadIdx_x * TN + tile_x_idx};

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
