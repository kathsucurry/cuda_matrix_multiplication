#pragma once


/**
 * Corresponds to kernel 6: vectorize SMEM and GMEM access.
 */
template <size_t NUM_THREADS, size_t BM, size_t BN, size_t BK, size_t TM, size_t TN>
__global__ void __launch_bounds__(NUM_THREADS) vectorize_gemm(
    int M, int N, int K, float alpha,
    float *__restrict__ A, float *__restrict__ B, float beta, float *__restrict__ C
) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    float out_values[TM * TN] = {0.0f};
    float reg_M[TM] = {0.0f};
    float reg_N[TN] = {0.0f};

    size_t const threadIdx_x{threadIdx.x % (BN / TN)};
    size_t const threadIdx_y{threadIdx.x / (BN / TN)};

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

        for (size_t A_load_offset{0}; A_load_offset < BM; A_load_offset += stride_A) {
            reinterpret_cast<float4 *>(&As[(A_block_row_idx + A_load_offset) * BK + A_block_col_idx])[0] =
                reinterpret_cast<float4 *>(&A[(A_block_row_idx + A_load_offset) * K + A_block_col_idx])[0];
        }

        for (size_t B_load_offset{0}; B_load_offset < BK; B_load_offset += stride_B) {
            reinterpret_cast<float4 *>(&Bs[(B_block_row_idx + B_load_offset) * BN + B_block_col_idx])[0] =
                reinterpret_cast<float4 *>(&B[(B_block_row_idx + B_load_offset) * N + B_block_col_idx])[0];
        }
        __syncthreads();

        A += BK;
        B += BK * N;

        // Execute the dot product.
        for (size_t k{0}; k < BK; ++k) {
            for (size_t tile_y_idx{0}; tile_y_idx < TM; ++tile_y_idx)
                reg_M[tile_y_idx] = As[(threadIdx_y * TM + tile_y_idx) * BK + k];
            
            for (size_t tile_x_idx{0}; tile_x_idx < TN; ++tile_x_idx)
                reg_N[tile_x_idx] = Bs[k * BN + (threadIdx_x * TN + tile_x_idx)];


            for (size_t tile_y_idx{0}; tile_y_idx < TM; ++tile_y_idx) {
                for (size_t tile_x_idx{0}; tile_x_idx < TN; ++tile_x_idx) {
                    out_values[tile_y_idx * TN + tile_x_idx] +=
                        reg_M[tile_y_idx] * reg_N[tile_x_idx];
                }
            }
        }
        __syncthreads();
    }

    for (size_t tile_y_idx{0}; tile_y_idx < TM; ++tile_y_idx) {
        for (size_t tile_x_idx{0}; tile_x_idx < TN; tile_x_idx += 4) {
            size_t const cell_row_idx{threadIdx_y * TM + tile_y_idx};
            size_t const cell_col_idx{threadIdx_x * TN + tile_x_idx};

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
