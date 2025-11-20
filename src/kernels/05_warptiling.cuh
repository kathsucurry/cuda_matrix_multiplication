#pragma once


namespace wt {
    
template <size_t const BM, size_t const BN, size_t const BK>
__device__ void load_from_gmem(
    float *__restrict__ A, float *__restrict__ B, int N, int K,
    float *__restrict__ As, float *__restrict__ Bs,
    size_t A_block_row_idx, size_t A_block_col_idx,
    size_t B_block_row_idx, size_t B_block_col_idx,
    size_t stride_A, size_t stride_B
) {
    for (size_t A_load_offset{0}; A_load_offset < BM; A_load_offset += stride_A) {
        reinterpret_cast<float4 *>(&As[(A_block_row_idx + A_load_offset) * BK + A_block_col_idx])[0] =
            reinterpret_cast<float4 *>(&A[(A_block_row_idx + A_load_offset) * K + A_block_col_idx])[0];
    }
        
    for (size_t B_load_offset{0}; B_load_offset < BK; B_load_offset += stride_B) {
        reinterpret_cast<float4 *>(&Bs[(B_block_row_idx + B_load_offset) * BN + B_block_col_idx])[0] =
            reinterpret_cast<float4 *>(&B[(B_block_row_idx + B_load_offset) * N + B_block_col_idx])[0];
    }
}

template <size_t const BN, size_t const BK, size_t const WM, size_t const WN, size_t const TM, size_t const TN>
__device__ void compute_gemm(
    float *__restrict__ As, float *__restrict__ Bs,
    float *__restrict__ reg_M, float *__restrict__ reg_N, float *out_values,
    size_t warp_row_offset, size_t warp_col_offset,
    size_t thread_row_offset, size_t thread_col_offset
) {
    for (size_t k{0}; k < BK; ++k) {
        for (size_t tile_y_idx{0}; tile_y_idx < TM; ++tile_y_idx)
            reg_M[tile_y_idx] = As[(warp_row_offset + thread_row_offset + tile_y_idx) * BK + k];
        
        for (size_t tile_x_idx{0}; tile_x_idx < TN; ++tile_x_idx)
            reg_N[tile_x_idx] = Bs[k * BN + (warp_col_offset + thread_col_offset + tile_x_idx)];

        for (size_t tile_y_idx{0}; tile_y_idx < TM; ++tile_y_idx) {
            for (size_t tile_x_idx{0}; tile_x_idx < TN; ++tile_x_idx) {
                out_values[tile_y_idx * TN + tile_x_idx] += reg_M[tile_y_idx] * reg_N[tile_x_idx];
            }
        }
    }
}

};


/**
 * Corresponds to kernel 10: warptiling (no subdivision/usage of WMITER/WNITER).
 */
template <size_t NUM_THREADS, size_t BM, size_t BN, size_t BK, size_t WM, size_t WN, size_t TM, size_t TN>
__global__ void __launch_bounds__(NUM_THREADS) warptiling_gemm(
    int M, int N, int K, float alpha,
    float *__restrict__ A, float *__restrict__ B, float beta, float *__restrict__ C
) {
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
        wt::compute_gemm<BN, BK, WM, WN, TM, TN>(
            As, Bs, reg_M, reg_N, out_values, warp_row_offset, warp_col_offset,
            thread_row_offset, thread_col_offset
        );
        __syncthreads();
    }

    for (size_t tile_y_idx{0}; tile_y_idx < TM; ++tile_y_idx) {
        for (size_t tile_x_idx{0}; tile_x_idx < TN; tile_x_idx += 4) {
            size_t const cell_row_idx{warp_row_offset + thread_row_offset + tile_y_idx};
            size_t const cell_col_idx{warp_col_offset + thread_col_offset + tile_x_idx};

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
