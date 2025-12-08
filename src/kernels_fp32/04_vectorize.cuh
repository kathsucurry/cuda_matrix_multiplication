#pragma once


namespace vectorize {
    
template <typename T, uint const BM, uint const BN, uint const BK, const uint NUM_THREADS, const uint FACTOR>
__device__ void load_from_gmem(
    T *__restrict__ A, T *__restrict__ B, int N, int K,
    T *__restrict__ As, T *__restrict__ Bs,
    uint const thread_idx
) {
    constexpr uint stride_A{(NUM_THREADS << FACTOR) / BK};
    constexpr uint stride_B{(NUM_THREADS << FACTOR) / BN};

    uint const A_block_row_idx{thread_idx / (BK >> FACTOR)};
    uint const A_block_col_idx{(thread_idx % (BK >> FACTOR)) << FACTOR};
    uint const B_block_row_idx{thread_idx / (BN >> FACTOR)};
    uint const B_block_col_idx{(thread_idx % (BN >> FACTOR)) << FACTOR};

    for (int A_load_offset{0}; A_load_offset < BM; A_load_offset += stride_A) {
        reinterpret_cast<float4 *>(&As[(A_block_row_idx + A_load_offset) * BK + A_block_col_idx])[0] =
            reinterpret_cast<float4 *>(&A[(A_block_row_idx + A_load_offset) * K + A_block_col_idx])[0];
    }
        
    for (int B_load_offset{0}; B_load_offset < BK; B_load_offset += stride_B) {
        reinterpret_cast<float4 *>(&Bs[(B_block_row_idx + B_load_offset) * BN + B_block_col_idx])[0] =
            reinterpret_cast<float4 *>(&B[(B_block_row_idx + B_load_offset) * N + B_block_col_idx])[0];
    }
}


template <uint const TM, uint const TN>
__device__ void run_epilogue(
    float *__restrict__ C,
    float const *__restrict__ out_values,
    uint const N,
    uint const threadIdx_y,
    uint const threadIdx_x,
    float alpha,
    float beta
) {
    for (int tile_y_idx{0}; tile_y_idx < TM; ++tile_y_idx) {
        for (int tile_x_idx{0}; tile_x_idx < TN; tile_x_idx += 4) {
            uint const cell_row_idx{threadIdx_y * TM + tile_y_idx};
            uint const cell_col_idx{threadIdx_x * TN + tile_x_idx};

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

};


/**
 * Corresponds to kernel 6: vectorize SMEM and GMEM access.
 */
template <uint const NUM_THREADS, uint const BM, uint const BN, uint const BK, uint const TM, uint const TN>
__global__ void __launch_bounds__(NUM_THREADS) vectorize_gemm(
    int M, int N, int K, float alpha,
    float *__restrict__ A, float *__restrict__ B, float beta, float *__restrict__ C
) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    float out_values[TM * TN] = {0.0f};
    float reg_M[TM] = {0.0f};
    float reg_N[TN] = {0.0f};

    uint const threadIdx_x{threadIdx.x % (BN / TN)};
    uint const threadIdx_y{threadIdx.x / (BN / TN)};

    {
        uint const block_row_offset{blockIdx.y * BM};
        uint const block_col_offset{blockIdx.x * BN};

        A += block_row_offset * K;
        B += block_col_offset;
        C += block_row_offset * N + block_col_offset;
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
        for (int k{0}; k < BK; ++k) {
            for (int tile_y_idx{0}; tile_y_idx < TM; ++tile_y_idx)
                reg_M[tile_y_idx] = As[(threadIdx_y * TM + tile_y_idx) * BK + k];
            
            for (int tile_x_idx{0}; tile_x_idx < TN; ++tile_x_idx)
                reg_N[tile_x_idx] = Bs[k * BN + (threadIdx_x * TN + tile_x_idx)];


            for (int tile_y_idx{0}; tile_y_idx < TM; ++tile_y_idx) {
                for (int tile_x_idx{0}; tile_x_idx < TN; ++tile_x_idx) {
                    out_values[tile_y_idx * TN + tile_x_idx] +=
                        reg_M[tile_y_idx] * reg_N[tile_x_idx];
                }
            }
        }
        __syncthreads();
    }

    // Stage 3: epilogue; output stores.
    vectorize::run_epilogue<TM, TN>(C, out_values, N, threadIdx_y, threadIdx_x,alpha, beta);
}
