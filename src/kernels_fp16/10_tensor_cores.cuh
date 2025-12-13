#pragma once

#include <cuda_bf16.h>
#include <mma.h>

#include "../kernels_templated/04_vectorize.cuh"


namespace wt_tc {
    
template <uint const BM, uint const BN, uint const BK>
__device__ void load_gmem_to_smem(
    __nv_bfloat16 *__restrict__ A, __nv_bfloat16 *__restrict__ B, int N, int K,
    __nv_bfloat16 *__restrict__ As, __nv_bfloat16 *__restrict__ Bs,
    uint const A_block_row_idx, uint const A_block_col_idx,
    uint const B_block_row_idx, uint const B_block_col_idx,
    uint const stride_A, uint const stride_B
) {
    for (int A_load_offset{0}; A_load_offset < BM; A_load_offset += stride_A) {
        reinterpret_cast<float4 *>(&As[(A_block_row_idx + A_load_offset) * BK + A_block_col_idx])[0] =
            reinterpret_cast<float4 *>(&A[(A_block_row_idx + A_load_offset) * K + A_block_col_idx])[0];
    }
        
    for (int B_load_offset{0}; B_load_offset < BK; B_load_offset += stride_B) {
        reinterpret_cast<float4 *>(&Bs[(B_block_row_idx + B_load_offset) * BN + B_block_col_idx])[0] =
            reinterpret_cast<float4 *>(&B[(B_block_row_idx + B_load_offset) * N + B_block_col_idx])[0];
    }
}


template <uint const BN, uint const BK,
          uint const WMMA_M, uint const WMMA_N, uint const WMMA_K,
          uint const NUM_WMMA_M, uint const NUM_WMMA_N>
__device__ __forceinline__ void compute_dot_products(
    __nv_bfloat16 *__restrict__ As,
    __nv_bfloat16 *__restrict__ Bs,
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, nvcuda::wmma::row_major> a_frag,
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, nvcuda::wmma::row_major> b_frag,
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frags[NUM_WMMA_M][NUM_WMMA_N]
) {
    for (uint wmma_k_offset{0u}; wmma_k_offset < BK; wmma_k_offset += WMMA_K) {
        for (int wmma_row_idx{0}; wmma_row_idx < NUM_WMMA_M; ++wmma_row_idx) {
            nvcuda::wmma::load_matrix_sync(
                a_frag,
                &As[(wmma_row_idx * WMMA_M) * BK + wmma_k_offset],
                BK
            );
            for (int wmma_col_idx{0}; wmma_col_idx < NUM_WMMA_N; ++wmma_col_idx) {
                nvcuda::wmma::load_matrix_sync(
                    b_frag,
                    &Bs[wmma_k_offset * BN + (wmma_col_idx * WMMA_N)],
                    BN
                );

                // Perform matrix multiplication.
                nvcuda::wmma::mma_sync(
                    acc_frags[wmma_row_idx][wmma_col_idx],
                    a_frag,
                    b_frag,
                    acc_frags[wmma_row_idx][wmma_col_idx]
                );
            }
        }
    }
}


template <uint const WMMA_M, uint const WMMA_N, uint const WMMA_K, uint const NUM_WMMA_M, uint const NUM_WMMA_N>
__device__ void run_epilogue(
    float *__restrict__ C,
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frags[NUM_WMMA_M][NUM_WMMA_N],
    uint const N,
    float alpha,
    float beta
) {
    nvcuda::wmma::fragment<
        nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    for (int wmma_row_idx{0}; wmma_row_idx < NUM_WMMA_M; ++wmma_row_idx) {
        for (int wmma_col_idx{0}; wmma_col_idx < NUM_WMMA_N; ++wmma_col_idx) {
            uint const C_row_offset{wmma_row_idx * WMMA_M};
            uint const C_col_offset{wmma_col_idx * WMMA_N};

            nvcuda::wmma::load_matrix_sync(
                c_frag,
                &C[C_row_offset * N + C_col_offset],
                N,
                nvcuda::wmma::mem_row_major
            );

            for (int i{0}; i < c_frag.num_elements; ++i) {
                c_frag.x[i] = alpha * acc_frags[wmma_row_idx][wmma_col_idx].x[i] + beta * c_frag.x[i];
            }

            nvcuda::wmma::store_matrix_sync(
                &C[C_row_offset * N + C_col_offset], c_frag, N, nvcuda::wmma::mem_row_major
            );
        }
    }
}

};


template <uint const NUM_THREADS, uint const BM, uint const BN, uint const BK,
          uint const WM, uint const WN, uint const WMMA_M, uint const WMMA_N, uint const WMMA_K>
__global__ void __launch_bounds__(NUM_THREADS) tensor_cores_gemm(
    int M, int N, int K, float alpha,
    __nv_bfloat16 *__restrict__ A,
    __nv_bfloat16 *__restrict__ B,
    float beta,
    float         *__restrict__ C
) {
    __shared__ __nv_bfloat16 As[BM * BK];
    __shared__ __nv_bfloat16 Bs[BK * BN];

    __nv_bfloat16 *As_warp{nullptr}, *Bs_warp{nullptr};

    constexpr uint NUM_WMMA_M{WM / WMMA_M};
    constexpr uint NUM_WMMA_N{WN / WMMA_N};
    
    {
        uint const warp_idx{threadIdx.x / 32};
        uint const warp_row_offset{(warp_idx / (BN / WN)) * WM};
        uint const warp_col_offset{(warp_idx % (BN / WN)) * WN};

        uint const block_row_offset{blockIdx.y * BM};
        uint const block_col_offset{blockIdx.x * BN};

        A += block_row_offset * K;
        B += block_col_offset;
        C += (block_row_offset + warp_row_offset) * N + (block_col_offset + warp_col_offset);

        As_warp = &As[warp_row_offset * BK];
        Bs_warp = &Bs[warp_col_offset];
    }

    // Declare fragments/registers.
    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<
        nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frags[NUM_WMMA_M][NUM_WMMA_N];

    // Initialize the accumulator fragments.
    for (int wmma_row_idx{0}; wmma_row_idx < NUM_WMMA_M; ++wmma_row_idx)
        for (int wmma_col_idx{0}; wmma_col_idx < NUM_WMMA_N; ++wmma_col_idx)
            nvcuda::wmma::fill_fragment(acc_frags[wmma_row_idx][wmma_col_idx], 0.0f);

    for (int k_offset{0}; k_offset < K; k_offset += BK) {
        // Stage 1: shared-memory stores.
        vectorize::load_gmem_to_smem<__nv_bfloat16, BM, BN, BK, NUM_THREADS>(
            A, B, N, K,
            As, Bs,
            threadIdx.x
        );
        __syncthreads();

        A += BK;
        B += BK * N;

        // Stage 2: dot-product computation.
        wt_tc::compute_dot_products<BN, BK, WMMA_M, WMMA_N, WMMA_K, NUM_WMMA_M, NUM_WMMA_N>(
            As_warp, Bs_warp, a_frag, b_frag, acc_frags);
        __syncthreads();
    }

    // Stage 3: epilogue + output stores.
    wt_tc::run_epilogue<WMMA_M, WMMA_N, WMMA_K, NUM_WMMA_M, NUM_WMMA_N>(
        C, acc_frags, N, alpha, beta);
}
