#include "runner.cuh"


void run_cublas(
    cublasHandle_t handle, int M, int N, int K, float alpha,
    float *A, float *B, float beta, float *C
) {
    // Recall that:
    // 1) All matrices are stored in row-major order.
    // 2) cuBLAS uses column-major order (let's say C* = A* x B*).
    // So we want (C*)^T = C = (A* x B*)^T
    // --> (B*)^T x (A*)^T --> B x A.
    cublasStatus_t status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
                 K, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT);

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "cublasGemmEx failed: " << status << std::endl;
    }
}


void run_cublas(
    cublasHandle_t handle, int M, int N, int K, float alpha,
    __nv_bfloat16 *A, __nv_bfloat16 *B, float beta, float *C
) {
    cublasStatus_t status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16BF,
                 K, A, CUDA_R_16BF, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT);
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "cublasGemmEx failed: " << status << std::endl;
    }
}


template <typename T>
void run_cublas(
    cublasHandle_t handle, int M, int N, int K, float alpha,
    T *A, T *B, float beta, T *C
) = delete;


void run_sharedmem_block_tiling(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    constexpr uint BLOCK_DIM{32};
    assert((N % BLOCK_DIM == 0) && (M % BLOCK_DIM == 0) && (K % BLOCK_DIM == 0));

    dim3 grid_dim(CEIL_DIV(N, BLOCK_DIM), CEIL_DIV(M, BLOCK_DIM));
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM, 1);

    block_tiling_gemm<BLOCK_DIM><<<grid_dim, block_dim>>>(M, N, K, alpha, A, B, beta, C);
}


void run_2d_thread_coarsening(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    constexpr uint BLOCK_DIM{128};
    // BM: the size of block vertically; BN: the size of block horizontally. 
    constexpr uint BM{BLOCK_DIM}, BN{BLOCK_DIM};
    // TM, TN: the number of rows, columns processed by each thread, respectively.
    constexpr uint TM{8}, TN{8};

    // Recall that BK determines the length of the tile and the shared memory size.
    constexpr uint BK{32};
    // For the sake of simplicity, we make sure that BK * BM is divisible by BLOCK_DIM.
    static_assert(((BK * BM) % BLOCK_DIM == 0) && ((BK * BN) % BLOCK_DIM == 0));
    // The assertion below supports LOAD_ITER_M/N calculation/usage.
    assert((BLOCK_DIM % BK == 0) && (BLOCK_DIM % BN == 0));

    dim3 grid_dim(CEIL_DIV(N, BLOCK_DIM), CEIL_DIV(M, BLOCK_DIM));
    dim3 block_dim(BM * BN / (TM * TN));
    thread_coarsening_2d_gemm<BM * BN / (TM * TN), BM, BN, BK, TM, TN>
        <<<grid_dim, block_dim>>>(M, N, K, alpha, A, B, beta, C);
}


void run_vectorize(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    constexpr uint BLOCK_DIM{128};
    // BM: the size of block vertically; BN: the size of block horizontally. 
    constexpr uint BM{BLOCK_DIM}, BN{BLOCK_DIM};
    // TM, TN: the number of rows, columns processed by each thread, respectively.
    constexpr uint TM{16}, TN{8};

    constexpr uint BK{32};
    // Updated requirement given that we use float4 for vectorizing.
    static_assert(((BK * BM) % (4 * BLOCK_DIM) == 0) && ((BK * BN) % (4 * BLOCK_DIM) == 0));

    static_assert(BK % 4 == 0);
    static_assert((TM % 4 == 0) && (TN % 4 == 0));

    dim3 grid_dim(CEIL_DIV(N, BLOCK_DIM), CEIL_DIV(M, BLOCK_DIM));
    dim3 block_dim(BM * BN / (TM * TN));
    vectorize_gemm<BM * BN / (TM * TN), BM, BN, BK, TM, TN>
        <<<grid_dim, block_dim>>>(M, N, K, alpha, A, B, beta, C);
}


void run_vectorize(int M, int N, int K, float alpha, __nv_bfloat16 *A, __nv_bfloat16 *B, float beta, float *C) {
    constexpr uint BLOCK_DIM{128};
    // BM: the size of block vertically; BN: the size of block horizontally. 
    constexpr uint BM{BLOCK_DIM}, BN{BLOCK_DIM};
    // TM, TN: the number of rows, columns processed by each thread, respectively.
    constexpr uint TM{16}, TN{8};

    constexpr uint BK{32};
    // Updated requirement given that we use float4 for vectorizing.
    static_assert(((BK * BM) % (2 * BLOCK_DIM) == 0) && ((BK * BN) % (2 * BLOCK_DIM) == 0));

    static_assert(BK % 2 == 0);
    static_assert((TM % 2 == 0) && (TN % 2 == 0));

    dim3 grid_dim(CEIL_DIV(N, BLOCK_DIM), CEIL_DIV(M, BLOCK_DIM));
    dim3 block_dim(BM * BN / (TM * TN));
    vectorize_gemm<BM * BN / (TM * TN), BM, BN, BK, TM, TN>
        <<<grid_dim, block_dim>>>(M, N, K, alpha, A, B, beta, C);
}


void run_warptiling(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    // The overall method can be separated into two steps:
    // 1) Loading data from global memory to shared memory --> similar to the previous kernel.
    // 2) Compute the dot product between elements --> where warptiling is implemented.
    
    constexpr uint NUM_THREADS{128};
    // BM: the size of block vertically; BN: the size of block horizontally. 
    constexpr uint BM{128}, BN{128};
    // TM, TN: the number of rows, columns processed by each thread, respectively.
    constexpr uint TM{16}, TN{8};
    constexpr uint BK{32};
    // Updated requirement given that we use float4 for vectorizing.
    static_assert(((BK * BM) % (4 * NUM_THREADS) == 0) && ((BK * BN) % (4 * NUM_THREADS) == 0));
    
    // WM, WN: the number of cell  rows, columns processed by each warp, respectively.
    constexpr uint WM{32}, WN{128};
    static_assert((BN % WN == 0) && (BM % WM == 0));
    static_assert((BN / WN) * (BM / WM) == NUM_THREADS / 32);

    static_assert(WM * WN / 32 == TM * TN);

    dim3 block_dim(NUM_THREADS);
    dim3 grid_dim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    warptiling_gemm<NUM_THREADS, BM, BN, BK, WM, WN, TM, TN>
        <<<grid_dim, block_dim>>>(M, N, K, alpha, A, B, beta, C);
}


void run_warptiling(int M, int N, int K, float alpha, __nv_bfloat16 *A, __nv_bfloat16 *B, float beta, float *C) {
    // The overall method can be separated into two steps:
    // 1) Loading data from global memory to shared memory --> similar to the previous kernel.
    // 2) Compute the dot product between elements --> where warptiling is implemented.
    
    constexpr uint NUM_THREADS{128};
    // BM: the size of block vertically; BN: the size of block horizontally. 
    constexpr uint BM{128}, BN{128};
    // TM, TN: the number of rows, columns processed by each thread, respectively.
    constexpr uint TM{16}, TN{8};
    constexpr uint BK{32};
    // Updated requirement given that we use float4 for vectorizing.
    static_assert(((BK * BM) % (4 * NUM_THREADS) == 0) && ((BK * BN) % (4 * NUM_THREADS) == 0));
    
    // WM, WN: the number of cell  rows, columns processed by each warp, respectively.
    constexpr uint WM{32}, WN{128};
    static_assert((BN % WN == 0) && (BM % WM == 0));
    static_assert((BN / WN) * (BM / WM) == NUM_THREADS / 32);

    static_assert(WM * WN / 32 == TM * TN);

    dim3 block_dim(NUM_THREADS);
    dim3 grid_dim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    warptiling_gemm<NUM_THREADS, BM, BN, BK, WM, WN, TM, TN>
        <<<grid_dim, block_dim>>>(M, N, K, alpha, A, B, beta, C);
}


void run_warptiling_subdivided(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    // The overall method can be separated into two steps:
    // 1) Loading data from global memory to shared memory --> similar to the previous kernel.
    // 2) Compute the dot product between elements --> where warptiling is implemented.
    
    constexpr uint NUM_THREADS{128};
    // BM: the size of block vertically; BN: the size of block horizontally. 
    constexpr uint BM{128}, BN{128};
    // TM, TN: the number of rows, columns processed by each thread, respectively.
    constexpr uint TM{8}, TN{4};
    static_assert((TM % 4 == 0) && (TN % 4 == 0));
    
    constexpr uint BK{32};
    // Updated requirement given that we use float4 for vectorizing.
    static_assert(((BK * BM) % (4 * NUM_THREADS) == 0) && ((BK * BN) % (4 * NUM_THREADS) == 0));
    
    // WM, WN: the number of cell  rows, columns processed by each warp, respectively.
    constexpr uint WM{32}, WN{128};
    static_assert((BN % WN == 0) && (BM % WM == 0));
    static_assert((BN / WN) * (BM / WM) == NUM_THREADS / 32);

    constexpr uint WNITER{2};
    static_assert((WM * WN) % (32 * TM * TN * WNITER) == 0);
    constexpr uint WMITER{WM * WN / (32 * TM * TN * WNITER)};
    static_assert((WM % WMITER == 0) & (WN % WNITER == 0));

    dim3 block_dim(NUM_THREADS);
    dim3 grid_dim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    warptiling_subdivided_gemm<NUM_THREADS, BM, BN, BK, WM, WN, WNITER, TM, TN>
        <<<grid_dim, block_dim>>>(M, N, K, alpha, A, B, beta, C);
}


void run_transpose_shared_a(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    // The overall method can be separated into two steps:
    // 1) Loading data from global memory to shared memory --> similar to the previous kernel.
    // 2) Compute the dot product between elements --> where warptiling is implemented.
    
    constexpr uint NUM_THREADS{128};
    // BM: the size of block vertically; BN: the size of block horizontally. 
    constexpr uint BM{128}, BN{128};
    // TM, TN: the number of rows, columns processed by each thread, respectively.
    constexpr uint TM{8}, TN{4};
    constexpr uint BK{32};
    // Updated requirement given that we use float4 for vectorizing.
    static_assert(((BK * BM) % (4 * NUM_THREADS) == 0) && ((BK * BN) % (4 * NUM_THREADS) == 0));
    
    // WM, WN: the number of cell  rows, columns processed by each warp, respectively.
    constexpr uint WM{32}, WN{128};
    static_assert((BN % WN == 0) && (BM % WM == 0));
    static_assert((BN / WN) * (BM / WM) == NUM_THREADS / 32);

    constexpr uint WNITER{2};
    static_assert((WM * WN) % (32 * TM * TN * WNITER) == 0);
    constexpr uint WMITER{WM * WN / (32 * TM * TN * WNITER)};
    static_assert((WM % WMITER == 0) & (WN % WNITER == 0));

    dim3 block_dim(NUM_THREADS);
    dim3 grid_dim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    warptiling_transposed_a_gemm<NUM_THREADS, BM, BN, BK, WM, WN, WNITER, TM, TN>
        <<<grid_dim, block_dim>>>(M, N, K, alpha, A, B, beta, C);
}


void run_swizzled_smem(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    // The overall method can be separated into two steps:
    // 1) Loading data from global memory to shared memory --> similar to the previous kernel.
    // 2) Compute the dot product between elements --> where warptiling is implemented.
    
    constexpr size_t NUM_THREADS{128};
    // BM: the size of block vertically; BN: the size of block horizontally. 
    constexpr size_t BM{128}, BN{128};
    // TM, TN: the number of rows, columns processed by each thread, respectively.
    constexpr size_t TM{8}, TN{4};
    constexpr size_t BK{32};
    // Updated requirement given that we use float4 for vectorizing.
    static_assert(((BK * BM) % (4 * NUM_THREADS) == 0) && ((BK * BN) % (4 * NUM_THREADS) == 0));
    
    // WM, WN: the number of cell  rows, columns processed by each warp, respectively.
    constexpr size_t WM{32}, WN{128};
    static_assert((BN % WN == 0) && (BM % WM == 0));
    static_assert((BN / WN) * (BM / WM) == NUM_THREADS / 32);

    constexpr size_t WNITER{2};
    static_assert((WM * WN) % (32 * TM * TN * WNITER) == 0);
    constexpr size_t WMITER{WM * WN / (32 * TM * TN * WNITER)};
    static_assert((WM % WMITER == 0) & (WN % WNITER == 0));

    dim3 block_dim(NUM_THREADS);
    dim3 grid_dim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    warptiling_swizzled_smem_gemm<NUM_THREADS, BM, BN, BK, WM, WN, WNITER, TM, TN>
        <<<grid_dim, block_dim>>>(M, N, K, alpha, A, B, beta, C);
}


void run_double_buffering(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    // The overall method can be separated into two steps:
    // 1) Loading data from global memory to shared memory --> similar to the previous kernel.
    // 2) Compute the dot product between elements --> where warptiling is implemented.
    
    constexpr uint NUM_THREADS{128};
    // BM: the size of block vertically; BN: the size of block horizontally. 
    constexpr uint BM{128}, BN{128};
    // TM, TN: the number of rows, columns processed by each thread, respectively.
    constexpr uint TM{8}, TN{4};
    static_assert((TM % 4 == 0) && (TN % 4 == 0));
    
    constexpr uint BK{16};
    // Updated requirement given that we use float4 for vectorizing.
    static_assert(((BK * BM) % (4 * NUM_THREADS) == 0) && ((BK * BN) % (4 * NUM_THREADS) == 0));
    
    // WM, WN: the number of cell  rows, columns processed by each warp, respectively.
    constexpr uint WM{32}, WN{128};
    static_assert((BN % WN == 0) && (BM % WM == 0));
    static_assert((BN / WN) * (BM / WM) == NUM_THREADS / 32);

    constexpr uint WNITER{2};
    static_assert((WM * WN) % (32 * TM * TN * WNITER) == 0);
    constexpr uint WMITER{WM * WN / (32 * TM * TN * WNITER)};
    static_assert((WM % WMITER == 0) & (WN % WNITER == 0));

    dim3 block_dim(NUM_THREADS);
    dim3 grid_dim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    double_buffering_gemm<NUM_THREADS, BM, BN, BK, WM, WN, WNITER, TM, TN>
        <<<grid_dim, block_dim>>>(M, N, K, alpha, A, B, beta, C);
}


void run_double_buffering(int M, int N, int K, float alpha, __nv_bfloat16 *A, __nv_bfloat16 *B, float beta, float *C) {
    // The overall method can be separated into two steps:
    // 1) Loading data from global memory to shared memory --> similar to the previous kernel.
    // 2) Compute the dot product between elements --> where warptiling is implemented.
    
    constexpr uint NUM_THREADS{128};
    // BM: the size of block vertically; BN: the size of block horizontally. 
    constexpr uint BM{128}, BN{128};
    // TM, TN: the number of rows, columns processed by each thread, respectively.
    constexpr uint TM{8}, TN{4};
    static_assert((TM % 2 == 0) && (TN % 2 == 0));
    
    constexpr uint BK{32};
    // Updated requirement given that we use float2 for vectorizing.
    static_assert(((BK * BM) % (2 * NUM_THREADS) == 0) && ((BK * BN) % (2 * NUM_THREADS) == 0));
    
    // WM, WN: the number of cell  rows, columns processed by each warp, respectively.
    constexpr uint WM{32}, WN{128};
    static_assert((BN % WN == 0) && (BM % WM == 0));
    static_assert((BN / WN) * (BM / WM) == NUM_THREADS / 32);

    constexpr uint WNITER{2};
    static_assert((WM * WN) % (32 * TM * TN * WNITER) == 0);
    constexpr uint WMITER{WM * WN / (32 * TM * TN * WNITER)};
    static_assert((WM % WMITER == 0) & (WN % WNITER == 0));

    dim3 block_dim(NUM_THREADS);
    dim3 grid_dim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    double_buffering_gemm<NUM_THREADS, BM, BN, BK, WM, WN, WNITER, TM, TN>
        <<<grid_dim, block_dim>>>(M, N, K, alpha, A, B, beta, C);
}


void run_tensor_cores(int M, int N, int K, float alpha, __nv_bfloat16 *A, __nv_bfloat16 *B, float beta, float *C) {
    // The overall method can be separated into two steps:
    // 1) Loading data from global memory to shared memory --> similar to the previous kernel.
    // 2) Compute the dot product between elements --> where warptiling is implemented.
    constexpr uint NUM_THREADS{128};
    // BM: the size of block vertically; BN: the size of block horizontally. 
    constexpr uint BM{128}, BN{128};
    constexpr uint BK{32};


    // Updated requirement given that we use float4 for vectorizing the load.
    static_assert(((BK * BM) % (2 * NUM_THREADS) == 0) && ((BK * BN) % (2 * NUM_THREADS) == 0));
    
    // WM, WN: the number of cell  rows, columns processed by each warp, respectively.
    constexpr uint WM{32}, WN{128};
    static_assert((BN % WN == 0) && (BM % WM == 0));
    static_assert((BN / WN) * (BM / WM) == NUM_THREADS / 32);

    // Options include: 16x16x16, 32x8x16, 8x32x16, 16x16x8.
    constexpr uint WMMA_M{16};
    constexpr uint WMMA_N{16};
    constexpr uint WMMA_K{16};

    static_assert((BK % WMMA_K == 0) && (WM % WMMA_M == 0) && (WN % WMMA_N == 0));

    dim3 block_dim(NUM_THREADS);
    dim3 grid_dim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    tensor_cores_gemm<NUM_THREADS, BM, BN, BK, WM, WN, WMMA_M, WMMA_N, WMMA_K>
        <<<grid_dim, block_dim>>>(M, N, K, alpha, A, B, beta, C);
}


void run_tensor_cores_memcpy_async(int M, int N, int K, float alpha, __nv_bfloat16 *A, __nv_bfloat16 *B, float beta, float *C) {
    // The overall method can be separated into two steps:
    // 1) Loading data from global memory to shared memory --> similar to the previous kernel.
    // 2) Compute the dot product between elements --> where warptiling is implemented.
    constexpr uint NUM_THREADS{128};
    // BM: the size of block vertically; BN: the size of block horizontally. 
    constexpr uint BM{128}, BN{128};
    constexpr uint BK{32};

    // Updated requirement given that we use float4 for vectorizing the load.
    static_assert(((BK * BM) % (2 * NUM_THREADS) == 0) && ((BK * BN) % (2 * NUM_THREADS) == 0));
    
    // WM, WN: the number of cell  rows, columns processed by each warp, respectively.
    constexpr uint WM{32}, WN{128};
    static_assert((BN % WN == 0) && (BM % WM == 0));
    static_assert((BN / WN) * (BM / WM) == NUM_THREADS / 32);

    // Options include: 16x16x16, 32x8x16, 8x32x16, 16x16x8.
    constexpr uint WMMA_M{16};
    constexpr uint WMMA_N{16};
    constexpr uint WMMA_K{16};

    static_assert((BK % WMMA_K == 0) && (WM % WMMA_M == 0) && (WN % WMMA_N == 0));

    dim3 block_dim(NUM_THREADS);
    dim3 grid_dim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    tensor_cores_memcpy_async_gemm<NUM_THREADS, BM, BN, BK, WM, WN, WMMA_M, WMMA_N, WMMA_K>
        <<<grid_dim, block_dim>>>(M, N, K, alpha, A, B, beta, C);
}


void run_kernel(
    int kernel_num, int M, int N, int K, float alpha, float *A, float *B,
    float beta, float *C, cublasHandle_t handle
) {
    switch (kernel_num) {
    case 0:
        run_cublas(handle, M, N, K, alpha, A, B, beta, C);
        break;
    case 1:
        run_naive(M, N, K, alpha, A, B, beta, C);
        break;
    case 2:
        run_sharedmem_block_tiling(M, N, K, alpha, A, B, beta, C);
        break;
    case 3:
        run_2d_thread_coarsening(M, N, K, alpha, A, B, beta, C);
        break;
    case 4:
        assert((N % 4 == 0) && (K % 4 == 0));
        run_vectorize(M, N, K, alpha, A, B, beta, C);
        break;
    case 5:
        run_warptiling(M, N, K, alpha, A, B, beta, C);
        break;
    case 6:
        run_warptiling_subdivided(M, N, K, alpha, A, B, beta, C);
        break;
    case 7:
        run_transpose_shared_a(M, N, K, alpha, A, B, beta, C);
        break;
    case 8:
        run_swizzled_smem(M, N, K, alpha, A, B, beta, C);
        break;
    case 9:
        run_double_buffering(M, N, K, alpha, A, B, beta, C);
        break;
    default:
        throw std::invalid_argument("Invalid kernel number.");
    }
}


void run_kernel(
    int kernel_num, int M, int N, int K, float alpha, __nv_bfloat16 *A, __nv_bfloat16 *B,
    float beta, float *C, cublasHandle_t handle
) {
    switch (kernel_num) {
    case 0:
        run_cublas(handle, M, N, K, alpha, A, B, beta, C);
        break;
    case 1:
        run_naive(M, N, K, alpha, A, B, beta, C);
        break;
    case 4:
        assert((N % 2 == 0) && (K % 2 == 0));
        run_vectorize(M, N, K, alpha, A, B, beta, C);
        break;
    case 5:
        run_warptiling(M, N, K, alpha, A, B, beta, C);
        break;
    case 9:
        run_double_buffering(M, N, K, alpha, A, B, beta, C);
        break;
    case 10:
        run_tensor_cores(M, N, K, alpha, A, B, beta, C);
        break;
    case 11:
        run_tensor_cores_memcpy_async(M, N, K, alpha, A, B, beta, C);
        break;
    default:
        throw std::invalid_argument("Invalid kernel number.");
    }
}