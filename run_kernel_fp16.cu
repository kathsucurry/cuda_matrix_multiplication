#include <cuda_bf16.h>
#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>

#include "src/runner.cuh"
#include "src/host_utils.cuh"


int main(int argc, char **argv) {
    int const kernel_num{get_kernel_input(argc, argv)};

    // Set CUDA context.
    CHECK_CUDA_ERROR(cudaSetDevice(0));
    
    // Create a handle to be used for using cublas.
    cublasHandle_t cublas_handle;
    if (cublasCreate(&cublas_handle)) {
        std::cerr << "Error encountered when creating a cublas handler." << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // Generate matrices.
    // std::vector<size_t> const MATRIX_SIZE = {128, 256, 512, 1024, 2048, 4096, 8192};
    std::vector<uint> const MATRIX_SIZE = {4096};

    uint const max_matrix_size = MATRIX_SIZE[MATRIX_SIZE.size() - 1];
    std::cout << "Max matrix size: " << max_matrix_size << std::endl;

    // Prepare host and device matrices variables.
    // Define the type (float/fp32, ...) accordingly.
    __nv_bfloat16 *A{nullptr}, *B{nullptr};
    __nv_bfloat16 *A_d{nullptr}, *B_d{nullptr};
    float *C{nullptr}, *C_ref{nullptr};
    float *C_d{nullptr}, *C_ref_d{nullptr};
    prepare_matrices(A, B, C, C_ref, A_d, B_d, C_d, C_ref_d, max_matrix_size);

    measure_performance(
        MATRIX_SIZE, cublas_handle, kernel_num, A, B, C, C_ref, A_d, B_d, C_d, C_ref_d);

    free_matrices(A, B, C, C_ref, A_d, B_d, C_d, C_ref_d);
    
    cublasDestroy(cublas_handle);

    return 0;
}
