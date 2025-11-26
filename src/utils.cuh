#pragma once

#include <cuda/std/type_traits>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <sys/time.h>

#include "traits.cuh"


#define EPS_FP32 1e-2
#define EPS_FP16 1e-1
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define KERNEL_NUM 20
#define CHECK_CUDA_ERROR(value) check_cuda_error((value), #value, __FILE__, __LINE__)
#define CHECK_LAST_CUDA_ERROR() check_last_cuda_error(__FILE__, __LINE__)


void check_cuda_error(cudaError_t error, char const *func, char const *file, int line);
void check_last_cuda_error(char const *file, int line);

int get_kernel_input(int argc, char **argv);

template <typename T>
void randomize_matrix(T *matrix, size_t N) {
    using traits = float_traits<T>;
    using compute_t = typename traits::compute_t;

    struct timeval time{};
    gettimeofday(&time, nullptr);
    srand(time.tv_usec);

    for (size_t i{0}; i < N; ++i) {
        compute_t value = compute_t(rand() % 5) + compute_t(0.01) * compute_t(rand() % 5);
        value = (rand() % 2 == 0) ? value : (value * compute_t(-1.));
        matrix[i] = traits::from_compute(value);
        // matrix[i] = (i % 128) + 1;
    }
}

template <typename T>
bool verify_matrix(T *matrix_1, T *matrix_2, size_t N) {
    using traits = float_traits<T>;
    
    double diff{0.0};
    double eps{::cuda::std::is_same_v<T, float> ? EPS_FP32 : EPS_FP16};


    for (size_t i{0}; i < N; ++i) {
        diff = std::fabs(
            static_cast<double>(traits::to_compute(matrix_1[i])) -
            static_cast<double>(traits::to_compute(matrix_2[i])));
        if (isnan(diff) || diff > eps) {
            printf(
                "Divergence encountered with at %zu with diff %5.4f; expected: %5.4f but actual: %5.4f\n",
                i, diff, traits::to_compute(matrix_1[i]), traits::to_compute(matrix_2[i]));
            return false;
        }
    }    
    return true;
}


template <typename T, typename U>
void prepare_matrices(
    T     *&A,
    T     *&B,
    U     *&C,
    U     *&C_ref,
    T     *&A_d,
    T     *&B_d,
    U     *&C_d,
    U     *&C_ref_d,
    size_t matrix_size
) {
    A     = (T *)malloc(sizeof(T) * matrix_size * matrix_size);
    B     = (T *)malloc(sizeof(T) * matrix_size * matrix_size);
    C     = (U *)malloc(sizeof(U) * matrix_size * matrix_size);
    C_ref = (U *)malloc(sizeof(U) * matrix_size * matrix_size); 

    randomize_matrix(A, matrix_size * matrix_size);
    randomize_matrix(B, matrix_size * matrix_size);
    randomize_matrix(C, matrix_size * matrix_size);

    CHECK_CUDA_ERROR(cudaMalloc((void **)&A_d, sizeof(T) * matrix_size * matrix_size));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&B_d, sizeof(T) * matrix_size * matrix_size));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&C_d, sizeof(U) * matrix_size * matrix_size));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&C_ref_d, sizeof(U) * matrix_size * matrix_size));

    CHECK_CUDA_ERROR(cudaMemcpy(A_d, A, sizeof(T) * matrix_size * matrix_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(B_d, B, sizeof(T) * matrix_size * matrix_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(C_d, C, sizeof(U) * matrix_size * matrix_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(C_ref_d, C, sizeof(U) * matrix_size * matrix_size, cudaMemcpyHostToDevice));
}


template <typename T, typename U>
void free_matrices(
    T *&A,
    T *&B,
    U *&C,
    U *&C_ref,
    T *&A_d,
    T *&B_d,
    U *&C_d,
    U *&C_ref_d
) {
    free(A);
    free(B);
    free(C);
    free(C_ref);

    CHECK_CUDA_ERROR(cudaFree(A_d));
    CHECK_CUDA_ERROR(cudaFree(B_d));
    CHECK_CUDA_ERROR(cudaFree(C_d));
    CHECK_CUDA_ERROR(cudaFree(C_ref_d));
}