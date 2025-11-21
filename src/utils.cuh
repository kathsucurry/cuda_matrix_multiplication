#pragma once

#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <sys/time.h>

#define EPS 1e-2
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define KERNEL_NUM 10
#define CHECK_CUDA_ERROR(value) check_cuda_error((value), #value, __FILE__, __LINE__)
#define CHECK_LAST_CUDA_ERROR() check_last_cuda_error(__FILE__, __LINE__)

void check_cuda_error(cudaError_t error, char const *func, char const *file, int line);
void check_last_cuda_error(char const *file, int line);

int get_kernel_input(int argc, char **argv);

template <typename T>
void randomize_matrix(T *matrix, size_t N) {
    struct timeval time{};
    gettimeofday(&time, nullptr);
    srand(time.tv_usec);

    for (size_t i{0}; i < N; ++i) {
        T value = (T)(rand() % 5) + 0.01 * (rand() % 5);
        value = (rand() % 2 == 0) ? value : value * (-1.);
        matrix[i] = value;
        // matrix[i] = (i % 128) + 1;
    }
}

template <typename T>
bool verify_matrix(T *matrix_1, T *matrix_2, size_t N) {
    double diff{0.0};

    for (size_t i{0}; i < N; ++i) {
        diff = std::fabs(matrix_1[i] - matrix_2[i]);
        if (isnan(diff) || diff > EPS) {
            printf(
                "Divergence encountered with at %zu with diff %5.4f; expected: %5.4f but actual: %5.4f\n",
                i, diff, matrix_1[i], matrix_2[i]);
            return false;
        }
    }    
    return true;
}


template <typename T>
void prepare_matrices(
    T     *&A,
    T     *&B,
    T     *&C,
    T     *&C_ref,
    T     *&A_d,
    T     *&B_d,
    T     *&C_d,
    T     *&C_ref_d,
    size_t matrix_size
) {
    A     = (T *)malloc(sizeof(T) * matrix_size * matrix_size);
    B     = (T *)malloc(sizeof(T) * matrix_size * matrix_size);
    C     = (T *)malloc(sizeof(T) * matrix_size * matrix_size);
    C_ref = (T *)malloc(sizeof(T) * matrix_size * matrix_size); 

    randomize_matrix(A, matrix_size * matrix_size);
    randomize_matrix(B, matrix_size * matrix_size);
    randomize_matrix(C, matrix_size * matrix_size);

    CHECK_CUDA_ERROR(cudaMalloc((void **)&A_d, sizeof(T) * matrix_size * matrix_size));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&B_d, sizeof(T) * matrix_size * matrix_size));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&C_d, sizeof(T) * matrix_size * matrix_size));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&C_ref_d, sizeof(T) * matrix_size * matrix_size));

    CHECK_CUDA_ERROR(cudaMemcpy(A_d, A, sizeof(T) * matrix_size * matrix_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(B_d, B, sizeof(T) * matrix_size * matrix_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(C_d, C, sizeof(T) * matrix_size * matrix_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(C_ref_d, C, sizeof(T) * matrix_size * matrix_size, cudaMemcpyHostToDevice));
}


template <typename T>
void free_matrices(
    T *&A,
    T *&B,
    T *&C,
    T *&C_ref,
    T *&A_d,
    T *&B_d,
    T *&C_d,
    T *&C_ref_d
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