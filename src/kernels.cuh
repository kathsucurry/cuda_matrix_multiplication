#pragma once

#include <cublas_v2.h>

void run_kernel(
    int kernel_num, int m, int n, int k, float alpha, float *A, float *B,
    float beta, float *C, cublasHandle_t handle
);

bool verify_matrix(float *matrix_1, float *matrix_2, size_t N);

void run_kernel(
    int kernel_num, int m, int n, int k, float alpha, float *A, float *B,
    float beta, float *C, cublasHandle_t handle
) {

}


bool verify_matrix(float *matrix_1, float *matrix_2, size_t N) {

}