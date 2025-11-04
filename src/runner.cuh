#pragma once

#include <cublas_v2.h>
#include <stdexcept>


void run_kernel(
    int kernel_num, int M, int N, int K, float alpha, float *A, float *B,
    float beta, float *C, cublasHandle_t handle
);
