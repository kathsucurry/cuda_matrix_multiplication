# General matrix multiplication optimization with CUDA

Given 3 matrices A (m x k | **row-major**), B (k x n | **row-major**), C (m x n | **row-major**), two constants α and β, perform the following operation:

```
C = α * AB + β * C
```

The implemented kernels are able to achieve the following performance on RTX 5070 Ti:

**SGEMM: >95% cuBLAS performance**

| Kernel # | Performance (TFLOPs/s) | % cuBLAS performance |
|:---|:---:|:---:|
| cuBLAS | 31.66 | 100% |
| kernel 01: naive | 2.05 | 6.47% |
| kernel 02: block tiling | 3.49 | 11.02% |
| kernel 03: 2D thread coarsening | 20.05 | 63.33% |
| kernel 04: vectorized memory access | 26.16 | 82.63% |
| kernel 05: warp tiling | 26.19 | 82.72% |
| kernel 06: warp tiling, subdivided | 28.70 | 90.65% |
| kernel 07: transposing `As` | 29.71 | 93.84% |
| kernel 08: asynchronous copy + double buffering | 30.28 | 95.64% |


**MP-GEMM: >101% cuBLAS performance**

| Kernel # | Performance (TFLOPs/s) | % cuBLAS performance |
|:---|:---:|:---:|
| cuBLAS | 88.81 | 100% |
| kernel 09: tensor cores (wmma API) | 65.38 | 73.62% |
| kernel 10: tensor cores + async gmem loads  | 73.77 | 83.06% |
| kernel 11: tensor cores + double buffering | 80.33 | 90.45% |
| kernel 12: tensor cores + three-level pipeline | 87.13 | 98.11% |
| kernel 13: tensor cores (mma) | 76.93 | 86.62% |
| kernel 14: tensor cores (mma) swizzled | 84.24 | 94.85% |
| kernel 15: tensor cores (mma) swizzled + three-level pipeline | 90.31 | 101.69% |


The post describing the approaches can be found [here](https://kathsucurry.github.io/cuda/2025/12/22/gemm.html). Additionally, I used [Simon's code](https://github.com/siboehm/SGEMM_CUDA) to setup the matrix generation, performance measurement, and kernel verification.


## Build the project

For debugging purposes:
```
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake --build . -j
```