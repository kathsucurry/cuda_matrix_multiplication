#pragma once

#include <cuda_bf16.h> 


template <typename T>
struct float_traits; 


template <>
struct float_traits<float> {
    using compute_t = float;
    
    static __host__ __device__ compute_t to_compute(float x) { return x; }
    static __host__ __device__ float from_compute(compute_t x) { return x; }
};


template <>
struct float_traits<__nv_bfloat16> {
    using compute_t = float;
    
    static __host__ __device__ compute_t to_compute(__nv_bfloat16 x) {
        return __bfloat162float(x);
    }
    static __host__ __device__ __nv_bfloat16 from_compute(compute_t x) {
        return __float2bfloat16(x);
    }
};
