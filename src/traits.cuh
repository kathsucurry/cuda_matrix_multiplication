#include <cuda_fp16.h> 


template <typename T>
struct scalar_traits; 


template <>
struct scalar_traits<double> {
    using compute_t = double;
    
    static __host__ __device__ compute_t to_compute(double x) { return x; }
    static __host__ __device__ compute_t from_compute(compute_t x) { return x; }
};


template <>
struct scalar_traits<float> {
    using compute_t = float;
    
    static __host__ __device__ compute_t to_compute(float x) { return x; }
    static __host__ __device__ compute_t from_compute(compute_t x) { return x; }
};


template <>
struct scalar_traits<__half> {
    using compute_t = float;
    
    static __host__ __device__ compute_t to_compute(__half x) { return __half2float(x); }
    static __host__ __device__ compute_t from_compute(compute_t x) { return __float2half(x); }
};
