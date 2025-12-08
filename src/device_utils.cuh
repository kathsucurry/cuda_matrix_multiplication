#pragma once


template <uint const N>
__device__ constexpr uint ilog2() {
    static_assert(N > 0, "N must be positive.");
    uint x{0};
    uint tmp{N};
    while (tmp > 1) {
        tmp >>= 1;
        ++x;
    }
    return x;
}
