# Matrix multiplication optimization with CUDA

Given 3 matrices A (m x k | **row-major**), B (k x n | **column-major**), C (m x n | **row-major**), two constants α and β, perform the following operation:

```
C = α * AB + β * C
```

I used the code setup built and described [here](https://siboehm.com/articles/22/CUDA-MMM).


## Build the project

For debugging purposes:
```
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake --build . -j
```