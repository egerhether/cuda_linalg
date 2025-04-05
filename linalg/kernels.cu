#include "matrix.cuh"

namespace gpu {

    __global__ void add(float *a, float *b, float *result, int N)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < N)
            result[idx] = a[idx] + b[idx];
    }

    __global__ void add(float *a, float b, float *result, int N)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < N)
            result[idx] = a[idx] + b;
    }

    __global__ void fill(float *arr, float val, int N)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < N)
            arr[idx] = val;
    }
}
