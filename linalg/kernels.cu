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

    __global__ void matmul(float *a, float val, float *result, int N)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < N)
            result[idx] = val * a[idx];
    }

    // a is of dimensions [d1, d2], b is of dimensions [d2, d3]
    __global__ void matmul(float *a, float *b, float *result, int d1, int d2, int d3)
    {
        int idx = blockDim.y * blockIdx.y + threadIdx.y;
        int jdx = blockDim.x * blockIdx.x + threadIdx.x;

        if (idx < d1 && jdx < d3) {
            float val = 0;
            for (int kdx = 0; kdx != d2; ++kdx)
                val += a[idx * d2 + kdx] * b[kdx * d3 + jdx];
        }
    }

    __global__ void copy(float *arr, float *target, int N)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < N)
            target[idx] = arr[idx];
    }
}
