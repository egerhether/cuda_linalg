#include "matrix.cuh"
#include <cooperative_groups.h>

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

    __global__ void sub(float *a, float *b, float *result, int N)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < N)
            result[idx] = a[idx] = b[idx];
    }

    __global__ void sub(float *a, float b, float *result, int N)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < N)
            result[idx] = a[idx] - b;
    }

    __global__ void matmul(float *a, float val, float *result, int N)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < N)
            result[idx] = val * a[idx];
    }

    // a is of dimensions [d1, d2], b is of dimensions [d2, d3]
    // TODO: improve this using shared memoery!
    __global__ void matmul(float *a, float *b, float *result, int d1, int d2, int d3)
    {
        int idx = blockDim.y * blockIdx.y + threadIdx.y;
        int jdx = blockDim.x * blockIdx.x + threadIdx.x;

        if (idx < d1 && jdx < d3) {
            float val = 0;
            for (int kdx = 0; kdx != d2; ++kdx)
                val += a[idx * d2 + kdx] * b[kdx * d3 + jdx];

            result[idx * d3 + jdx] = val;
        }
    }

    __global__ void div(float *a, float val, float *result, int N)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < N)
            result[idx] = a[idx] / val;
    }

    __global__ void div(float *a, float *b, float *result, int N)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < N)
            result[idx] = a[idx] / b[idx];
    }

    // matrix of dimensions [d1, d2]
    __global__ void transpose(float *arr, float *target, int d1, int d2, int block_rows)
    {
        // using 32 as tile size as gpus optimized for this size of threads
        __shared__ float tile[32 * 32];
        int idx = blockIdx.x * 32 + threadIdx.y;
        int jdx = blockIdx.y * 32 + threadIdx.x;

        for (int kdx = 0; kdx != 32; kdx += block_rows)
            if (idx < d2 && (jdx + kdx) < d1)
                tile[(threadIdx.y + kdx) * 32 + threadIdx.x] = arr[(jdx + kdx) * d2 + idx];

        cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
        cooperative_groups::sync(block);

        for (int kdx = 0; kdx != 32; kdx += block_rows)
            if (idx < d2 && (jdx + kdx) < d1)
                target[idx * d1 + jdx + kdx] = tile[(threadIdx.y + kdx) * 32 + threadIdx.x];
    }

    __global__ void fill(float *arr, float val, int N)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < N)
            arr[idx] = val;
    }
    __global__ void copy(float *arr, float *target, int N)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < N)
            target[idx] = arr[idx];
    }
}
