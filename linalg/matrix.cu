#include "matrix.cuh"

namespace gpu {

    __global__ void gpu_add(float *a, float *b, float *result, int N)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < N)
            result[idx] = a[idx] + b[idx];
    }

    __global__ void gpu_add(float *a, float b, float *result, int N)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < N)
            result[idx] = a[idx] + b;
    }
}

namespace linalg {

    // potentially parallelize this later on
    Matrix::Matrix(int rows, int columns, float value)
    {
        d_data = new float[rows * columns];
        for (int idx = 0; idx != rows * columns; ++idx)
            d_data[idx] = value;

        d_shape = std::pair<int, int>(rows, columns);
    }

    Matrix::Matrix(float *data, int rows, int columns)
    {
        d_data  = data;
        d_shape = std::pair<int, int>(rows, columns);
    }

    Matrix Matrix::add(float value)
    {
        int N = d_shape.first * d_shape.second;

        float *c = new float[N];

        float *cuda_a, *cuda_c;

        // memory alloc on gpu
        cudaMalloc(&cuda_a, N * sizeof(float));
        cudaMalloc(&cuda_c, N * sizeof(float));

        // copy vectors to gpu
        cudaMemcpy(cuda_a, d_data, N * sizeof(float), cudaMemcpyHostToDevice);

        // initialize size of gpu to run on
        int threads = 256;
        int blocks  = ceil(float(N) / threads);

        gpu::gpu_add<<<blocks, threads>>>(cuda_a, value, cuda_c, N);

        cudaMemcpy(c, cuda_c, N * sizeof(float), cudaMemcpyDeviceToHost);

        Matrix result = Matrix(c, d_shape.first, d_shape.second);

        delete[] c;

        return result;
    }

    Matrix Matrix::add(Matrix &matrix)
    {
        int N = d_shape.first * d_shape.second;

        float *c = new float[N];

        float *cuda_a, *cuda_b, *cuda_c;

        // memory alloc on gpu
        cudaMalloc(&cuda_a, N * sizeof(float));
        cudaMalloc(&cuda_b, N * sizeof(float));
        cudaMalloc(&cuda_c, N * sizeof(float));

        // copy vectors to gpu
        cudaMemcpy(cuda_a, d_data, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_b, matrix.get_data(), N * sizeof(float), cudaMemcpyHostToDevice);

        // initialize size of gpu to run on
        int threads = 256;
        int blocks  = ceil(float(N) / threads);

        gpu::gpu_add<<<blocks, threads>>>(cuda_a, cuda_b, cuda_c, N);

        cudaMemcpy(c, cuda_c, N * sizeof(float), cudaMemcpyDeviceToHost);

        Matrix result = Matrix(c, d_shape.first, d_shape.second);

        delete[] c;

        return result;
    }

    float *Matrix::get_data()
    {
        return d_data;
    }

}
