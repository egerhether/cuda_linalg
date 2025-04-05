#include "matrix.cuh"

namespace linalg {

    Matrix::Matrix(int rows, int columns, float value)
    {
        d_data = new float[rows * columns];
        d_shape = std::pair<int, int>(rows, columns);
        fill(value);
    }

    Matrix::Matrix(float *data, int rows, int columns)
    {
        d_data = data;
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
        int blocks = ceil(float(N) / threads);

        gpu::add<<<blocks, threads>>>(cuda_a, value, cuda_c, N);

        cudaMemcpy(c, cuda_c, N * sizeof(float), cudaMemcpyDeviceToHost);

        Matrix result = Matrix(c, d_shape.first, d_shape.second);

        delete[] c;

        return result;
    }

    Matrix::~Matrix()
    {
        delete[] d_data;
    }

    Matrix Matrix::add(Matrix &matrix)
    {
        int N = d_shape.first * d_shape.second;

        if (d_shape != matrix.shape())
            throw "Matrix dimensions must match for addition!";

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
        int blocks = ceil(float(N) / threads);

        gpu::add<<<blocks, threads>>>(cuda_a, cuda_b, cuda_c, N);

        cudaMemcpy(c, cuda_c, N * sizeof(float), cudaMemcpyDeviceToHost);

        Matrix result = Matrix(c, d_shape.first, d_shape.second);

        delete[] c;

        return result;
    }

    void Matrix::fill(float value)
    {
        int N = d_shape.first * d_shape.second;
        // for large matrices - parallelize
        if (N > 10000) {
            float *cuda_arr;

            cudaMalloc(&cuda_arr, N * sizeof(float));

            int threads = 256;
            int blocks = ceil(float(N) / threads);
            gpu::fill<<<blocks, threads>>>(cuda_arr, value, N);

            cudaMemcpy(d_data, cuda_arr, N * sizeof(float), cudaMemcpyDeviceToHost);
        } else
            for (int idx = 0; idx != N; ++idx)
                d_data[idx] = value;
    }

    float Matrix::mean()
    {
        int N = d_shape.first * d_shape.second;
        float sum = 0.0;
        for (int idx = 0; idx != N; ++idx)
            sum += d_data[idx];

        return sum / N;
    }

    float *Matrix::get_data()
    {
        return d_data;
    }

    std::pair<int, int> &Matrix::shape()
    {
        return d_shape;
    }

}
