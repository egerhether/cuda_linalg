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
        d_data = new float[rows * columns];

        copy(data);

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

        cudaFree(cuda_a);
        cudaFree(cuda_c);

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

        cudaFree(cuda_a);
        cudaFree(cuda_b);
        cudaFree(cuda_c);

        return result;
    }

    Matrix Matrix::mult(float value)
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

        gpu::matmul<<<blocks, threads>>>(cuda_a, value, cuda_c, N);

        cudaMemcpy(c, cuda_c, N * sizeof(float), cudaMemcpyDeviceToHost);

        Matrix result = Matrix(c, d_shape.first, d_shape.second);

        delete[] c;

        cudaFree(cuda_a);
        cudaFree(cuda_c);

        return result;
    }

    Matrix Matrix::mult(Matrix &matrix)
    {

        if (d_shape.second != matrix.shape().first)
            throw "Inner dimensions must match for matmul!";

        // sizes of each matrix involved
        int Na = d_shape.first * d_shape.second;
        int Nb = matrix.shape().first * matrix.shape().second;
        int Nc = d_shape.first * matrix.shape().second;

        float *c = new float[Nc];

        float *cuda_a, *cuda_b, *cuda_c;

        cudaMalloc(&cuda_a, Na * sizeof(float));
        cudaMalloc(&cuda_b, Nb * sizeof(float));
        cudaMalloc(&cuda_c, Nc * sizeof(float));

        cudaMemcpy(cuda_a, d_data, Na * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_b, matrix.get_data(), Nb * sizeof(float), cudaMemcpyHostToDevice);

        dim3 dim_block(32, 32, 1);
        dim3 dim_grid(ceil(Nc / 32.0), ceil(Na / 32.0), 1);
        gpu::matmul<<<dim_grid, dim_block>>>(cuda_a, cuda_b, cuda_c, d_shape.first, d_shape.second, matrix.shape().second);

        cudaMemcpy(c, cuda_c, Nc * sizeof(float), cudaMemcpyDeviceToHost);

        Matrix result = Matrix(c, d_shape.first, matrix.shape().second);

        delete[] c;

        cudaFree(cuda_a);
        cudaFree(cuda_b);
        cudaFree(cuda_c);

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

            cudaFree(cuda_arr);
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

    void Matrix::copy(float *data)
    {
        int N = d_shape.first * d_shape.second;

        if (N > 10000) {
            float *cuda_arr, *cuda_target;

            cudaMalloc(&cuda_arr, N * sizeof(float));
            cudaMalloc(&cuda_target, N * sizeof(float));

            cudaMemcpy(cuda_arr, data, N * sizeof(float), cudaMemcpyHostToDevice);

            int threads = 256;
            int blocks = ceil(float(N) / threads);
            gpu::copy<<<blocks, threads>>>(cuda_arr, cuda_target, N);

            cudaMemcpy(d_data, cuda_target, N * sizeof(float), cudaMemcpyDeviceToHost);

            cudaFree(cuda_arr);
            cudaFree(cuda_target);
        } else
            for (int idx = 0; idx != N; ++idx)
                d_data[idx] = data[idx];
    }

}
