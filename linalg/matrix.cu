#include "cuda_utils.cuh"
#include "matrix.cuh"
#include <iostream>
#include <string>

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
        d_shape = std::pair<int, int>(rows, columns);
        copy(data);
    }

    Matrix::~Matrix()
    {
        delete[] d_data;
    }

    void Matrix::inv()
    {
        // TODO: write nice inverse
    }

    void Matrix::transpose()
    {
    }

    Matrix Matrix::add(float value)
    {
        int N = d_shape.first * d_shape.second;

        float *c = new float[N];

        float *cuda_a, *cuda_c;

        // memory alloc on gpu
        cudaError_t err = cudaMalloc(&cuda_a, N * sizeof(float));
        cuda_check(err);
        err = cudaMalloc(&cuda_c, N * sizeof(float));
        cuda_check(err);

        // copy vectors to gpu
        err = cudaMemcpy(cuda_a, d_data, N * sizeof(float), cudaMemcpyHostToDevice);
        cuda_check(err);

        // initialize size of gpu to run on
        int threads = 256;
        int blocks = ceil(float(N) / threads);

        gpu::add<<<blocks, threads>>>(cuda_a, value, cuda_c, N);

        err = cudaMemcpy(c, cuda_c, N * sizeof(float), cudaMemcpyDeviceToHost);
        cuda_check(err);

        Matrix result = Matrix(c, d_shape.first, d_shape.second);

        delete[] c;

        cudaFree(cuda_a);
        cudaFree(cuda_c);

        return result;
    }

    Matrix Matrix::add(Matrix &matrix)
    {
        int N = d_shape.first * d_shape.second;

        if (d_shape != matrix.shape())
            throw std::string("Matrix dimensions must match for addition!");

        float *c = new float[N];

        float *cuda_a, *cuda_b, *cuda_c;

        // memory alloc on gpu
        cudaError_t err = cudaMalloc(&cuda_a, N * sizeof(float));
        cuda_check(err);
        err = cudaMalloc(&cuda_b, N * sizeof(float));
        cuda_check(err);
        err = cudaMalloc(&cuda_c, N * sizeof(float));
        cuda_check(err);

        // copy vectors to gpu
        err = cudaMemcpy(cuda_a, d_data, N * sizeof(float), cudaMemcpyHostToDevice);
        cuda_check(err);
        err = cudaMemcpy(cuda_b, matrix.get_data(), N * sizeof(float), cudaMemcpyHostToDevice);
        cuda_check(err);

        // initialize size of gpu to run on
        int threads = 256;
        int blocks = ceil(float(N) / threads);

        gpu::add<<<blocks, threads>>>(cuda_a, cuda_b, cuda_c, N);
        err = cudaGetLastError();
        cuda_check(err);

        err = cudaMemcpy(c, cuda_c, N * sizeof(float), cudaMemcpyDeviceToHost);
        cuda_check(err);

        Matrix result = Matrix(c, d_shape.first, d_shape.second);

        delete[] c;

        cudaFree(cuda_a);
        cudaFree(cuda_b);
        cudaFree(cuda_c);

        return result;
    }

    Matrix Matrix::operator+(float value)
    {
        return add(value);
    }

    Matrix Matrix::operator+(Matrix &matrix)
    {
        return add(matrix);
    }

    Matrix Matrix::mult(float value)
    {
        int N = d_shape.first * d_shape.second;

        float *c = new float[N];

        float *cuda_a, *cuda_c;

        // memory alloc on gpu
        cudaError_t err = cudaMalloc(&cuda_a, N * sizeof(float));
        cuda_check(err);
        err = cudaMalloc(&cuda_c, N * sizeof(float));
        cuda_check(err);

        // copy vectors to gpu
        err = cudaMemcpy(cuda_a, d_data, N * sizeof(float), cudaMemcpyHostToDevice);
        cuda_check(err);

        // initialize size of gpu to run on
        int threads = 256;
        int blocks = ceil(float(N) / threads);

        gpu::matmul<<<blocks, threads>>>(cuda_a, value, cuda_c, N);

        err = cudaMemcpy(c, cuda_c, N * sizeof(float), cudaMemcpyDeviceToHost);
        cuda_check(err);

        Matrix result = Matrix(c, d_shape.first, d_shape.second);

        delete[] c;

        cudaFree(cuda_a);
        cudaFree(cuda_c);

        return result;
    }

    Matrix Matrix::mult(Matrix &matrix)
    {

        if (d_shape.second != matrix.shape().first)
            throw std::string("Inner dimensions must match for matmul!");

        // sizes of each matrix involved
        int Na = d_shape.first * d_shape.second;
        int Nb = matrix.shape().first * matrix.shape().second;
        int Nc = d_shape.first * matrix.shape().second;

        float *c = new float[Nc];

        float *cuda_a, *cuda_b, *cuda_c;

        cudaError_t err = cudaMalloc(&cuda_a, Na * sizeof(float));
        cuda_check(err);
        err = cudaMalloc(&cuda_b, Nb * sizeof(float));
        cuda_check(err);
        err = cudaMalloc(&cuda_c, Nc * sizeof(float));
        cuda_check(err);

        err = cudaMemcpy(cuda_a, d_data, Na * sizeof(float), cudaMemcpyHostToDevice);
        cuda_check(err);
        err = cudaMemcpy(cuda_b, matrix.get_data(), Nb * sizeof(float), cudaMemcpyHostToDevice);
        cuda_check(err);

        dim3 dim_block(32, 32, 1);
        dim3 dim_grid(ceil(Nc / 32.0), ceil(Na / 32.0), 1);
        gpu::matmul<<<dim_grid, dim_block>>>(cuda_a, cuda_b, cuda_c, d_shape.first, d_shape.second, matrix.shape().second);

        err = cudaMemcpy(c, cuda_c, Nc * sizeof(float), cudaMemcpyDeviceToHost);
        cuda_check(err);

        Matrix result = Matrix(c, d_shape.first, matrix.shape().second);

        delete[] c;

        cudaFree(cuda_a);
        cudaFree(cuda_b);
        cudaFree(cuda_c);

        return result;
    }

    Matrix Matrix::operator*(float value)
    {
        return mult(value);
    }

    Matrix Matrix::operator*(Matrix &matrix)
    {
        return mult(matrix);
    }

    void Matrix::fill(float value)
    {
        int N = d_shape.first * d_shape.second;
        // for large matrices - parallelize
        if (N > 10000) {
            float *cuda_arr;

            cudaError_t err = cudaMalloc(&cuda_arr, N * sizeof(float));
            cuda_check(err);

            int threads = 256;
            int blocks = ceil(float(N) / threads);
            gpu::fill<<<blocks, threads>>>(cuda_arr, value, N);

            err = cudaMemcpy(d_data, cuda_arr, N * sizeof(float), cudaMemcpyDeviceToHost);
            cuda_check(err);

            cudaFree(cuda_arr);
        } else
            for (int idx = 0; idx != N; ++idx)
                d_data[idx] = value;
    }

    float Matrix::at(int row, int column)
    {
        return d_data[row * d_shape.second + column];
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

            cudaError_t err = cudaMalloc(&cuda_arr, N * sizeof(float));
            cuda_check(err);
            err = cudaMalloc(&cuda_target, N * sizeof(float));
            cuda_check(err);

            err = cudaMemcpy(cuda_arr, data, N * sizeof(float), cudaMemcpyHostToDevice);
            cuda_check(err);

            int threads = 256;
            int blocks = ceil(float(N) / threads);
            gpu::copy<<<blocks, threads>>>(cuda_arr, cuda_target, N);

            err = cudaMemcpy(d_data, cuda_target, N * sizeof(float), cudaMemcpyDeviceToHost);
            cuda_check(err);

            cudaFree(cuda_arr);
            cudaFree(cuda_target);
        } else
            for (int idx = 0; idx != N; ++idx)
                d_data[idx] = data[idx];
    }

    void Matrix::print()
    {
        if (d_shape.first * d_shape.second > 20) {
            // figure out what to do for large matrices
        } else {
            std::cout << '[';
            for (int idx = 0; idx != d_shape.first; ++idx) {
                std::cout << '[';
                for (int jdx = 0; jdx != d_shape.second; ++jdx) {
                    std::cout << at(idx, jdx);
                    if (jdx != d_shape.second - 1)
                        std::cout << ", ";
                }
                std::cout << ']';
                if (idx != d_shape.first - 1)
                    std::cout << '\n';
            }
            std::cout << "]\n";
        }
    }

}
