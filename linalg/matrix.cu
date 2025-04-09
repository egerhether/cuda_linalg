#include "cuda_utils.cuh"
#include "matrix.cuh"
#include <iostream>
#include <string>

namespace linalg {

    Matrix::Matrix(int rows, int columns, float value)
    {
        d_data = new float[rows * columns];
        d_shape = std::pair<int, int>(rows, columns);
        gpu(); // by default initialize to operate on gpu
        fill(value);
    }

    Matrix::Matrix(float *data, int rows, int columns)
    {
        d_data = new float[rows * columns];
        d_shape = std::pair<int, int>(rows, columns);
        gpu(); // by default initialize to operate on gpu
        copy(data);
    }

    Matrix::Matrix(Matrix const &other)
    {
        float *d_data = new float[other.shape().first * other.shape().second];
        copy(other.get_data());
        d_shape = other.shape();
    }

    Matrix &Matrix::operator=(Matrix const &other)
    {
        if (this != &other) {
            delete[] d_data;

            d_data = new float[other.shape().first * other.shape().second];
            copy(other.get_data());
            d_shape = other.shape();
        }

        return *this;
    }

    Matrix::~Matrix()
    {
        delete[] d_data;
    }

    void Matrix::gpu()
    {
        d_cuda = true;
    }

    void Matrix::cpu()
    {
        d_cuda = false;
    }

    Matrix Matrix::inv()
    {
        if (d_cuda)
            return gpu_inv();

        return cpu_inv();
    }

    Matrix Matrix::gpu_inv()
    {
        Matrix mat(2, 2, 1.0);
        return mat;
    }

    Matrix Matrix::cpu_inv()
    {
        if (d_shape.first != d_shape.second)
            throw std::string("Matrix not invertible!");

        int N = d_shape.first;

        float *augmented = new float[N * N * 2];

        // create augmented matrix [A, I] and temp matrix [A] for swapping later
        for (int idx = 0; idx != N; ++idx)
            for (int jdx = 0; jdx != N; ++jdx) {
                augmented[idx * 2 * N + jdx] = at(idx, jdx);
                augmented[idx * 2 * N + jdx + N] = (idx == jdx) ? 1 : 0;
            }

        for (int idx = 0; idx != N; ++idx) {

            float pivot = augmented[idx * 2 * N + idx];
            if (pivot == 0)
                throw std::string("Matrix not invertible!");

            for (int jdx = 0; jdx != 2 * N; ++jdx)
                augmented[idx * 2 * N + jdx] /= pivot;

            for (int kdx = 0; kdx != N; ++kdx) {
                if (kdx == idx)
                    continue;
                float factor = augmented[kdx * 2 * N + idx];
                for (int jdx = 0; jdx != 2 * N; ++jdx)
                    augmented[kdx * 2 * N + jdx] -= factor * augmented[idx * 2 * N + jdx];
            }
        }

        float *inv = new float[N * N];

        for (int idx = 0; idx != N; ++idx)
            for (int jdx = 0; jdx != N; ++jdx)
                inv[idx * N + jdx] = augmented[idx * 2 * N + jdx + N];

        Matrix inverse(inv, N, N);

        delete[] augmented;
        delete[] inv;

        return inverse;
    }

    Matrix Matrix::transpose()
    {
        if (d_cuda)
            return gpu_transpose();

        return cpu_tranpose();
    }

    Matrix Matrix::cpu_tranpose()
    {

        Matrix transpose(d_shape.second, d_shape.first, 0.0);

        for (int idx = 0; idx != d_shape.first; ++idx)
            for (int jdx = 0; jdx != d_shape.second; ++jdx)
                transpose.set(jdx, idx, at(idx, jdx));

        return transpose;
    }

    Matrix Matrix::gpu_transpose()
    {
        int N = d_shape.first * d_shape.second;
        int block_rows = 8;

        float *c = new float[N];

        float *cuda_a, *cuda_c;

        cudaError_t err = cudaMalloc(&cuda_a, N * sizeof(float));
        cuda_check(err);
        err = cudaMalloc(&cuda_c, N * sizeof(float));

        err = cudaMemcpy(cuda_a, d_data, N * sizeof(float), cudaMemcpyHostToDevice);
        cuda_check(err);

        dim3 dimBlock(32, block_rows, 1);
        dim3 dimGrid((d_shape.second * 32 - 1) / 32, (d_shape.first + 32 - 1) / 32, 1);

        gpu::transpose<<<dimGrid, dimBlock>>>(cuda_a, cuda_c, d_shape.first, d_shape.second, block_rows);

        err = cudaMemcpy(c, cuda_c, N * sizeof(float), cudaMemcpyDeviceToHost);
        cuda_check(err);

        Matrix result = Matrix(c, d_shape.second, d_shape.first);

        delete[] c;

        cudaFree(cuda_a);
        cudaFree(cuda_c);

        return result;
    }

    Matrix Matrix::add(float value)
    {
        if (d_cuda)
            return gpu_add(value);

        return cpu_add(value);
    }

    Matrix Matrix::cpu_add(float value)
    {
        int x = d_shape.first;
        int y = d_shape.second;

        Matrix result(x, y, 0.0);

        for (int idx = 0; idx != x * y; ++idx)
            result.set(idx, d_data[idx] + value);

        return result;
    }

    Matrix Matrix::gpu_add(float value)
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
        int threads = 32;
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
        if (d_cuda)
            return gpu_add(matrix);

        return cpu_add(matrix);
    }

    Matrix Matrix::cpu_add(Matrix &matrix)
    {
        if (d_shape != matrix.shape())
            throw std::string("Matrix dimensions must match for addition!");

        int x = d_shape.first;
        int y = d_shape.second;

        Matrix result(x, y, 0.0);

        float *matrix_data = matrix.get_data();

        for (int idx = 0; idx != x * y; ++idx)
            result.set(idx, matrix_data[idx] + d_data[idx]);

        return result;
    }

    Matrix Matrix::gpu_add(Matrix &matrix)
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
        int threads = 32;
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

    Matrix Matrix::sub(float value)
    {
        if (d_cuda)
            return gpu_sub(value);

        return cpu_sub(value);
    }

    Matrix Matrix::cpu_sub(float value)
    {
        int x = d_shape.first;
        int y = d_shape.second;

        Matrix result(x, y, 0.0);

        for (int idx = 0; idx != x * y; ++idx)
            result.set(idx, d_data[idx] - value);

        return result;
    }

    Matrix Matrix::gpu_sub(float value)
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
        int threads = 32;
        int blocks = ceil(float(N) / threads);

        gpu::sub<<<blocks, threads>>>(cuda_a, value, cuda_c, N);

        err = cudaMemcpy(c, cuda_c, N * sizeof(float), cudaMemcpyDeviceToHost);
        cuda_check(err);

        Matrix result = Matrix(c, d_shape.first, d_shape.second);

        delete[] c;

        cudaFree(cuda_a);
        cudaFree(cuda_c);

        return result;
    }

    Matrix Matrix::sub(Matrix &matrix)
    {
        if (d_cuda)
            return gpu_sub(matrix);

        return cpu_sub(matrix);
    }

    Matrix Matrix::cpu_sub(Matrix &matrix)
    {
        if (d_shape != matrix.shape())
            throw std::string("Matrix dimensions must match for subtraction!");

        int x = d_shape.first;
        int y = d_shape.second;

        Matrix result(x, y, 0.0);

        float *matrix_data = matrix.get_data();

        for (int idx = 0; idx != x * y; ++idx)
            result.set(idx, matrix_data[idx] - d_data[idx]);

        return result;
    }

    Matrix Matrix::gpu_sub(Matrix &matrix)
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
        int threads = 32;
        int blocks = ceil(float(N) / threads);

        gpu::sub<<<blocks, threads>>>(cuda_a, cuda_b, cuda_c, N);
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

    Matrix Matrix::operator-(float value)
    {
        return sub(value);
    }

    Matrix Matrix::operator-(Matrix &matrix)
    {
        return sub(matrix);
    }

    Matrix Matrix::mult(float value)
    {
        if (d_cuda)
            return gpu_mult(value);

        return cpu_mult(value);
    }

    Matrix Matrix::cpu_mult(float value)
    {

        Matrix result(d_shape.first, d_shape.second, 0.0);

        for (int idx = 0; idx != d_shape.first; ++idx)
            for (int jdx = 0; jdx != d_shape.second; ++jdx)
                result.set(idx, jdx, at(idx, jdx) * value);

        return result;
    }

    Matrix Matrix::gpu_mult(float value)
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
        int threads = 32;
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
        if (d_cuda)
            return gpu_mult(matrix);

        return cpu_mult(matrix);
    }

    // TODO: later - make it more efficient!
    Matrix Matrix::cpu_mult(Matrix &matrix)
    {
        if (d_shape.second != matrix.shape().first)
            throw std::string("Inner dimensions must match for matmul!");

        Matrix result(d_shape.first, matrix.shape().second, 0.0);

        for (int idx = 0; idx != d_shape.first; ++idx) {
            for (int jdx = 0; jdx != matrix.shape().second; ++jdx) {
                float sum = 0;
                for (int kdx = 0; kdx != d_shape.second; ++kdx)
                    sum += at(idx, kdx) * matrix.at(kdx, jdx);
                result.set(idx, jdx, sum);
            };
        }

        return result;
    }

    Matrix Matrix::gpu_mult(Matrix &matrix)
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

    Matrix Matrix::div(float value)
    {
        if (d_cuda)
            return gpu_div(value);

        return cpu_div(value);
    }

    Matrix Matrix::cpu_div(float value)
    {
        int x = d_shape.first;
        int y = d_shape.second;

        Matrix result(x, y, 0.0);

        for (int idx = 0; idx != x * y; ++idx)
            result.set(idx, d_data[idx] / value);

        return result;
    }

    Matrix Matrix::gpu_div(float value)
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
        int threads = 32;
        int blocks = ceil(float(N) / threads);

        gpu::div<<<blocks, threads>>>(cuda_a, value, cuda_c, N);

        err = cudaMemcpy(c, cuda_c, N * sizeof(float), cudaMemcpyDeviceToHost);
        cuda_check(err);

        Matrix result = Matrix(c, d_shape.first, d_shape.second);

        delete[] c;

        cudaFree(cuda_a);
        cudaFree(cuda_c);

        return result;
    }

    Matrix Matrix::div(Matrix &matrix)
    {
        if (d_cuda)
            return gpu_div(matrix);

        return cpu_div(matrix);
    }

    Matrix Matrix::cpu_div(Matrix &matrix)
    {
        if (d_shape != matrix.shape())
            throw std::string("Matrix dimensions must match for division!");

        int x = d_shape.first;
        int y = d_shape.second;

        Matrix result(x, y, 0.0);

        float *matrix_data = matrix.get_data();

        for (int idx = 0; idx != x * y; ++idx)
            result.set(idx, matrix_data[idx] / d_data[idx]);

        return result;
    }

    Matrix Matrix::gpu_div(Matrix &matrix)
    {
        int N = d_shape.first * d_shape.second;

        if (d_shape != matrix.shape())
            throw std::string("Matrix dimensions must match for division!");

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
        int threads = 32;
        int blocks = ceil(float(N) / threads);

        gpu::div<<<blocks, threads>>>(cuda_a, cuda_b, cuda_c, N);
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

    Matrix Matrix::operator/(float value)
    {
        return div(value);
    }

    Matrix Matrix::operator/(Matrix &matrix)
    {
        return div(matrix);
    }

    void Matrix::fill(float value)
    {
        int N = d_shape.first * d_shape.second;
        // for large matrices - parallelize
        if (N > 10000) {
            float *cuda_arr;

            cudaError_t err = cudaMalloc(&cuda_arr, N * sizeof(float));
            cuda_check(err);

            int threads = 32;
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

    float *Matrix::get_data() const
    {
        return d_data;
    }

    std::pair<int, int> const &Matrix::shape() const
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

            int threads = 32;
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
        std::cout << '\n';
        if (d_shape.first * d_shape.second > 25) {
            std::cout << '[';
            for (int idx = 0; idx != 3; ++idx) {
                std::cout << '[';
                for (int jdx = 0; jdx != 3; ++jdx) {
                    std::cout << at(idx, jdx) << ", ";
                }
                std::cout << "... " << at(idx, d_shape.second - 1) << "]\n";
            }
            std::cout << "...\n[";
            for (int jdx = 0; jdx != 3; ++jdx) {
                std::cout << at(d_shape.first - 1, jdx) << ", ";
            }
            std::cout << "... " << at(d_shape.first - 1, d_shape.second - 1) << "]],";
            std::cout << " shape: (" << d_shape.first << ", " << d_shape.second << ")\n";

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
        std::cout << '\n';
    }

    void Matrix::set(int idx, float value)
    {
        if (idx >= d_shape.first * d_shape.second)
            throw std::string("Index accessed out of range!");
        d_data[idx] = value;
    }

    void Matrix::set(int row, int column, float value)
    {
        if (row * d_shape.second + column >= d_shape.first * d_shape.second)
            throw std::string("Index accessed out of range");
        d_data[row * d_shape.second + column] = value;
    }

}
