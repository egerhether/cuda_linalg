#ifndef MATRIX_INCLUDED
#define MATRIX_INCLUDED

namespace gpu {
    __global__ void add(float *a, float *b, float *result, int N);
    __global__ void add(float *a, float b, float *result, int N);
    __global__ void matmul(float *a, float val, float *result, int N);
    __global__ void matmul(float *a, float *b, float *result, int d1, int d2, int d3);
    __global__ void transpose(float *arr, float *target, int N);
    __global__ void fill(float *arr, float val, int N);
    __global__ void copy(float *arr, float *target, int N);
}

namespace linalg {
    class Matrix {

        float *d_data;
        std::pair<int, int> d_shape;

    public:
        Matrix(int rows, int columns, float value);
        Matrix(float *data, int rows, int columns);

        ~Matrix();

        void inv();
        void transpose();

        Matrix add(float value);
        Matrix add(Matrix &matrix);
        Matrix operator+(float value);
        Matrix operator+(Matrix &matrix);

        Matrix mult(float value);
        Matrix mult(Matrix &matrix);
        Matrix operator*(float value);
        Matrix operator*(Matrix &matrix);

        float at(int row, int column);

        // utils and such
        void fill(float value);
        float mean();
        std::pair<int, int> &shape();
        void print();

        float *get_data();

    private:
        void copy(float *data);
    };
}

#endif
