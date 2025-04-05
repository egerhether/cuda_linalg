#ifndef MATRIX_INCLUDED
#define MATRIX_INCLUDED

namespace gpu {
    __global__ void add(float *a, float *b, float *result, int N);
    __global__ void add(float *a, float b, float *result, int N);
    __global__ void matmul(float *a, float *b, float *result, int N);
    __global__ void fill(float *arr, float val, int N);
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

        Matrix add(float value);
        Matrix add(Matrix &matrix);

        Matrix mult(float value);
        Matrix mult(Matrix &matrix);

        // utils and such
        void fill(float value);
        float mean();
        std::pair<int, int> &shape();

        float *get_data();
    };
}

#endif
