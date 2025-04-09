#ifndef MATRIX_INCLUDED
#define MATRIX_INCLUDED

namespace gpu {
    __global__ void add(float *a, float *b, float *result, int N);
    __global__ void add(float *a, float b, float *result, int N);
    __global__ void sub(float *a, float *b, float *result, int N);
    __global__ void sub(float *a, float b, float *result, int N);
    __global__ void matmul(float *a, float val, float *result, int N);
    __global__ void matmul(float *a, float *b, float *result, int d1, int d2, int d3);
    __global__ void div(float *a, float val, float *result, int N);
    __global__ void div(float *a, float *b, float *result, int N);
    __global__ void transpose(float *arr, float *target, int d1, int d2, int block_rows);
    __global__ void fill(float *arr, float val, int N);
    __global__ void copy(float *arr, float *target, int N);
}

namespace linalg {
    class Matrix {

        float *d_data;
        std::pair<int, int> d_shape;
        bool d_cuda;

    public:
        Matrix(int rows, int columns, float value);
        Matrix(float *data, int rows, int columns);
        Matrix(Matrix const &other);

        Matrix &operator=(Matrix const &other);

        ~Matrix();

        Matrix inv();
        Matrix transpose();

        Matrix add(float value);
        Matrix add(Matrix &matrix);
        Matrix operator+(float value);
        Matrix operator+(Matrix &matrix);

        Matrix sub(float value);
        Matrix sub(Matrix &matrix);
        Matrix operator-(float value);
        Matrix operator-(Matrix &matrix);

        Matrix mult(float value);
        Matrix mult(Matrix &matrix);
        Matrix operator*(float value);
        Matrix operator*(Matrix &matrix);

        Matrix div(float value);
        Matrix div(Matrix &matrix);
        Matrix operator/(float value);
        Matrix operator/(Matrix &matrix);

        float at(int row, int column);

        // utils and such
        void fill(float value);
        float mean();
        std::pair<int, int> const &shape() const;
        void print();
        void set(int idx, float value);
        void set(int row, int column, float value);

        float *get_data() const;

        void gpu();
        void cpu();

    private:
        void copy(float *data);

        Matrix cpu_inv();
        Matrix gpu_inv();

        Matrix cpu_tranpose();
        Matrix gpu_transpose();

        Matrix cpu_add(float value);
        Matrix cpu_add(Matrix &matrix);
        Matrix gpu_add(float value);
        Matrix gpu_add(Matrix &matrix);

        Matrix cpu_sub(float value);
        Matrix cpu_sub(Matrix &matrix);
        Matrix gpu_sub(float value);
        Matrix gpu_sub(Matrix &matrix);

        Matrix cpu_mult(float value);
        Matrix cpu_mult(Matrix &matrix);
        Matrix gpu_mult(float value);
        Matrix gpu_mult(Matrix &matrix);

        Matrix cpu_div(float value);
        Matrix cpu_div(Matrix &matrix);
        Matrix gpu_div(float value);
        Matrix gpu_div(Matrix &matrix);
    };
}

#endif
