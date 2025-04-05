#include "vector.cuh"

namespace linalg {

    Vector::Vector(float *data, int length)
        : d_data(data, 1, length)
    {
        d_length = length;
    }

    Vector::Vector(int length, float value)
        : d_data(1, length, value)
    {
        d_length = length;
    }

    Vector::Vector(Matrix &matrix)
        : d_data(matrix)
    {
        std::pair<int, int> shape = matrix.shape();
        d_length = shape.first * shape.second;
    }

    Vector Vector::add(float value)
    {
        Matrix result = d_data.add(value);
        Vector vec_result(result);
        return vec_result;
    }

    
}
