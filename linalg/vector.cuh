#ifndef VECTOR_INCLUDED
#define VECTOR_INCLUDED

#include "matrix.cuh"

namespace linalg {

    class Vector {
        Matrix d_data;
        int d_length;

    public:
        Vector(float *data, int size);
        Vector(int length, float value);
        Vector(Matrix &matrix);

        Vector add(float value);
        Vector add(Vector &vector);

        float inner(Vector &vector);
        Matrix &outer(Vector &vector);
    };
}

#endif
