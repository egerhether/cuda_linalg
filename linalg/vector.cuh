#ifndef VECTOR_INCLUDED
#define VECTOR_INCLUDED

#include "matrix.cuh"

namespace linalg {

    class Vector {
        Matrix d_data;
        int length;

    public:
        Vector(float *d_data);
        Vector(int length);
        Vector(Matrix &matrix);

        Vector &add(float value);
        Vector &add(Vector &vector);

        float inner(Vector &vector);
        Matrix &outer(Vector &vector);
    };
}

#endif
