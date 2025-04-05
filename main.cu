#include "linalg/matrix.cuh"

int main()
{
    linalg::Matrix a = linalg::Matrix(1000, 1000, 4.5);
    linalg::Matrix b = linalg::Matrix(1000, 1000, 3.5);

    linalg::Matrix c = a.add(b);
}
