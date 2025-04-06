#include "linalg/matrix.cuh"
#include <iostream>

int main()
{
    linalg::Matrix a = linalg::Matrix(1000, 2000, 4.5);
    linalg::Matrix b = linalg::Matrix(2000, 1000, 4.2);

    linalg::Matrix c = a.mult(b);

    std::cout << c.shape().first << ' ' << c.shape().second << '\n';
}
