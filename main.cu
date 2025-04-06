#include "linalg/matrix.cuh"
#include <iostream>
#include <string>

int main()
{
    try {
        float data[] = { 1.2, 2.4, 1.1, 1.7, 5.3, 56.4 };

        linalg::Matrix matrix(data, 3, 2);
        linalg::Matrix matrix2(2, 3, 1.0);

        linalg::Matrix result = matrix * matrix2;

        matrix.print();

        matrix2.print();

        result.print();

    } catch (std::string e) {

        std::cout << e << '\n';
    }
}
