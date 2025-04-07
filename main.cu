#include "linalg/matrix.cuh"
#include <chrono>
#include <iostream>
#include <string>

int main(int argc, char **argv)
{
    try {
        int size = 10000;
        if (argc != 1)
            size = std::stoi(argv[1]);

        linalg::Matrix mat(size, size, 1.0);

        // benchmark transpose
        // gpu
        mat.gpu();
        auto begin = std::chrono::steady_clock::now();
        linalg::Matrix transpose = mat.transpose();
        auto end = std::chrono::steady_clock::now();
        auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        std::cout << "[GPU] Transposing matrix of size " << size << " took " << time_diff << " ms.\n";

        // cpu
        mat.cpu();
        begin = std::chrono::steady_clock::now();
        transpose = mat.transpose();
        end = std::chrono::steady_clock::now();
        time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        std::cout << "[CPU] Transposing matrix of size " << size << " took " << time_diff << " ms.\n";

        // bechmark addition
        linalg::Matrix to_add(size, size, 4.3);
        // gpu
        to_add.gpu();
        mat.gpu();
        begin = std::chrono::steady_clock::now();
        linalg::Matrix sum = mat + to_add;
        end = std::chrono::steady_clock::now();
        time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        std::cout << "[GPU] Matrix addtion of size " << size << " took " << time_diff << " ms.\n";

        // cpu
        to_add.cpu();
        mat.cpu();
        begin = std::chrono::steady_clock::now();
        sum = mat + to_add;
        end = std::chrono::steady_clock::now();
        time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        std::cout << "[CPU] Matrix addtion of size " << size << " took " << time_diff << " ms.\n";

        // benchmark matmul
        // smaller matrices as it is a heavier operation
        linalg::Matrix to_mult(size / 5, size / 5, 3.2);
        linalg::Matrix mult_mat = linalg::Matrix(size / 5, size / 5, 1.0);
        // gpu
        to_mult.gpu();
        mult_mat.gpu();
        begin = std::chrono::steady_clock::now();
        linalg::Matrix prod = mult_mat * to_mult;
        end = std::chrono::steady_clock::now();
        time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        std::cout << "[GPU] Matrix multiplication of size " << size / 5 << " took " << time_diff << " ms.\n";

        // cpu
        to_mult.cpu();
        mult_mat.cpu();
        begin = std::chrono::steady_clock::now();
        prod = mult_mat * to_mult;
        end = std::chrono::steady_clock::now();
        time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        std::cout << "[CPU] Matrix multiplication of size " << size / 5 << " took " << time_diff << " ms.\n";

    } catch (std::string e) {

        std::cout << e << '\n';
    } catch (std::logic_error e) {
        std::cout << e.what() << '\n';
    }
}
