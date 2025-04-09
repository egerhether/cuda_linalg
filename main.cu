#include "linalg/matrix.cuh"
#include <chrono>
#include <iostream>
#include <string>

int main(int argc, char **argv)
{
    try {
        int size = 10000;

        if (argc != 1 && strcmp(argv[1], "-d") == 0) {
            linalg::Matrix test(5, 4, 1.0);
            linalg::Matrix test2(4, 4, 1.4);
            test.fill_random();
            test2.fill_random();
            linalg::Matrix gpu_t = test * test2;
            gpu_t.print();
            test.cpu();
            linalg::Matrix cpu_t = test * test2;
            cpu_t.print();
            linalg::Matrix diff = gpu_t - cpu_t;
            diff.print();
            std::cout << diff.norm() << '\n';
            return 0;
        }

        if (argc != 1)
            size = std::stoi(argv[1]);

        linalg::Matrix mat(size, size, 1.0);

        // benchmark transpose
        std::cout << "Tranpose benchmark\n";
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
        linalg::Matrix transpose_cpu = mat.transpose();
        end = std::chrono::steady_clock::now();
        time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        std::cout << "[CPU] Transposing matrix of size " << size << " took " << time_diff << " ms.\n";

        // difference
        linalg::Matrix diff = transpose - transpose_cpu;
        float err = diff.norm();
        std::cout << "Difference between results: " << err << "\n\n";

        // bechmark addition
        std::cout << "Addition benchmark\n";
        linalg::Matrix to_add(size, size, 4.3);
        // gpu
        to_add.gpu();
        mat.gpu();
        begin = std::chrono::steady_clock::now();
        linalg::Matrix sum = mat + to_add;
        end = std::chrono::steady_clock::now();
        time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        std::cout << "[GPU] Matrix addition of size " << size << " took " << time_diff << " ms.\n";

        // cpu
        to_add.cpu();
        mat.cpu();
        begin = std::chrono::steady_clock::now();
        linalg::Matrix sum_cpu = mat + to_add;
        end = std::chrono::steady_clock::now();
        time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        std::cout << "[CPU] Matrix addition of size " << size << " took " << time_diff << " ms.\n";

        // difference
        diff = sum - sum_cpu;
        err = diff.norm();
        std::cout << "Difference between results: " << err << "\n\n";

        // benchmark matmul
        std::cout << "Matmul benchmark\n";
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
        linalg::Matrix prod_cpu = mult_mat * to_mult;
        end = std::chrono::steady_clock::now();
        time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        std::cout << "[CPU] Matrix multiplication of size " << size / 5 << " took " << time_diff << " ms.\n";

        // difference
        diff = prod - prod_cpu;
        err = diff.norm();
        std::cout << "Difference between results: " << err << "\n\n";

        // benchmark inverse
        std::cout << "Inverse benchmark\n";
        linalg::Matrix to_inv(size / 5, size / 5, 1.0);
        to_inv.fill_random();
        // gpu
        to_inv.gpu();
        begin = std::chrono::steady_clock::now();
        linalg::Matrix inverse = to_inv.inv();
        end = std::chrono::steady_clock::now();
        time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        std::cout << "[GPU] Matrix inverse of size " << size / 5 << " took " << time_diff << " ms.\n";

        // cpu
        to_inv.cpu();
        begin = std::chrono::steady_clock::now();
        linalg::Matrix inverse_cpu = to_inv.inv();
        end = std::chrono::steady_clock::now();
        time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        std::cout << "[CPU] Matrix inverse of size " << size / 5 << " took " << time_diff << " ms.\n";

        // difference
        diff = inverse - inverse_cpu;
        err = diff.norm();
        std::cout << "Difference between results: " << err << "\n\n";

    } catch (std::string e) {

        std::cout << e << '\n';
    } catch (std::logic_error e) {
        std::cout << e.what() << '\n';
    }
}
