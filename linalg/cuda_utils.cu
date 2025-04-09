#include "cuda_utils.cuh"
#include <string>

void cuda_check(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess) {
        std::string msg = "CUDA ERROR: " + std::string(cudaGetErrorString(err));
        msg += std::string(" in file ") + std::string(file);
        msg += std::string(" and line ") + std::to_string(line);
        throw msg;
    }
}
