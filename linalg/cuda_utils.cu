#include "cuda_utils.cuh"
#include <string>

void cuda_check(cudaError_t err)
{
    if (err != cudaSuccess)
        throw std::string(cudaGetErrorString(err));
}
