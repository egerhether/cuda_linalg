## About

Simple Cuda C++ library for linear algebra operations. `main.cu` is a benchmark of all included operations so far against cpu-based implementations. Run `./main.cu <matrix_size>` to benchmark desired matrix size. By default size of $10000$ is used with `size / 5`used for matrix multiplication, as it is very slow on the cpu.
