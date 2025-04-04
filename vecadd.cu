#include <random>

using namespace std;

__global__ void gpu_vec_add(float *a, float *b, float *c, int N) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < N)
		c[idx] = a[idx] + b[idx];
}

int main(int argc, char **argv) {

	int N;
	if (argc != 0)
		N = atoi(argv[1]);
	else
		N = 10000;

	float *a = new float[N];
	float *b = new float[N];
	float *c = new float[N];

	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<float> generator(0.0, 6.0);

	for (int idx = 0; idx != N; ++idx) {
		a[idx] = generator(gen);
		b[idx] = generator(gen);
	}

	float *cuda_a, *cuda_b, *cuda_c;
	float *c2 = new float[N];

	// memory alloc on gpu
	cudaMalloc(&cuda_a, N * sizeof(float));
	cudaMalloc(&cuda_b, N * sizeof(float));
	cudaMalloc(&cuda_c, N * sizeof(float));

	// copy vectors to gpu
	cudaMemcpy(cuda_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

	// initialize size of gpu to run on
	int threads = 256;
	int blocks = ceil(float(N) / threads);

	gpu_vec_add<<<blocks, threads>>>(cuda_a, cuda_b, cuda_c, N);

	cudaMemcpy(c2, cuda_c, N * sizeof(float), cudaMemcpyDeviceToHost);

	delete[] a;
	delete[] b;
	delete[] c;
	delete[] c2;
}
