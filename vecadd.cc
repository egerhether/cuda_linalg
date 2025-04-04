#include <random>

using namespace std;

void cpu_vec_add(float *a, float *b, float *c, int N) {
	for (int idx = 0; idx != N; ++idx)
		c[idx] = a[idx] + b[idx];
}

int main(int argc, char **argv)
{
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

	cpu_vec_add(a, b, c, N);

}
