// Compare two hand-computed dot products. The input buffer is row-major.
#include <cstdio>
#include <cstdlib>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define BOTH __host__ __device__
#else
#define BOTH
#endif

BOTH float row_dot(const float* row, const float* x, int cols) {
    float sum = 0.0f;
    for (int j = 0; j < cols; ++j) {
        sum += row[j] * x[j];
    }
    return sum;
}

#ifdef __CUDACC__
__global__ void gemv(const float* w, const float* x, float* y, int rows, int cols) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows) {
        y[i] = row_dot(w + i * cols, x, cols);
    }
}

void check(cudaError_t error) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(error));
        std::exit(1);
    }
}
#endif

int main() {
    constexpr int rows = 2, cols = 3;
    // Two rows of W followed by x. The formula is y = W x.
    const float input[] = {2, -1, 3, 0, 4, 1, 1, 2, 1};
    float y[rows] = {};
#ifdef __CUDACC__
    const char* backend = "CUDA";
    float* buffer = nullptr;
    check(cudaMalloc(&buffer, sizeof(input) + sizeof(y)));
    check(cudaMemcpy(buffer, input, sizeof(input), cudaMemcpyHostToDevice));
    const float* x = buffer + rows * cols;
    float* device_y = buffer + rows * cols + cols;
    constexpr int threads = 128;
    gemv<<<(rows + threads - 1) / threads, threads>>>(buffer, x, device_y, rows, cols);
    check(cudaGetLastError());
    check(cudaMemcpy(y, device_y, sizeof(y), cudaMemcpyDeviceToHost));
    check(cudaFree(buffer));
#else
    const char* backend = "CPU";
    for (int i = 0; i < rows; ++i) {
        y[i] = row_dot(input + i * cols, input + rows * cols, cols);
    }
#endif
    std::printf("%s: [%g, %g]\n", backend, y[0], y[1]);
    return (y[0] == 3.0f && y[1] == 9.0f) ? 0 : 1;
}
