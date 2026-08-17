#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err_ = (call);                                         \
        if (err_ != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                \
                    cudaGetErrorName(err_), __FILE__, __LINE__,            \
                    cudaGetErrorString(err_));                             \
            return 1;                                                      \
        }                                                                  \
    } while (0)

#define CUDA_CHECK_KERNEL()                 \
    do {                                    \
        CUDA_CHECK(cudaGetLastError());     \
        CUDA_CHECK(cudaDeviceSynchronize());\
    } while (0)

__global__ void saxpy_kernel(const float *x, float *y, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        y[i] = 2.0f * x[i] + y[i];
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <n>\n", argv[0]);
        return 1;
    }

    char *end = nullptr;
    long parsed = strtol(argv[1], &end, 10);
    if (*end != '\0' || parsed < 0) {
        fprintf(stderr, "n must be a non-negative integer\n");
        return 1;
    }

    int n = (int)parsed;
    if ((long)n != parsed) {
        fprintf(stderr, "n is too large\n");
        return 1;
    }

    if (n == 0) {
        printf("SUM=0\n");
        return 0;
    }

    size_t bytes = (size_t)n * sizeof(float);

    float *h_x = (float *)malloc(bytes);
    float *h_y = (float *)malloc(bytes);
    if (!h_x || !h_y) {
        fprintf(stderr, "host allocation failed\n");
        free(h_x);
        free(h_y);
        return 1;
    }

    for (int i = 0; i < n; i++) {
        h_x[i] = ((i % 2048) - 1024) * 0.5f;
        h_y[i] = (i % 1024) - 512;
    }

    float *d_x = nullptr;
    float *d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    CUDA_CHECK(cudaEventRecord(start));
    saxpy_kernel<<<blocks, threads>>>(d_x, d_y, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));

    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += (double)h_y[i];
    }

    printf("SUM=%.0f n=%d kernel_ms=%.3f\n", sum, n, kernel_ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_x);
    cudaFree(d_y);
    free(h_x);
    free(h_y);
    return 0;
}
