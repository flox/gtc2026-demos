// gpu_gemm.cu — Tiled SGEMM benchmark
//
// Demonstrates Flox-managed CUDA C++ toolchain (nvcc + gcc13 + cmake).
// Runs a shared-memory tiled matrix multiply, verifies against CPU reference,
// and reports timing + GFLOPS.
//
// Usage:  gpu-gemm [N]     (default N = 2048)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static const int TILE = 16;

// ---------------------------------------------------------------------------
// Tiled SGEMM kernel  (C = A * B,  all row-major,  NxN square)
// ---------------------------------------------------------------------------
__global__ void sgemm_tiled(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int N)
{
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] = (row < N && aCol < N) ? A[row * N + aCol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

// ---------------------------------------------------------------------------
// CPU reference  (naive O(N^3))
// ---------------------------------------------------------------------------
static void sgemm_cpu(const float* A, const float* B, float* C, int N)
{
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

// ---------------------------------------------------------------------------
// Verify GPU result against CPU reference
// ---------------------------------------------------------------------------
static bool verify(const float* gpu, const float* cpu, int N)
{
    // Use relative tolerance that scales with matrix size.
    // FP32 accumulates error proportional to N for dot products.
    float tol = 1e-3f * N;
    for (int i = 0; i < N * N; ++i) {
        float diff = fabsf(gpu[i] - cpu[i]);
        float denom = fmaxf(fabsf(cpu[i]), 1.0f);
        if (diff / denom > tol) {
            fprintf(stderr, "Mismatch at [%d]: gpu=%.6f  cpu=%.6f  diff=%.6f\n",
                    i, gpu[i], cpu[i], diff);
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Print device properties
// ---------------------------------------------------------------------------
static void print_device_info()
{
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count == 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        exit(EXIT_FAILURE);
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("============================================================\n");
    printf("  GPU Device Information\n");
    printf("============================================================\n");
    printf("  Device         : %s\n", prop.name);
    printf("  Compute cap.   : %d.%d\n", prop.major, prop.minor);
    printf("  SMs            : %d\n", prop.multiProcessorCount);
    printf("  Global memory  : %.1f GB\n",
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Clock rate     : %d MHz\n", prop.clockRate / 1000);
    printf("  Memory clock   : %d MHz\n", prop.memoryClockRate / 1000);
    printf("  Memory bus     : %d-bit\n", prop.memoryBusWidth);
    printf("------------------------------------------------------------\n");
    printf("  Built with gcc : %d.%d.%d\n", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
    printf("  CUDA version   : %d.%d\n", __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__);
    printf("============================================================\n\n");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int N = 2048;
    if (argc > 1) {
        N = atoi(argv[1]);
        if (N <= 0 || N > 16384) {
            fprintf(stderr, "Usage: %s [N]  (64 <= N <= 16384, default 2048)\n", argv[0]);
            return 1;
        }
    }

    // --- Device info ---
    print_device_info();

    size_t bytes = (size_t)N * N * sizeof(float);

    // --- Host allocations ---
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);   // GPU result
    float* h_ref = NULL;                  // CPU reference (only for small N)

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    // Initialize with reproducible random values
    srand(42);
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = (float)(rand() % 1000) / 500.0f - 1.0f;   // [-1, 1]
        h_B[i] = (float)(rand() % 1000) / 500.0f - 1.0f;
    }

    // --- Device allocations ---
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // --- Warm up ---
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    sgemm_tiled<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Timed run ---
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int RUNS = 10;
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < RUNS; ++r)
        sgemm_tiled<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / RUNS;

    // FLOPS = 2 * N^3 (multiply + add per output element)
    double flops = 2.0 * (double)N * (double)N * (double)N;
    double gflops = flops / (avg_ms * 1e6);

    // --- Copy result back ---
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // --- Verify against CPU for small matrices ---
    bool verified = false;
    bool correct = false;
    if (N <= 2048) {
        h_ref = (float*)malloc(bytes);
        if (h_ref) {
            sgemm_cpu(h_A, h_B, h_ref, N);
            correct = verify(h_C, h_ref, N);
            verified = true;
        }
    }

    // --- Report ---
    printf("SGEMM Benchmark  (%d x %d)\n", N, N);
    printf("------------------------------------------------------------\n");
    printf("  Kernel time    : %.3f ms  (avg of %d runs)\n", avg_ms, RUNS);
    printf("  Performance    : %.1f GFLOPS\n", gflops);
    if (verified)
        printf("  Verification   : %s\n", correct ? "PASS" : "FAIL");
    else
        printf("  Verification   : skipped (N > 2048)\n");
    printf("============================================================\n");

    // --- Cleanup ---
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_ref);

    return (verified && !correct) ? 1 : 0;
}
