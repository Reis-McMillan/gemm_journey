#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <hip/hip_runtime.h>

// Macro for simple HIP error checking
#define HIP_CHECK(command) \
    { \
        hipError_t status = command; \
        if (status != hipSuccess) { \
            std::cerr << "HIP Error: " << hipGetErrorString(status) \
                      << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

// performs matrix multiplication on the CPU
inline void cpu_gemm(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int l = 0; l < K; ++l) {
                sum += A[i * K + l] * B[l * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}


inline bool verify_results(const std::vector<float>& gpu_res, const std::vector<float>& cpu_res, int M, int N, float tolerance = 1e-3f) {
    for (int i = 0; i < M * N; ++i) {
        if (std::abs(gpu_res[i] - cpu_res[i]) > tolerance) {
            std::cerr << "Mismatch at index " << i << " -> GPU: " << gpu_res[i] << ", CPU: " << cpu_res[i] << std::endl;
            return false;
        }
    }
    return true;
}

// inits matrix from [0, mod_factor)
// casts them as a float decimal; e.g. 1 -> .1
inline void init_matrix(std::vector<float>& mat, int size, int mod_factor) {
    for (int i = 0; i < size; ++i) {
        mat[i] = static_cast<float>(i % mod_factor) * 0.1f;
    }
}

// Times a kernel launch (passed as a callable) on the GPU using HIP events.
// Runs `warmup` untimed launches first, then averages over `iters` timed
// launches. Returns the average milliseconds per launch.
template <typename Launch>
inline float time_kernel_ms(Launch launch, int warmup = 5, int iters = 50) {
    for (int i = 0; i < warmup; ++i) launch();
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < iters; ++i) launch();
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float total_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&total_ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    return total_ms / iters;
}

// Prints average runtime and the corresponding throughput for an MxNxK GEMM.
// A GEMM performs 2*M*N*K floating point operations (one multiply + one add).
inline void print_perf(const std::string& name, float avg_ms, int M, int N, int K) {
    double gflops = (2.0 * M * N * K) / (avg_ms / 1000.0) / 1e9;
    std::cout << name << " | " << avg_ms << " ms/iter | " << gflops << " GFLOP/s" << std::endl;
}