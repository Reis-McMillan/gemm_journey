#pragma once

#include <iostream>
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