#include "../common.hpp"

constexpr int M = 512;
constexpr int K = 512;
constexpr int N = 512;


__global__ void naive_gemm_kernel(const float* A, const float* B, float* C, int m, int n, int k) {
    // get the current column index of B that this thread corresponds to
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // get the current row index of A that his thread corresponds to
    int row = blockIdx.y * blockDim.y + threadIdx.y

    if (row < m && col < n) {
        float sum = 0.0f;
        
        for (int l = 0; l < k; ++l) {
            // memory is allocated in a linear fashion on the GPU
            // (row * k) and (l * n) account for this fact
            sum += A[row * k + l] * B[l * n + col];
        }
        
        C[row * n + col] = sum;
    }
}

int main() {
    std::cout << "Starting Naive GEMM (" << M << "x" << K << "x" << N << ")..." << std::endl;

    // the size in bytes of each matrix
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // initialize host side matrices, stored in 1-D, row-major order
    // i.e:
    // | a, b |
    // | c, d | 
    // -> [a, b, c, d] 
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    // initialize two variables for matrix C, calculation and verification
    std::vector<float> h_C(M * N, 0.0f);
    std::vector<float> cpu_C(M * N, 0.0f);

    // initialize the matrices with values
    init_matrix(h_A, M * K, 10);
    init_matrix(h_B, K * N, 7);

    // allocate memory on GPU
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    HIP_CHECK(hipMalloc(&d_A, size_A));
    HIP_CHECK(hipMalloc(&d_B, size_B));
    HIP_CHECK(hipMalloc(&d_C, size_C));

    // copy memory from host to GPU
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), size_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), size_B, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_C, 0, size_C));

    // each GPU block contains 16 x 16 = 256 threads
    dim3 threadsPerBlock(16, 16);
    // this determines the number of blocks as ceil(N / 16)
    // and ceil(M / 16), so 512/16 = 32 and 512/16 = 32
    // 32 x 32 = 1024 blocks in total
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // kernel launch macro...
    // args are kernel to execute: naive_gemm_kernel
    // block grid
    // thread grid
    // additional shared memory to allocate
    // hipStream: 0 -> NULL stream
    // kernel args A, B, C, m, n, and k
    hipLaunchKernelGGL(naive_gemm_kernel, numBlocks, threadsPerBlock, 0, 0, d_A, d_B, d_C, M, N, K);

    // check for errors and wait for kernel to finish
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // copy data from GPU back to host
    HIP_CHECK(hipMemcpy(h_C.data(), d_C, size_C, hipMemcpyDeviceToHost));

    // perform CPU-side verification
    std::cout << "Verifying..." << std::endl;
    cpu_gemm(h_A, h_B, cpu_C, M, N, K);
    
    if (verify_results(h_C, cpu_C, M, N)) {
        std::cout << "SUCCESS!" << std::endl;
    } else {
        std::cout << "FAILURE!" << std::endl;
    }

    // free memory on the GPU
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));

    return 0;
}