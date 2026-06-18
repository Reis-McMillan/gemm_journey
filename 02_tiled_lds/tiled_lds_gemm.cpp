#include "../common.hpp"

constexpr int M = 512;
constexpr int K = 512;
constexpr int N = 512;

constexpr int TILE_SIZE = 16;

__global__ void tiled_gemm_kernel(const float* A, const float* B, float* C, int m, int n, int k) {
    // Allocate static local shared memory (LDS) for the current tiles
    // This memory is shared among all threads within a single block
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    // Local thread indices inside the current block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global row and column indices mapping to Matrix C
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    // loops over the 16x16 tiles to compute 16x16 tile of C
    // e.g. to compute a 16x16 tile of C, we need to matrix multiply
    // 16 x K region of A and K x 16 region of B... this loops over 
    // the various 16x16 tiles of each region  
    for (int p = 0; p < (k + TILE_SIZE - 1) / TILE_SIZE; ++p) {
        
        // 1. COOPERATIVE LOAD: Threads work together to load data into LDS
        // Load element from Matrix A into tileA
        if (row < m && (p * TILE_SIZE + tx) < k) {
            tileA[ty][tx] = A[row * k + (p * TILE_SIZE + tx)];
        } else {
            tileA[ty][tx] = 0.0f; // Zero-pad if out of matrix bounds
        }

        // Load element from Matrix B into tileB
        if (col < n && (p * TILE_SIZE + ty) < k) {
            tileB[ty][tx] = B[(p * TILE_SIZE + ty) * n + col];
        } else {
            tileB[ty][tx] = 0.0f; // Zero-pad if out of matrix bounds
        }

        // 2. SYNCHRONIZE: Barrier execution
        // We MUST wait until all threads in the block have finished writing to LDS
        // before any thread attempts to read from it!
        __syncthreads();

        // 3. COMPUTE: Multiply the two localized tiles together
        for (int l = 0; l < TILE_SIZE; ++l) {
            sum += tileA[ty][l] * tileB[l][tx];
        }

        // 4. SYNCHRONIZE: Barrier execution
        // We MUST wait until all threads are completely done reading from LDS
        // before we loop around and let the next phase overwrite these tiles!
        __syncthreads();
    }

    // Write the final accumulated dot product to global memory
    if (row < m && col < n) {
        C[row * n + col] = sum;
    }
}

int main() {
    std::cout << "Starting Tiled LDS GEMM (" << M << "x" << K << "x" << N << ")..." << std::endl;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);
    std::vector<float> cpu_C(M * N, 0.0f);

    init_matrix(h_A, M * K, 10);
    init_matrix(h_B, K * N, 7);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    HIP_CHECK(hipMalloc(&d_A, size_A));
    HIP_CHECK(hipMalloc(&d_B, size_B));
    HIP_CHECK(hipMalloc(&d_C, size_C));

    HIP_CHECK(hipMemcpy(d_A, h_A.data(), size_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), size_B, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_C, 0, size_C));

    // Notice our block dimensions strictly match TILE_SIZE (16x16)
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, 
                   (M + TILE_SIZE - 1) / TILE_SIZE);

    hipLaunchKernelGGL(tiled_gemm_kernel, numBlocks, threadsPerBlock, 0, 0, d_A, d_B, d_C, M, N, K);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_C.data(), d_C, size_C, hipMemcpyDeviceToHost));

    std::cout << "Verifying..." << std::endl;
    cpu_gemm(h_A, h_B, cpu_C, M, N, K);
    
    if (verify_results(h_C, cpu_C, M, N)) {
        std::cout << "SUCCESS! Tiled LDS kernel matches CPU." << std::endl;
    } else {
        std::cout << "FAILURE!" << std::endl;
    }

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));

    return 0;
}