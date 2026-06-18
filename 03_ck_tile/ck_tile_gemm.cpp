#include <iostream>
#include <numeric>
#include <initializer_list>
#include <vector>

// Include the foundational CK Tile structures
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"

// Define namespaces for ease of readability
using namespace ck_tile;

int main() {
    std::cout << "Starting CK Tile GEMM Stage 3..." << std::endl;

    // 1. Establish structural matrix dimensions
    // M x K * K x N = M x N
    index_t M = 512;
    index_t N = 512;
    index_t K = 512;

    // Define strides (Row-Major: stride equals number of elements per row)
    index_t stride_A = K;
    index_t stride_B = N;
    index_t stride_C = N;

    // 2. Specify Data Types and Matrix Layout Layouts
    // We will use standard float (fp32) matching Stages 1 & 2
    using DataType = float;
    using ALayout  = TensorLayout::RowMajor;
    using BLayout  = TensorLayout::RowMajor;
    using CLayout  = TensorLayout::RowMajor;

    // 3. Allocate Host Tensors using CK Tile's built-in host utility classes
    HostTensor<DataType> h_a({M, K}, {stride_A, 1});
    HostTensor<DataType> h_b({K, N}, {stride_B, 1});
    HostTensor<DataType> h_c({M, N}, {stride_C, 1}); // GPU destination
    HostTensor<DataType> cpu_c({M, N}, {stride_C, 1}); // Validation anchor

    // 4. Initialize Data (Using uniform linear sequence patterns)
    for(int i = 0; i < M * K; ++i) h_a::mData[i] = static_cast<float>(i % 10) * 0.1f;
    for(int i = 0; i < K * N; ++i) h_b::mData[i] = static_cast<float>(i % 7) * 0.1f;

    // 5. Establish Device (GPU) Buffers
    DeviceMem d_a(M * K * sizeof(DataType));
    DeviceMem d_b(K * N * sizeof(DataType));
    DeviceMem d_c(M * N * sizeof(DataType));

    // Copy allocated buffers from Host to Device
    d_a.ToDevice(h_a.mData.data());
    d_b.ToDevice(h_b.mData.data());

    // 6. Define the Compile-Time Structural Pipeline Configurations
    // This tells CK Tile what block sizes to leverage. 
    // It automatically maps these dimensions down to Wavefront allocations and LDS layouts.
    using BlockGemmShape = TileShape<Sequence<128, 128, 16>>; // M, N, K Tile configuration
    using BlockWarpShape = Tuple<Sequence<4, 4, 1>, Sequence<32, 32, 16>>; // Execution distribution

    // Assemble the complete unified Kernel Problem Definition
    using GemmProblem = GemmProblemTraits<
                            DataType, ALayout,
                            DataType, BLayout,
                            DataType, CLayout,
                            BlockGemmShape,
                            BlockWarpShape>;

    // Pull the appropriate blockwise execution pipeline
    using GemmKernel = GemmPipeline_Blockwise<GemmProblem>;

    // 7. Invoke Execution Configuration via the host-side launcher
    // CK Tile manages grids, block sizes, and execution kernel boundaries under the hood.
    auto kargs = GemmKernel::MakeKargs(
        d_a.GetDeviceBuffer(),
        d_b.GetDeviceBuffer(),
        d_c.GetDeviceBuffer(),
        M, N, K,
        stride_A, stride_B, stride_C
    );

    dim3 grid_dim = GemmKernel::GridSize(M, N, K);
    dim3 block_dim = GemmKernel::BlockSize();

    // Launch the instantiated metadata template kernel
    hipLaunchKernelGGL(
        GemmKernel::Kernel,
        grid_dim,
        block_dim,
        0, 0, // Shared memory byte sizes, stream tracks
        kargs
    );

    // Synchronize to block host thread execution until GPU operations resolve
    hipDeviceSynchronize();

    // Pull calculations back into destination container
    d_c.FromDevice(h_c.mData.data());

    // 8. Run Verification checks
    std::cout << "Validating accuracy against reference calculations..." << std::endl;
    
    // CPU reference check using the underlying HostTensor utilities
    reference_gemm(h_a, h_b, cpu_c);

    bool pass = check_error(h_c, cpu_c);

    if (pass) {
        std::cout << "SUCCESS! CK Tile kernel outputs verified." << std::endl;
    } else {
        std::cout << "FAILURE! Calculations mismatch detected." << std::endl;
    }

    return 0;
}