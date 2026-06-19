#include <iostream>
#include <numeric>
#include <initializer_list>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/epilogue.hpp"

#include "../common.hpp"

using namespace ck_tile;

int main() {
    std::cout << "Starting CK Tile GEMM Stage 3..." << std::endl;

    index_t M = 512;
    index_t N = 512;
    index_t K = 512;

    // strides are row-major: so number of elements per row
    index_t stride_A = K;
    index_t stride_B = N;
    index_t stride_C = N;

    // CK Tile uses WMMA
    // see: https://gpuopen.com/learn/wmma_on_rdna3/
    // A and B need must be fp16, accumulate to fp32
    using ADataType   = half_t;
    using BDataType   = half_t;
    using AccDataType = float;
    using CDataType   = float;
    // specifying layout in memory - row major
    using ALayout  = tensor_layout::gemm::RowMajor;
    using BLayout  = tensor_layout::gemm::RowMajor;
    using CLayout  = tensor_layout::gemm::RowMajor;

    // CK utility class for specifying host tensor
    // first specify dimensions, next specify
    HostTensor<ADataType> h_a({M, K}, {stride_A, 1});
    HostTensor<BDataType> h_b({K, N}, {stride_B, 1});
    HostTensor<CDataType> h_c({M, N}, {stride_C, 1}); // GPU destination
    HostTensor<CDataType> cpu_c({M, N}, {stride_C, 1}); // Vblidation anchor

    // initializing data
    for(int i = 0; i < M * K; ++i) h_a.mData[i] = type_convert<ADataType>(static_cast<float>(i % 10) * 0.1f);
    for(int i = 0; i < K * N; ++i) h_b.mData[i] = type_convert<BDataType>(static_cast<float>(i % 7) * 0.1f);

    // creating device buffers
    DeviceMem d_a(h_a.get_element_space_size_in_bytes());
    DeviceMem d_b(h_b.get_element_space_size_in_bytes());
    DeviceMem d_c(h_c.get_element_space_size_in_bytes());

    // copy matrices into device buffers
    d_a.ToDevice(h_a.mData.data());
    d_b.ToDevice(h_b.mData.data());
    // C intialized to zeros
    d_c.SetZero();

    // BlockTile  = {M, N, K} per workgroup tile
    // BlockWarps = {M, N, K} warp distribution within the block
    // WarpTile   = {M, N, K} per-warp matrix-core tile
    
    // here we are tiling the matrices across blocks
    // A is split into 128x32 regions across the blocks
    // B is split into 32x128 regions across the blocks
    // each block will calculate a 128x128 region of C
    using BlockTile  = sequence<128, 128, 32>;
    // total of 4 warps per block: 2 x 2 x 1 = 4
    // 2 sets of warps split across M
    // 2 sets of warps split across N
    // 1 warp deep across K
    // so one warp is responsbile for outputting a 64x64
    // warp uses 64x32 region of A
    // warp uses 32x64 region of B
    // region of C
    using BlockWarps = sequence<2, 2, 1>;
    // each warp will compute a 16x16 tile at once
    using WarpTile   = sequence<16, 16, 16>;
    using BlockGemmShape = TileGemmShape<BlockTile, BlockWarps, WarpTile>;

    // (no padding needed: 512 is a multiple of the 128/128/32 block tile).
    using Traits = TileGemmUniversalTraits<false, false, false, /*DoubleSmemBuffer=*/false,
                                           ALayout, BLayout, CLayout>;

    using GemmProblem = UniversalGemmPipelineProblem<ADataType, BDataType, CDataType,
                                                     BlockGemmShape, Traits>;
    using GemmPipeline = GemmPipelineAgBgCrCompV3<GemmProblem>;

    // After the WMMA instructions, each lane holds its piece of the 16x16 result
    // in the register layout the matrix core dictates. Writing those fragments
    // straight to global memory would be uncoalesced (adjacent lanes don't map to
    // adjacent C elements). CShuffle stages the tile through LDS and reads it back
    // rearranged, so the final global store is coalesced (and vectorized).
    using Epilogue = CShuffleEpilogue<
        CShuffleEpilogueProblem<ADataType, BDataType, tuple<>, AccDataType, CDataType,
                                tuple<>, CLayout, element_wise::PassThrough,
                                BlockGemmShape::kM, BlockGemmShape::kN,
                                /*MWave=*/2, /*NWave=*/2,
                                /*MPerXdl=*/16, /*NPerXdl=*/16, /*KPerXdl=*/16,
                                /*isCTransposed=*/false,
                                memory_operation_enum::set>>;

    // group_size    = ceil(M0*N0 / GroupNum);   // ceil(16/8) = 2
    // group_id_y    = block_1d_id / GroupNum;    // id / 8
    // group_id_x    = block_1d_id % GroupNum;    // id % 8
    // remap_id      = group_id_x * group_size + group_id_y;   // (id%8)*2 + (id/8)
    // GemmSpatiallyLocalTilePartitioner maps each block id -> (m_tile, n_tile) in
    // two stages:
    //  1) GroupNum: re-deals block ids across "big groups" to SPREAD consecutive
    //     blocks apart  -> load-balancing across dies/shader engines (multi-die HW).
    //       group_size = ceil(M0*N0 / GroupNum)   // ceil(16/8) = 2
    //       remap_id   = (id % GroupNum)*group_size + (id / GroupNum)  // (id%8)*2 + id/8
    //  2) M01: height of a tile column-strip blocks walk down before moving right.
    //     THIS is the L2-locality knob -- a C-row of tiles reuses one A row-panel,
    //     so a strip walk keeps that panel hot in L2 and avoids global re-reads.
    //     With M01 = 1 below, this swizzle is DISABLED (plain row-major traversal).
    using TilePartitioner = GemmSpatiallyLocalTilePartitioner<BlockGemmShape, 8, 1>;

    using Kernel = GemmKernel<TilePartitioner, GemmPipeline, Epilogue>;

    // 7. Build host args and kernel args
    auto host_args = GemmHostArgs(
        d_a.GetDeviceBuffer(),
        d_b.GetDeviceBuffer(),
        d_c.GetDeviceBuffer(),
        1, // k_batch
        M, N, K,
        stride_A, stride_B, stride_C);

    auto kargs     = Kernel::MakeKernelArgs(host_args);
    dim3 grid_dim  = Kernel::GridSize(M, N, 1);
    dim3 block_dim = Kernel::BlockSize();

    if(!Kernel::IsSupportedArgument(kargs)) {
        std::cerr << "Kernel arguments are not supported for this problem." << std::endl;
        return 1;
    }

    // The kernel allocates its LDS statically on-device, so the dynamic LDS
    // byte size passed here is 0. Passing time_kernel_=true makes launch_kernel
    // run warmup + repeat launches and return the average milliseconds.
    float avg_ms = launch_kernel(stream_config{nullptr, true},
                                 make_kernel(Kernel{}, grid_dim, block_dim, 0, kargs));
    HIP_CHECK_ERROR(hipGetLastError());
    print_perf("CK Tile GEMM (fp16)", avg_ms, M, N, K);

    // Synchronize and pull results back into the destination container
    hipDeviceSynchronize();
    d_c.FromDevice(h_c.mData.data());

    // 8. Run Verification checks
    std::cout << "Validating accuracy against reference calculations..." << std::endl;

    // CPU reference check using the underlying HostTensor utilities
    reference_gemm<ADataType, BDataType, AccDataType, CDataType>(h_a, h_b, cpu_c);

    bool pass = check_err(h_c.mData, cpu_c.mData);

    if (pass) {
        std::cout << "SUCCESS! CK Tile kernel outputs verified." << std::endl;
    } else {
        std::cout << "FAILURE! Calculations mismatch detected." << std::endl;
    }

    return 0;
}
