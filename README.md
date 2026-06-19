# GEMM Journey

A progressive tour of matrix multiplication (`C = A × B`) on AMD GPUs with HIP,
going from a textbook kernel to the hardware matrix cores. Each stage computes
the same 512×512×512 GEMM, verifies the result against a CPU reference, and
prints its runtime and throughput.

| Stage | Executable | Technique |
|-------|-----------|-----------|
| 1 | `01_naive_gemm`  | One thread per output element, all reads from global memory |
| 2 | `02_tiled_gemm`  | Shared-memory (LDS) tiling to reuse data across a block |
| 3 | `03_ck_tile_gemm`| [CK Tile](https://github.com/ROCm/composable_kernel) library using the WMMA matrix cores |

## Prerequisites

- **ROCm 7.x** (developed against 7.2.1). The CK Tile headers ship inside ROCm
  at `/opt/rocm/include/ck_tile`, so no separate Composable Kernel checkout is
  needed.
- **CMake ≥ 3.18**
- An **AMD RDNA3 (gfx11) GPU**.

### A note on this GPU (gfx1102 → gfx1100)

This machine has a **gfx1102** card, which ROCm does not officially support. The
standard workaround is to make it present itself as the supported **gfx1100**:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
```

Because the runtime then loads **gfx1100** code objects, the device code must be
**compiled for gfx1100** to match — otherwise the loader fails at runtime with
*"No compatible code objects found for gfx1100."* This is why `CMakeLists.txt`
pins the target architecture:

```cmake
set(GPU_TARGET "gfx1100" CACHE STRING "AMD GPU architecture to compile device code for")
```

If you are on a different GPU, override it at configure time, e.g.
`-DGPU_TARGET=gfx1100` for a 7900-series card, or `-DGPU_TARGET=gfx90a` for CDNA.

## Build

From the repository root:

```bash
mkdir -p build && cd build
cmake .. -DCK_REPO_DIR=/opt/rocm
make -j$(nproc)
```

- `CK_REPO_DIR` points at the directory whose `include/ck_tile` holds the CK Tile
  headers. With ROCm's bundled copy that is simply `/opt/rocm`.
- To build for a different GPU, add `-DGPU_TARGET=<arch>` (see note above).

The three executables are produced in the `build/` directory.

## Run

Make sure the GFX override is set (see above), then run any stage from `build/`:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0   # if not already in your environment

./01_naive_gemm
./02_tiled_gemm
./03_ck_tile_gemm
```

Each program initializes the matrices, runs the GPU kernel (with warmup +
averaged timed launches), checks the output against a CPU GEMM, and reports
performance. Example output:

```
Naive GEMM          | 0.527 ms/iter | 509  GFLOP/s
Tiled LDS GEMM      | 0.214 ms/iter | 1253 GFLOP/s
CK Tile GEMM (fp16) | 0.069 ms/iter | 3869 GFLOP/s
```

(Throughput is `2·M·N·K / time`; exact numbers vary by GPU and clocks.)

## Notes

- **Stage 3 uses fp16 inputs with fp32 accumulation.** The gfx11 WMMA matrix
  cores have no fp32-input instruction, so a pure-fp32 matrix-core GEMM is not
  possible on this hardware — Stages 1–2 (fp32, regular ALUs) and Stage 3
  (fp16, matrix cores) are therefore not a precision-for-precision comparison.
- Shared helpers (`time_kernel_ms`, `print_perf`, CPU reference, verification)
  live in `common.hpp`.
