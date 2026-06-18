cmake_minimum_required(VERSION 3.18)
project(HIP_GEMM_Complete_Journey LANGUAGES CXX)

# 1. Require the ROCm / HIP compiler ecosystem
find_package(hip REQUIRED)

# 2. Add optimization levels and modern C++ requirements
# NOTE: Composable Kernel requires C++17 or C++20 due to its advanced template structures
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

# 3. LOCATE THE COMPOSABLE KERNEL HEADERS
# We define a cache variable pointing to where you cloned the repo.
# You can override this from the command line if your path is different!
set(CK_REPO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/composable_kernel" CACHE PATH "Path to Composable Kernel repository")

# Verify the path exists so you get a clear error if it's missing
if(NOT EXISTS "${CK_REPO_DIR}")
    message(FATAL_ERROR "Could not find the Composable Kernel repo at: ${CK_REPO_DIR}. Please set -DCK_REPO_DIR=/path/to/repo")
endif()

# 4. PULL THE HEADER DIRECTORIES
# This tells the compiler exactly where to look when it hits #include <ck_tile/...>
include_directories(${CMAKE_CURRENT_SOURCE_DIR}) # For your common.hpp
include_directories("${CK_REPO_DIR}/include")    # Core CK library headers
include_directories("${CK_REPO_DIR}/include/ck_tile") # CK Tile specific engine headers

# -----------------------------------------------------------------------------
# TARGETS: Compile all the files
# -----------------------------------------------------------------------------

# STAGE 1: Naive GEMM
add_executable(01_naive_gemm 01_naive/naive_gemm.cpp)
target_link_libraries(01_naive_gemm PRIVATE hip::device)

# STAGE 2: Tiled LDS GEMM
add_executable(02_tiled_gemm 02_tiled_lds/tiled_gemm.cpp)
target_link_libraries(02_tiled_gemm PRIVATE hip::device)

# STAGE 3: CK Tile GEMM
add_executable(03_ck_tile_gemm 03_ck_tile/my_ck_tile_gemm.cpp)
target_link_libraries(03_ck_tile_gemm PRIVATE hip::device)