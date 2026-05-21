if (NOT EXISTS $ENV{ROCM_PATH})
    if (NOT EXISTS /opt/rocm)
        set(ROCM_PATH /usr)
    else()
        set(ROCM_PATH /opt/rocm)
    endif()
else()
    set(ROCM_PATH $ENV{ROCM_PATH})
endif()

list(APPEND CMAKE_PREFIX_PATH  ${ROCM_PATH})
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}/lib64/cmake")

if (NOT DEFINED CMAKE_HIP_FLAGS_DEBUG)
    set(CMAKE_HIP_FLAGS_DEBUG "-g -O2")
endif()

# CMake on Windows doesn't support the HIP language yet
if (WIN32)
    set(CXX_IS_HIPCC TRUE)
else()
    string(REGEX MATCH "hipcc(\.bat)?$" CXX_IS_HIPCC "${CMAKE_CXX_COMPILER}")
endif()

if (CXX_IS_HIPCC)
    if (LINUX)
        if (NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
            message(WARNING "Only LLVM is supported for HIP, hint: CXX=/opt/rocm/llvm/bin/clang++")
        endif()

        # Suppress legacy hipcc warning — we already set CMAKE_HIP_COMPILER explicitly above
        # message(WARNING "Setting hipcc as the C++ compiler is legacy behavior."
        #         " Prefer setting the HIP compiler directly. See README for details.")
    endif()
else()
    # Forward (AMD)GPU_TARGETS to CMAKE_HIP_ARCHITECTURES.
    if(AMDGPU_TARGETS AND NOT GPU_TARGETS)
        set(GPU_TARGETS ${AMDGPU_TARGETS})
    endif()
    if(GPU_TARGETS AND NOT CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES ${GPU_TARGETS})
    endif()
    cmake_minimum_required(VERSION 3.21)
    enable_language(HIP)
endif()

find_package(hip     REQUIRED)
find_package(hipblas REQUIRED)
find_package(rocblas REQUIRED)

if (GGML_HIP_RCCL)
    find_package(rccl REQUIRED)
endif()

if (${hip_VERSION} VERSION_LESS 6.1)
    message(FATAL_ERROR "At least ROCM/HIP V6.1 is required")
endif()

message(STATUS "HIP and hipBLAS found")

file(GLOB   GGML_HEADERS_ROCM "../ggml-cuda/*.cuh")
list(APPEND GGML_HEADERS_ROCM "../../include/ggml-cuda.h")

file(GLOB   GGML_SOURCES_ROCM "../ggml-cuda/*.cu")
# iq4_dequant_rdn2.cu is #include'd in convert.cu (compile-time gated), exclude from separate compilation
list(FILTER GGML_SOURCES_ROCM EXCLUDE REGEX "iq4_dequant_rdn2\\.cu$")
file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-tile*.cu")
# Exclude D>=576 tile kernels: exceed HIP local memory limit (67584 > 65536)
list(FILTER SRCS EXCLUDE REGEX "dkq(576|640)")
list(APPEND GGML_SOURCES_ROCM ${SRCS})
file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-mma*.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})
file(GLOB   SRCS "../ggml-cuda/template-instances/mmq*.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})
file(GLOB   SRCS "../ggml-cuda/template-instances/mmf*.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})

if (GGML_CUDA_FA_ALL_QUANTS)
    file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-vec*.cu")
    list(APPEND GGML_SOURCES_ROCM ${SRCS})
    add_compile_definitions(GGML_CUDA_FA_ALL_QUANTS)
else()
    list(APPEND GGML_SOURCES_ROCM
        ../ggml-cuda/template-instances/fattn-vec-instance-f16-f16.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-q4_0-q4_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-q8_0-q8_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-bf16-bf16.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-turbo3_0-turbo3_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-turbo3_0-q8_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-q8_0-turbo3_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-turbo2_0-turbo2_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-turbo2_0-q8_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-q8_0-turbo2_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-turbo3_0-turbo2_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-turbo2_0-turbo3_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-turbo4_0-turbo4_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-turbo4_0-q8_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-q8_0-turbo4_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-turbo4_0-turbo3_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-turbo3_0-turbo4_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-turbo4_0-turbo2_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-turbo2_0-turbo4_0.cu)
endif()

ggml_add_backend_library(ggml-hip
                         ${GGML_HEADERS_ROCM}
                         ${GGML_SOURCES_ROCM}
                        )

# TODO: do not use CUDA definitions for HIP
if (NOT GGML_BACKEND_DL)
    target_compile_definitions(ggml PUBLIC GGML_USE_CUDA)
endif()

add_compile_definitions(GGML_USE_HIP)

if (GGML_CUDA_FORCE_MMQ)
    add_compile_definitions(GGML_CUDA_FORCE_MMQ)
endif()

if (GGML_CUDA_FORCE_CUBLAS)
    add_compile_definitions(GGML_CUDA_FORCE_CUBLAS)
endif()

if (GGML_CUDA_NO_PEER_COPY)
    add_compile_definitions(GGML_CUDA_NO_PEER_COPY)
endif()

if (GGML_HIP_GRAPHS)
    add_compile_definitions(GGML_HIP_GRAPHS)
endif()

if (GGML_HIP_NO_VMM)
    add_compile_definitions(GGML_HIP_NO_VMM)
endif()

if (GGML_HIP_ROCWMMA_FATTN)
    add_compile_definitions(GGML_HIP_ROCWMMA_FATTN)
    if (GPU_TARGETS MATCHES "gfx103")
        message(WARNING "GGML_HIP_ROCWMMA_FATTN=ON has no effect on gfx1030 (RDNA2) — no hardware matrix cores")
    endif()
endif()

if (NOT GGML_HIP_MMQ_MFMA)
    add_compile_definitions(GGML_HIP_NO_MMQ_MFMA)
endif()

if (GGML_HIP_RCCL)
    add_compile_definitions(GGML_USE_NCCL) # RCCL has the same interface as NCCL.
endif()

if (GGML_HIP_EXPORT_METRICS)
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Rpass-analysis=kernel-resource-usage --save-temps")
endif()

if (NOT GGML_CUDA_FA)
    add_compile_definitions(GGML_CUDA_NO_FA)
endif()

# RDNA2 optimization flags for the ggml-hip backend
# Enables BFE dequant, async pipeline (compile-time)
# RDNA2_ASYNC_PIPELINE removed — had zero kernel implementation (phantom gate).
# RDNA2_OPT_V1 retained as compile definition for iq4_dequant_rdn2.cu (dequant kernel
# used by test_dequant_rdn2 and dequant_microbench). NOT a production optimization gate
# — the standalone dequant path is cold during inference. Inline fusing in matmul is used.
add_compile_definitions(RDNA2_OPT_V1)
option(GGML_RDNA2_BFE_DISPATCHER "ggml-hip: use BFE v_bfe_u32 for Q4_K dequant" OFF)
if (GGML_RDNA2_BFE_DISPATCHER)
    add_compile_definitions(RDNA2_BFE_DISPATCHER)
endif()

# REMOVED: RDNA2_MOE_STREAM_V1 (archived per CR-011 — code removed from common.cuh)

if (RDNA2_VGPR_OPT_V1)
    add_compile_definitions(RDNA2_VGPR_OPT_V1)
endif()

# RDNA2_EXPERT_SORT: Expert-aware warp scheduling for MoE MMVQ kernel.
# Sorts channels by expert ID before kernel launch so consecutive warps access the
# same expert's weights — improving Infinity Cache hit rates on RDNA2.
option(RDNA2_EXPERT_SORT "rdna2: expert-aware warp scheduling for MoE MMVQ" OFF)
if (RDNA2_EXPERT_SORT)
    add_compile_definitions(RDNA2_EXPERT_SORT)
endif()

# RDNA2_CACHE_SWIZZLE: Cache-aware IQ4_XS tensor swizzling (CR-013).
# Converts AoS → SoA layout so qs[128] aligns to 128-byte RDNA2 Infinity Cache lines.
# When enabled, IQ4_XS tensors are swizzled at load time and a specialized vec_dot
# kernel uses the SoA layout — eliminating 5.9% cache line straddle waste.
# OFF by default — experimental, needs rocprofv3 L2/L3 cache miss validation.
option(RDNA2_CACHE_SWIZZLE "rdna2: cache-aware IQ4_XS SoA swizzling (CR-013)" OFF)
if (RDNA2_CACHE_SWIZZLE)
    add_compile_definitions(RDNA2_CACHE_SWIZZLE)
endif()

# Idea D: gfx1030 LLVM compiler tuning flags (v0.4.0)
# Applied unconditionally — these hint the scheduler and register allocator
# for the wave32+dp4a decode pattern on gfx1030.
# See opencode/agents/DEEP_ISA_MISSION.md for the full ISA roadmap.
# All three flags are safe, upstream LLVM flags for gfx1030+.
# (Re-enabled 2026-05-16: falsely accused of all-newline bug. Real cause was `-n` flag.)
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -mllvm -amdgpu-early-inline-all=true")
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -mllvm -amdgpu-spill-sgpr-to-vgpr")
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -mllvm -amdgpu-function-calls=false")
# set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -mllvm -amdgpu-enable-max-ilp-scheduling-strategy=1") # Not supported in this ROCm
# set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -ffast-math") # Causes build breaks in HIP edge cases
# set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --offload-compression") # Not supported in this ROCm
# set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -mllvm -amdgpu-enable-rewrite-out-of-range-value=1") # Not supported in ROCm 7.13

if (CXX_IS_HIPCC)
    set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES LANGUAGE CXX)
    if (WIN32 AND CMAKE_BUILD_TYPE STREQUAL "Debug")
        # CMake on Windows doesn't support the HIP language yet.
        # Therefore we workaround debug build's failure on HIP backend this way.
        set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES COMPILE_FLAGS "-O2 -g")
    endif()
    target_link_libraries(ggml-hip PRIVATE hip::device)
else()
    set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES LANGUAGE HIP)
endif()

if (GGML_STATIC)
    message(FATAL_ERROR "Static linking not supported for HIP/ROCm")
endif()

if (GGML_HIP_RCCL)
    target_link_libraries(ggml-hip PRIVATE ggml-base roc::rccl)
endif()

target_link_libraries(ggml-hip PRIVATE ggml-base hip::host roc::rocblas roc::hipblas)
