/**
 * ============================================================================
 * File: vector_add.hip
 * Project: Ryzen AI Heterogeneous Computing Lab (Phase 2: GPU Acceleration)
 * Author: Heterogeneous AI Lab Team
 * 
 * Purpose:
 *   A portable, single-source GPU kernel that performs element-wise vector addition:
 *       C[i] = A[i] + B[i]
 * 
 * Portability (AMD HIP & NVIDIA CUDA):
 *   HIP (Heterogeneous-Compute Interface for Portability) allows C++ code to run
 *   natively on AMD GPUs (via ROCm) and NVIDIA GPUs (via CUDA).
 *   When compiled with 'hipcc' targeting NVIDIA, it transpiles directly to CUDA.
 *   When compiled targeting AMD, it compiles directly to AMD GPU machine code.
 * ============================================================================
 */

#ifndef VECTOR_ADD_HIP
#define VECTOR_ADD_HIP

// Check if we are compiling with AMD HIP or standard CUDA
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__)
    #include <hip/hip_runtime.h>
    #define GPU_KERNEL __global__
    #define GPU_DEVICE __device__
    #define GPU_HOST   __host__
#elif defined(__CUDACC__)
    #include <cuda_runtime.h>
    #define GPU_KERNEL __global__
    #define GPU_DEVICE __device__
    #define GPU_HOST   __host__
#else
    // Fallback annotations for standard C++ compilers (GCC, Clang, MSVC)
    // allowing host compilation and static analysis without a GPU compiler installed.
    #define GPU_KERNEL
    #define GPU_DEVICE
    #define GPU_HOST
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/**
 * ----------------------------------------------------------------------------
 * GPU KERNEL: vectorAddKernel
 * ----------------------------------------------------------------------------
 * 
 * Execution Model:
 *   - In CPU programming, a 'for loop' iterates through 1,000,000 numbers one by one:
 *         for (int i = 0; i < N; i++) { C[i] = A[i] + B[i]; }
 * 
 *   - In GPU programming (SIMT = Single Instruction, Multiple Threads), we launch
 *     thousands of threads simultaneously. Each thread computes EXACTLY ONE element!
 * 
 * Thread Index Calculation:
 *   GPUs organize threads into a 2-level hierarchy:
 *     1. Grid: The entire execution space.
 *     2. Thread Block (or Workgroup): A cluster of threads scheduled together.
 * 
 *   Formula to compute the global thread ID:
 *       global_id = (block_index * block_dimension) + thread_index_inside_block
 * 
 *   In CUDA / HIP:
 *       int tid = blockDim.x * blockIdx.x + threadIdx.x;
 * 
 * Boundary Protection:
 *   If the total vector size N is not an exact multiple of the block size (e.g., 256),
 *   the last block will have extra threads. We MUST guard memory with:
 *       if (tid < N)
 * ----------------------------------------------------------------------------
 */
GPU_KERNEL void vectorAddKernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int N) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__) || defined(__CUDACC__)
    // 1. Calculate the unique global index for this thread
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // 2. Ensure thread does not write out-of-bounds (memory safety)
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
#else
    // CPU Fallback Simulation mode: runs sequentially if compiled with standard GCC
    for (int idx = 0; idx < N; idx++) {
        C[idx] = A[idx] + B[idx];
    }
#endif
}

/**
 * ----------------------------------------------------------------------------
 * HOST FUNCTION: launchVectorAdd
 * ----------------------------------------------------------------------------
 * Encapsulates the execution configuration:
 *   - Chooses optimal block size (typically 256 or 512 threads per block).
 *   - Calculates the required number of blocks: ceil(N / blockSize).
 * ----------------------------------------------------------------------------
 */
GPU_HOST void launchVectorAdd(const float* d_A,
                             const float* d_B,
                             float* d_C,
                             int N,
                             int threadsPerBlock = 256) {
    // Calculate total blocks required to cover N elements
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("[GPU Kernel Dispatch] Vector Size N = %d\n", N);
    printf("[GPU Kernel Dispatch] Threads Per Block = %d\n", threadsPerBlock);
    printf("[GPU Kernel Dispatch] Grid Blocks = %d (Total Threads = %d)\n",
           blocksPerGrid, blocksPerGrid * threadsPerBlock);

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__)
    // AMD HIP Kernel Launch Syntax
    hipLaunchKernelGGL(vectorAddKernel, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, d_A, d_B, d_C, N);
#elif defined(__CUDACC__)
    // NVIDIA CUDA Kernel Launch Syntax (Triple Chevrons)
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
#else
    // CPU Fallback execution
    vectorAddKernel(d_A, d_B, d_C, N);
#endif
}

#endif // VECTOR_ADD_HIP
