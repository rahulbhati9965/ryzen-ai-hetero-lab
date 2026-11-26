/**
 * ============================================================================
 * File: matrix_multiply.hip
 * Project: Ryzen AI Heterogeneous Computing Lab (Phase 3: Advanced GPU Kernels)
 * Author: Heterogeneous AI Lab Team
 * 
 * Purpose:
 *   Implements General Matrix Multiplication (GEMM): C = A x B
 *   Matrix A: [M x K], Matrix B: [K x N], Matrix C: [M x N]
 * 
 * Compares two distinct GPU architectures:
 *   1. Naive 2D Grid/Block Kernel:
 *      Every thread computes one element of C. Requires reading from slow Global VRAM
 *      for every single multiply-accumulate step (severe memory bandwidth bottleneck).
 * 
 *   2. Tiled GEMM Kernel using Shared Memory (LDS - Local Data Share on AMD):
 *      Cooperatively loads square tiles into ultra-fast on-chip SRAM cache.
 *      Reduces global VRAM traffic by a factor of TILE_WIDTH!
 * ============================================================================
 */

#ifndef MATRIX_MULTIPLY_HIP
#define MATRIX_MULTIPLY_HIP

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__)
    #include <hip/hip_runtime.h>
    #define GPU_KERNEL   __global__
    #define GPU_DEVICE   __device__
    #define GPU_HOST     __host__
    #define GPU_SHARED   __shared__
    #define GPU_BARRIER() __syncthreads()
#elif defined(__CUDACC__)
    #include <cuda_runtime.h>
    #define GPU_KERNEL   __global__
    #define GPU_DEVICE   __device__
    #define GPU_HOST     __host__
    #define GPU_SHARED   __shared__
    #define GPU_BARRIER() __syncthreads()
#else
    // Fallback annotations for standard CPU compilers (GCC, Clang)
    #define GPU_KERNEL
    #define GPU_DEVICE
    #define GPU_HOST
    #define GPU_SHARED
    #define GPU_BARRIER()
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE_WIDTH 16

/**
 * ----------------------------------------------------------------------------
 * 1. NAIVE 2D MATRIX MULTIPLICATION KERNEL
 * ----------------------------------------------------------------------------
 * 
 * Memory Layout (Row-Major):
 *   Element (r, c) is stored at: matrix[r * width + c]
 * 
 * In this naive approach:
 *   - Thread (x, y) computes element C[row, col].
 *   - To compute C[row, col], the thread loops k from 0 to K-1:
 *         sum += A[row * K + k] * B[k * N + col]
 *   - Problem: Every element in A and B is fetched from slow Global Memory K times!
 * ----------------------------------------------------------------------------
 */
GPU_KERNEL void naiveMatrixMulKernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int M, int K, int N) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__) || defined(__CUDACC__)
    // 2D Thread Indexing
    int col = blockDim.x * blockIdx.x + threadIdx.x; // Column index in C
    int row = blockDim.y * blockIdx.y + threadIdx.y; // Row index in C

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
#else
    // CPU Fallback Implementation
    for (int r = 0; r < M; r++) {
        for (int c = 0; c < N; c++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[r * K + k] * B[k * N + c];
            }
            C[r * N + c] = sum;
        }
    }
#endif
}

/**
 * ----------------------------------------------------------------------------
 * 2. OPTIMIZED TILED MATRIX MULTIPLICATION (SHARED MEMORY / LDS)
 * ----------------------------------------------------------------------------
 * 
 * Why Tiling Works:
 *   Instead of reading directly from slow Global VRAM (~192 GB/s on RTX 3050),
 *   we copy a sub-matrix ("tile") of size [TILE_WIDTH x TILE_WIDTH] into
 *   ultra-fast On-Chip Shared Memory (~8,000+ GB/s).
 * 
 * Algorithm Steps:
 *   1. Loop over phases: phase = 0 to (K / TILE_WIDTH - 1).
 *   2. All threads in the block cooperatively load 1 float from A and 1 from B
 *      into shared memory arrays: s_A[ty][tx] and s_B[ty][tx].
 *   3. Synchronize all threads (__syncthreads()) so the tile is fully loaded.
 *   4. Multiply the elements cached in shared memory.
 *   5. Synchronize again (__syncthreads()) before loading the next phase's tile.
 *   6. Write final accumulated sum to Global Memory C.
 * ----------------------------------------------------------------------------
 */
GPU_KERNEL void tiledMatrixMulKernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int M, int K, int N) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__) || defined(__CUDACC__)
    // Shared Memory scratchpads (On-chip SRAM cache per Thread Block)
    GPU_SHARED float s_A[TILE_WIDTH][TILE_WIDTH];
    GPU_SHARED float s_B[TILE_WIDTH][TILE_WIDTH];

    // Thread coordinates inside the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global matrix coordinates
    int col = blockDim.x * blockIdx.x + tx;
    int row = blockDim.y * blockIdx.y + ty;

    float sum = 0.0f;

    // Number of tiles needed across dimension K
    int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int phase = 0; phase < numTiles; phase++) {
        // Cooperative Load Tile from Matrix A into Shared Memory
        int a_col = phase * TILE_WIDTH + tx;
        if (row < M && a_col < K) {
            s_A[ty][tx] = A[row * K + a_col];
        } else {
            s_A[ty][tx] = 0.0f; // Zero-pad out of bounds
        }

        // Cooperative Load Tile from Matrix B into Shared Memory
        int b_row = phase * TILE_WIDTH + ty;
        if (b_row < K && col < N) {
            s_B[ty][tx] = B[b_row * N + col];
        } else {
            s_B[ty][tx] = 0.0f; // Zero-pad out of bounds
        }

        // Barrier: Wait for ALL threads in this block to finish loading their tiles
        GPU_BARRIER();

        // Compute dot product of the cached tiles from ultra-fast SRAM
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += s_A[ty][k] * s_B[k][tx];
        }

        // Barrier: Wait for all threads to finish computing before next phase overwrites s_A & s_B
        GPU_BARRIER();
    }

    // Write final accumulated result to Global Memory
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
#else
    // CPU Fallback
    naiveMatrixMulKernel(A, B, C, M, K, N);
#endif
}

/**
 * ----------------------------------------------------------------------------
 * HOST FUNCTION: launchMatrixMul
 * ----------------------------------------------------------------------------
 * Sets up 2D Grid and 2D Block dimensions:
 *   threadsPerBlock = (TILE_WIDTH, TILE_WIDTH) = (16, 16) = 256 threads
 *   blocksPerGrid   = (ceil(N / 16), ceil(M / 16))
 * ----------------------------------------------------------------------------
 */
GPU_HOST void launchMatrixMul(const float* A, const float* B, float* C,
                             int M, int K, int N, bool useTiled = true) {
    int blocks_x = (N + TILE_WIDTH - 1) / TILE_WIDTH;
    int blocks_y = (M + TILE_WIDTH - 1) / TILE_WIDTH;

    printf("[GEMM Launch] Dimensions: [%d x %d] * [%d x %d] -> [%d x %d]\n",
           M, K, K, N, M, N);
    printf("[GEMM Launch] Mode: %s\n", useTiled ? "Tiled (Shared Memory / LDS)" : "Naive 2D");
    printf("[GEMM Launch] Block Size: (%d, %d) = %d threads\n",
           TILE_WIDTH, TILE_WIDTH, TILE_WIDTH * TILE_WIDTH);
    printf("[GEMM Launch] Grid Size: (%d, %d) = %d total blocks\n",
           blocks_x, blocks_y, blocks_x * blocks_y);

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__)
    dim3 grid(blocks_x, blocks_y);
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    if (useTiled) {
        hipLaunchKernelGGL(tiledMatrixMulKernel, grid, block, 0, 0, A, B, C, M, K, N);
    } else {
        hipLaunchKernelGGL(naiveMatrixMulKernel, grid, block, 0, 0, A, B, C, M, K, N);
    }
#elif defined(__CUDACC__)
    dim3 grid(blocks_x, blocks_y);
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    if (useTiled) {
        tiledMatrixMulKernel<<<grid, block>>>(A, B, C, M, K, N);
    } else {
        naiveMatrixMulKernel<<<grid, block>>>(A, B, C, M, K, N);
    }
#else
    tiledMatrixMulKernel(A, B, C, M, K, N);
#endif
}

#endif // MATRIX_MULTIPLY_HIP
