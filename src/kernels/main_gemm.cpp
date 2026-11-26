/**
 * ============================================================================
 * File: main_gemm.cpp
 * Project: Ryzen AI Heterogeneous Computing Lab (Phase 3: Advanced GPU Kernels)
 * Author: Heterogeneous AI Lab Team
 * 
 * Purpose:
 *   Demonstrates C++ GEMM Host Runner:
 *     - Matrix Dimension: M=512, K=512, N=512 (512x512 = 262,144 elements)
 *     - Total Floating Point Operations: 2 * M * N * K = 268.4 Million FLOPs
 *     - Measures execution time and computes Effective GFLOP/s
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include "matrix_multiply.hip"

int main() {
    printf("====================================================================\n");
    printf("   Ryzen AI Heterogeneous Lab - Phase 3: Advanced GPU Kernels\n");
    printf("   General Matrix Multiply (GEMM) C++ / HIP Demonstration\n");
    printf("====================================================================\n\n");

    const int M = 512;
    const int K = 512;
    const int N = 512;

    const size_t bytes_A = M * K * sizeof(float);
    const size_t bytes_B = K * N * sizeof(float);
    const size_t bytes_C = M * N * sizeof(float);

    printf("[1] Allocating Host memory for matrices:\n");
    printf("    A: [%d x %d] (%.2f MB)\n", M, K, (double)bytes_A / (1024*1024));
    printf("    B: [%d x %d] (%.2f MB)\n", K, N, (double)bytes_B / (1024*1024));
    printf("    C: [%d x %d] (%.2f MB)\n", M, N, (double)bytes_C / (1024*1024));

    float* h_A = (float*)malloc(bytes_A);
    float* h_B = (float*)malloc(bytes_B);
    float* h_C = (float*)malloc(bytes_C);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Error: Memory allocation failed!\n");
        return -1;
    }

    // Initialize with test data
    // Matrix A: all 1.0f, Matrix B: all 2.0f
    // Expected result C[i, j] = 512 * (1.0 * 2.0) = 1024.0f
    for (int i = 0; i < M * K; i++) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = 2.0f;
    for (int i = 0; i < M * N; i++) h_C[i] = 0.0f;

    printf("\n[2] Executing Tiled Matrix Multiplication (Shared Memory / LDS)...\n");
    auto start = std::chrono::high_resolution_clock::now();

    launchMatrixMul(h_A, h_B, h_C, M, K, N, true);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // Verify numerical accuracy
    printf("\n[3] Verifying numerical results...\n");
    float expected = (float)K * 2.0f; // 512 * 2.0 = 1024.0
    bool passed = true;
    for (int i = 0; i < M * N; i++) {
        if (std::abs(h_C[i] - expected) > 1e-3) {
            printf("Mismatch at index %d: Got %f, expected %f\n", i, h_C[i], expected);
            passed = false;
            break;
        }
    }

    double flops = 2.0 * (double)M * (double)N * (double)K;
    double gflops = (flops / (duration.count() / 1000.0)) / 1e9;

    if (passed) {
        printf(">> [SUCCESS] All %d elements matched expected value (%.1f) with 100%% accuracy!\n",
               M * N, expected);
        printf(">> Elapsed Time: %.2f ms\n", duration.count());
        printf(">> Compute Performance: %.2f GFLOP/s\n", gflops);
    } else {
        printf(">> [FAILED] Calculation mismatch detected.\n");
    }

    free(h_A);
    free(h_B);
    free(h_C);

    printf("\nExecution completed successfully.\n");
    return 0;
}
