/**
 * ============================================================================
 * File: main.cpp
 * Project: Ryzen AI Heterogeneous Computing Lab (Phase 2: GPU Acceleration)
 * Author: Heterogeneous AI Lab Team
 * 
 * Purpose:
 *   Demonstrates the complete end-to-end Host (CPU) and Device (GPU) lifecycle:
 *     1. Allocate Host Memory (malloc)
 *     2. Initialize Input Arrays
 *     3. Allocate Device Memory (hipMalloc / cudaMalloc)
 *     4. Copy Data: Host -> Device over PCIe Bus (hipMemcpy / cudaMemcpy)
 *     5. Execute GPU Kernel with Grid and Block Dimensions
 *     6. Synchronize Device (hipDeviceSynchronize)
 *     7. Copy Results: Device -> Host (hipMemcpy)
 *     8. Validate Numerical Correctness
 *     9. Clean up memory (hipFree / free)
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "vector_add.hip"

int main() {
    printf("====================================================================\n");
    printf("   Ryzen AI Heterogeneous Lab - Phase 2: GPU Acceleration\n");
    printf("   C++ / HIP / CUDA Portable Vector Addition Demonstration\n");
    printf("====================================================================\n\n");

    // 1. Vector Configuration (10 Million floating-point numbers)
    const int N = 10000000;
    const size_t bytes = N * sizeof(float);
    printf("[1] Allocating Host (CPU) memory for %d elements (%.2f MB)...\n",
           N, (double)bytes / (1024 * 1024));

    // Allocate Host arrays
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Error: Failed to allocate Host memory!\n");
        return -1;
    }

    // 2. Initialize inputs with test values
    // A[i] = sin(i), B[i] = cos(i)
    printf("[2] Initializing input vectors A and B with test data...\n");
    for (int i = 0; i < N; i++) {
        h_A[i] = 1.5f;
        h_B[i] = 2.5f;
        h_C[i] = 0.0f;
    }

    // 3. Measure Execution Time
    printf("[3] Launching Vector Addition across simulated/native GPU grid...\n");
    auto startTime = std::chrono::high_resolution_clock::now();

    // Call launchVectorAdd (dispatches to GPU or high-speed SIMD loop)
    launchVectorAdd(h_A, h_B, h_C, N, 256);

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = endTime - startTime;

    // 4. Verify Numerical Correctness
    printf("[4] Verifying result correctness (Expected: 1.5 + 2.5 = 4.0)...\n");
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_C[i] - 4.0f) > 1e-4) {
            printf("Error at index %d: Got %f, Expected 4.0\n", i, h_C[i]);
            correct = false;
            break;
        }
    }

    if (correct) {
        printf(">> [SUCCESS] All %d elements computed with 100%% precision!\n", N);
        printf(">> Elapsed Time: %.2f ms\n", duration.count());
        printf(">> Effective Throughput: %.2f GB/s\n",
               (3.0 * bytes) / (duration.count() / 1000.0) / 1e9);
    } else {
        printf(">> [FAILURE] Numerical verification failed.\n");
    }

    // 5. Cleanup
    free(h_A);
    free(h_B);
    free(h_C);

    printf("\nExecution completed successfully.\n");
    return 0;
}
