"""
===============================================================================
Script: test_gemm.py
Project: Ryzen AI Heterogeneous Computing Lab (Phase 3: Advanced GPU Kernels)
Author: Heterogeneous AI Lab Team

Purpose:
  1. Demonstrate 2D GPU Grid & Block Coordinate Mapping for Matrix Multiplication.
  2. Benchmark General Matrix Multiply (GEMM: C = A x B) for [512 x 512] matrices:
     - Total FLOPs: 2 * M * N * K = 268,435,456 operations (268.4 MFLOPs)
     - Compare:
       * Paradigm 1: Python Triply-Nested Loop (Interpreted O(N^3))
       * Paradigm 2: Compiled Native C++ Kernel (WSL2 GCC with Tiling)
       * Paradigm 3: NumPy BLAS (OpenBLAS / MKL Vectorized)
       * Paradigm 4: PyTorch Multi-Threaded BLAS Engine
  3. Calculate GFLOP/s and Speedup Ratios.
===============================================================================
"""

import os
import sys
import time
import subprocess
import numpy as np
import torch

def print_banner(text: str):
    print("\n" + "=" * 75)
    print(f"   {text}")
    print("=" * 75)

def simulate_2d_gpu_grid(M=4, N=4, tile_size=2):
    """
    Simulates how a 2D GPU Grid and 2D Block map to Matrix coordinates.
    """
    print_banner("SIMULATION: 2D GPU Grid & Block Coordinate Mapping")
    print(f"Matrix Dimension: [{M} x {N}] (Total {M*N} elements)")
    print(f"Block Dimensions (blockDim.x, blockDim.y): ({tile_size}, {tile_size}) = {tile_size*tile_size} threads/block")
    
    grid_x = (N + tile_size - 1) // tile_size
    grid_y = (M + tile_size - 1) // tile_size
    print(f"Grid Dimensions  (gridDim.x, gridDim.y)  : ({grid_x}, {grid_y}) = {grid_x*grid_y} blocks\n")
    
    print(f"{'Block (bx, by)':^16} | {'Thread (tx, ty)':^18} | {'Matrix Element C[r, c]':^24} | {'Status':^12}")
    print("-" * 75)
    
    for by in range(grid_y):
        for bx in range(grid_x):
            for ty in range(tile_size):
                for tx in range(tile_size):
                    row = by * tile_size + ty
                    col = bx * tile_size + tx
                    
                    if row < M and col < N:
                        status = "Active"
                        elem = f"C[{row}, {col}]"
                    else:
                        status = "Masked"
                        elem = "Out of Bounds"
                        
                    print(f"({bx}, {by}):^16 | ({tx}, {ty}):^18 | {elem:^24} | {status:^12}")
    print("-" * 75)

def run_gemm_benchmarks():
    print_banner("PHASE 3 BENCHMARK: General Matrix Multiply (GEMM) [512 x 512]")
    
    M, K, N = 512, 512, 512
    total_flops = 2.0 * M * N * K  # 2 * 512 * 512 * 512 = 268,435,456 operations
    
    print(f"Matrix A : [{M} x {K}] (Float32, {M*K*4 / (1024*1024):.2f} MB)")
    print(f"Matrix B : [{K} x {N}] (Float32, {K*N*4 / (1024*1024):.2f} MB)")
    print(f"Matrix C : [{M} x {N}] (Float32, {M*N*4 / (1024*1024):.2f} MB)")
    print(f"Total Workload : {total_flops / 1e6:.1f} MFLOPs ({total_flops / 1e9:.3f} GFLOPs)\n")
    
    results = []

    # -------------------------------------------------------------------------
    # 1. Pure Python Triply-Nested Loop (Sampled 64x64)
    # -------------------------------------------------------------------------
    print("[1/4] Running Pure Python Nested Loop (Sampled 64x64)...")
    sub = 64
    py_a = [[1.0] * sub for _ in range(sub)]
    py_b = [[2.0] * sub for _ in range(sub)]
    py_c = [[0.0] * sub for _ in range(sub)]
    
    t0 = time.perf_counter()
    for i in range(sub):
        for j in range(sub):
            s = 0.0
            for k in range(sub):
                s += py_a[i][k] * py_b[k][j]
            py_c[i][j] = s
    t1 = time.perf_counter()
    
    # Scale from 64^3 to 512^3: scaling factor = (512 / 64)^3 = 8^3 = 512
    py_time_scaled_ms = (t1 - t0) * 512.0 * 1000.0
    py_gflops = (total_flops / (py_time_scaled_ms / 1000.0)) / 1e9
    results.append(("Python 3-Nested Loop", py_time_scaled_ms, py_gflops))
    print(f"      Estimated for [512x512]: {py_time_scaled_ms:.2f} ms ({py_gflops:.4f} GFLOP/s)")

    # -------------------------------------------------------------------------
    # 2. Compiled Native C++ GEMM Kernel (WSL2 GCC)
    # -------------------------------------------------------------------------
    print("\n[2/4] Running Compiled Native C++ Kernel via WSL2...")
    try:
        wsl_cmd = "wsl -e bash -c \"cd '/mnt/c/Users/rahul/Desktop/amd ai projects/ai project learning/src/kernels' && ./gemm_wsl\""
        output = subprocess.check_output(wsl_cmd, shell=True, text=True)
        cpp_time_ms = 0.0
        for line in output.splitlines():
            if "Elapsed Time:" in line:
                cpp_time_ms = float(line.split("Elapsed Time:")[1].replace("ms", "").strip())
        if cpp_time_ms > 0:
            cpp_gflops = (total_flops / (cpp_time_ms / 1000.0)) / 1e9
            results.append(("Native C++ Tiled Kernel", cpp_time_ms, cpp_gflops))
            print(f"      Time: {cpp_time_ms:.2f} ms ({cpp_gflops:.2f} GFLOP/s)")
    except Exception as e:
        print(f"      Could not run compiled executable: {e}")

    # -------------------------------------------------------------------------
    # 3. NumPy BLAS Matrix Multiply (OpenBLAS / MKL Vectorized)
    # -------------------------------------------------------------------------
    print("\n[3/4] Running NumPy BLAS (np.matmul)...")
    np_a = np.ones((M, K), dtype=np.float32)
    np_b = np.full((K, N), 2.0, dtype=np.float32)
    
    # Warmup
    _ = np.matmul(np_a, np_b)
    
    t0 = time.perf_counter()
    np_c = np.matmul(np_a, np_b)
    t1 = time.perf_counter()
    
    np_time_ms = (t1 - t0) * 1000.0
    np_gflops = (total_flops / (np_time_ms / 1000.0)) / 1e9
    results.append(("NumPy BLAS (Vectorized)", np_time_ms, np_gflops))
    print(f"      Time: {np_time_ms:.2f} ms ({np_gflops:.2f} GFLOP/s)")

    # -------------------------------------------------------------------------
    # 4. PyTorch Multi-Threaded BLAS Engine (torch.matmul)
    # -------------------------------------------------------------------------
    print("\n[4/4] Running PyTorch Optimized Engine (torch.matmul)...")
    torch_a = torch.ones((M, K), dtype=torch.float32)
    torch_b = torch.full((K, N), 2.0, dtype=torch.float32)
    
    # Warmup
    _ = torch.matmul(torch_a, torch_b)
    
    t0 = time.perf_counter()
    torch_c = torch.matmul(torch_a, torch_b)
    t1 = time.perf_counter()
    
    torch_time_ms = (t1 - t0) * 1000.0
    torch_gflops = (total_flops / (torch_time_ms / 1000.0)) / 1e9
    results.append(("PyTorch Multi-Threaded BLAS", torch_time_ms, torch_gflops))
    print(f"      Time: {torch_time_ms:.2f} ms ({torch_gflops:.2f} GFLOP/s)")

    # -------------------------------------------------------------------------
    # Performance Summary & Speedup Analysis
    # -------------------------------------------------------------------------
    print_banner("PERFORMANCE SUMMARY & SPEEDUP ANALYSIS")
    baseline_time = results[0][1]
    
    print(f"{'Paradigm':<32} | {'Time (ms)':>10} | {'Throughput':>14} | {'Speedup':>10}")
    print("-" * 75)
    for name, t_ms, gf in results:
        speedup = baseline_time / t_ms
        print(f"{name:<32} | {t_ms:>10.2f} | {gf:>11.2f} GFLOP/s| {speedup:>9.1f}x")
    print("-" * 75)
    
    print("\nKey Architectural Insights:")
    print("  1. GEMM is O(N^3) computation with O(N^2) memory footprint.")
    print("  2. Pure Python loops suffer catastrophically (hundreds of seconds).")
    print("  3. Shared Memory Tiling reduces Global Memory traffic by a factor of 16x,")
    print("     keeping compute units fed with high-speed L1 cache access.")

if __name__ == "__main__":
    simulate_2d_gpu_grid(M=4, N=4, tile_size=2)
    run_gemm_benchmarks()
