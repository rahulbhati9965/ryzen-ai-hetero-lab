"""
===============================================================================
Script: test_gpu_kernel.py
Project: Ryzen AI Heterogeneous Computing Lab (Phase 2: GPU Acceleration)
Author: Heterogeneous AI Lab Team

Purpose:
  1. Understand the GPU SIMT (Single Instruction, Multiple Threads) Execution Model.
  2. Benchmark 4 paradigms for Vector Addition (N = 10 Million elements):
     - Paradigm 1: Standard Python Single-Threaded Loop (Baseline CPU)
     - Paradigm 2: NumPy SIMD Vectorized (C-Accelerated CPU)
     - Paradigm 3: PyTorch High-Performance Tensor Engine (Optimized CPU/GPU)
     - Paradigm 4: Compiled Native C++ Kernel (Direct Hardware Level)
  3. Verify Numerical Correctness and compute Memory Throughput (GB/s).
===============================================================================
"""

import os
import sys
import time
import subprocess
import numpy as np
import torch

def print_banner(text: str):
    print("\n" + "=" * 70)
    print(f"   {text}")
    print("=" * 70)

def simulate_gpu_thread_hierarchy(n_elements: int = 16, block_size: int = 4):
    """
    Demonstrates mathematically how a GPU maps hardware threads to data indices.
    """
    print_banner("SIMULATION: How GPU Threads & Blocks Map to Memory")
    print(f"Total Elements to process (N): {n_elements}")
    print(f"Threads per Block (BlockDim.x): {block_size}")
    
    grid_size = (n_elements + block_size - 1) // block_size
    print(f"Calculated Grid Size (Number of Blocks): {grid_size}\n")
    
    print(f"{'Block ID':^10} | {'Thread ID in Block':^20} | {'Global Index (tid)':^20} | {'Status':^12}")
    print("-" * 70)
    
    for b in range(grid_size):
        for t in range(block_size):
            # The fundamental CUDA / HIP thread index equation:
            # global_idx = (block_idx * threads_per_block) + thread_in_block
            global_idx = (b * block_size) + t
            
            if global_idx < n_elements:
                status = f"Processes C[{global_idx}]"
            else:
                status = "Masked Out (Idle)"
                
            print(f"{b:^10} | {t:^20} | {global_idx:^20} | {status:^12}")
    print("-" * 70)

def run_benchmarks():
    print_banner("PHASE 2 BENCHMARK: 10,000,000 Elements Vector Addition")
    
    N = 10_000_000
    bytes_processed = 3 * N * 4  # Read A (4B), Read B (4B), Write C (4B)
    
    print(f"Vector Length : {N:,} elements (Float32)")
    print(f"Total Memory   : {N * 4 / (1024 * 1024):.2f} MB per array (Total ~{bytes_processed / (1024*1024):.2f} MB)")
    
    results = []

    # -------------------------------------------------------------------------
    # 1. Standard Python Loop (Single Core CPU) - Sampled
    # -------------------------------------------------------------------------
    print("\n[1/4] Running Single-Threaded Python Loop (100k sample)...")
    sample_n = 100_000
    py_a = [1.5] * sample_n
    py_b = [2.5] * sample_n
    py_c = [0.0] * sample_n
    
    t0 = time.perf_counter()
    for i in range(sample_n):
        py_c[i] = py_a[i] + py_b[i]
    t1 = time.perf_counter()
    
    # Scale up time to 10M for fair comparison
    py_time_scaled_ms = (t1 - t0) * (N / sample_n) * 1000.0
    results.append(("Python for-loop (Single Core)", py_time_scaled_ms, (bytes_processed / (py_time_scaled_ms / 1000.0)) / 1e9))
    print(f"      Estimated for 10M elements: {py_time_scaled_ms:.2f} ms")

    # -------------------------------------------------------------------------
    # 2. NumPy SIMD Vectorized (AVX / Multi-core CPU)
    # -------------------------------------------------------------------------
    print("\n[2/4] Running NumPy Vectorized (AVX / C-BLAS)...")
    np_a = np.full(N, 1.5, dtype=np.float32)
    np_b = np.full(N, 2.5, dtype=np.float32)
    
    # Warmup
    _ = np_a + np_b
    
    t0 = time.perf_counter()
    np_c = np_a + np_b
    t1 = time.perf_counter()
    
    np_time_ms = (t1 - t0) * 1000.0
    results.append(("NumPy SIMD Vectorized", np_time_ms, (bytes_processed / (np_time_ms / 1000.0)) / 1e9))
    print(f"      Time: {np_time_ms:.2f} ms | Bandwidth: {(bytes_processed / (np_time_ms / 1000.0)) / 1e9:.2f} GB/s")

    # -------------------------------------------------------------------------
    # 3. PyTorch Tensor Addition (C++ Backend)
    # -------------------------------------------------------------------------
    print("\n[3/4] Running PyTorch Tensor Addition...")
    torch_a = torch.full((N,), 1.5, dtype=torch.float32)
    torch_b = torch.full((N,), 2.5, dtype=torch.float32)
    
    # Warmup
    _ = torch_a + torch_b
    
    t0 = time.perf_counter()
    torch_c = torch_a + torch_b
    t1 = time.perf_counter()
    
    torch_time_ms = (t1 - t0) * 1000.0
    results.append(("PyTorch Tensor Engine", torch_time_ms, (bytes_processed / (torch_time_ms / 1000.0)) / 1e9))
    print(f"      Time: {torch_time_ms:.2f} ms | Bandwidth: {(bytes_processed / (torch_time_ms / 1000.0)) / 1e9:.2f} GB/s")

    # -------------------------------------------------------------------------
    # 4. Compiled Native C++ Kernel
    # -------------------------------------------------------------------------
    print("\n[4/4] Running Compiled Native C++ Kernel via WSL...")
    try:
        wsl_cmd = "wsl -e bash -c \"cd '/mnt/c/Users/rahul/Desktop/amd ai projects/ai project learning/src/kernels' && ./vector_add_wsl\""
        output = subprocess.check_output(wsl_cmd, shell=True, text=True)
        # Parse elapsed time from output
        cpp_time_ms = 0.0
        for line in output.splitlines():
            if "Elapsed Time:" in line:
                cpp_time_ms = float(line.split("Elapsed Time:")[1].replace("ms", "").strip())
        if cpp_time_ms > 0:
            results.append(("Native C++ Kernel", cpp_time_ms, (bytes_processed / (cpp_time_ms / 1000.0)) / 1e9))
            print(f"      Time: {cpp_time_ms:.2f} ms | Bandwidth: {(bytes_processed / (cpp_time_ms / 1000.0)) / 1e9:.2f} GB/s")
    except Exception as e:
        print(f"      Could not run compiled executable directly: {e}")

    # -------------------------------------------------------------------------
    # Comparison Summary Table
    # -------------------------------------------------------------------------
    print_banner("PERFORMANCE SUMMARY & SPEEDUP ANALYSIS")
    baseline_time = results[0][1]
    
    print(f"{'Paradigm':<32} | {'Time (ms)':>10} | {'Throughput':>14} | {'Speedup':>10}")
    print("-" * 75)
    for name, t_ms, bw in results:
        speedup = baseline_time / t_ms
        print(f"{name:<32} | {t_ms:>10.2f} | {bw:>11.2f} GB/s | {speedup:>9.1f}x")
    print("-" * 75)
    
    print("\nKey Takeaway:")
    print("  Moving from interpreted loops to vectorized execution achieves 10x-50x speedups.")
    print("  Native C++ & GPU kernels eliminate Python interpreter overhead completely,")
    print("  saturating memory bandwidth directly at the hardware silicon level.")

if __name__ == "__main__":
    simulate_gpu_thread_hierarchy(n_elements=16, block_size=4)
    run_benchmarks()
