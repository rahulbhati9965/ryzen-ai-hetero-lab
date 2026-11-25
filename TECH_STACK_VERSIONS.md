# 🛠️ Tech Stack & Version Registry
> **Project**: [ryzen-ai-hetero-lab](https://github.com/rahulbhati9965/ryzen-ai-hetero-lab)  
> **Last Updated**: November 24, 2026  
> **Status**: Living Document (Updated with every phase)

This file tracks the exact versions of all hardware, compilers, drivers, and software libraries used in this project to guarantee 100% reproducibility across environments.

---

## 1. Physical Hardware & System Environment

| Component | Detected Specification | Role in AI Lab |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 Home (Build 10.0.26300) | Primary host OS & developer workstation |
| **CPU** | AMD Ryzen 5 6600H (6 Cores / 12 Threads, Zen 3+) | Orchestration, data preprocessing, AVX2 fallback |
| **Dedicated GPU (dGPU)**| NVIDIA GeForce RTX 3050 6GB Laptop GPU (Ampere) | Heavy model training, Tensor Cores, native CUDA |
| **Integrated GPU (iGPU)**| AMD Radeon Graphics 660M (RDNA 2, 512MB Shared) | Low-power DirectML edge inference |
| **GPU Driver (NVIDIA)**| NVIDIA KMD Version 617.42 (CUDA UMD 13.4) | GPU compute kernel execution |
| **Linux Subsystem** | WSL2 (Ubuntu 24.04 LTS) | Direct GPU passthrough for AMD ROCm / HIP compilation |

---

## 2. Core AI & Deployment Frameworks

| Technology / Library | Exact Installed Version | Purpose / What It Does | Introduced In |
| :--- | :--- | :--- | :---: |
| **Python** | `3.11.1` | Base programming language (Official AMD & PyTorch standard) | Phase 0 |
| **PyTorch** | `2.15.0+cpu` | Neural network definition, tensor math, and graph export | Phase 0 |
| **TorchVision** | `0.29.2+cpu` | Pretrained vision models (MobileNetV2, ResNet) & datasets | Phase 0 |
| **ONNX** | `1.22.0` | Universal static neural network computation graph format | Phase 0 |
| **ONNX Runtime** | `1.32.0` | Cross-hardware inference engine (`ort.InferenceSession`) | Phase 0 |
| **AMD Quark** | `0.13+cpu.torch2.13` | Official AMD toolkit for INT8 / XINT8 Post-Training Quantization | Phase 1 |
| **NumPy** | `2.4.6` | Multi-dimensional numerical array processing for tensors | Phase 0 |

---

## 3. Developer Tooling & Reporting

| Tool / Package | Exact Installed Version | Purpose | Introduced In |
| :--- | :--- | :--- | :---: |
| **pip** | `26.3` | Python package installer and wheel resolver | Phase 0 |
| **ReportLab** | `5.1.0` | Programmatic PDF generation for phase milestone guides | Phase 0 |
| **Rich** | `15.0.0` | Formatted terminal output, tables, and colored progress logs | Phase 0 |
| **Pydantic** | `2.14.4` | Data validation and configuration parsing (used by Quark) | Phase 1 |
| **FastAPI** | `0.110.0+` | High-performance asynchronous REST API for model serving | Phase 1.5 |
| **Uvicorn** | `0.28.0+` | ASGI web server for containerized inference endpoints | Phase 1.5 |
| **Docker Desktop**| `4.99.0 (Engine 29.9.3)` | Local containerization, isolation, and packaging | Phase 1.5 |
| **Kubernetes (k8s)**| `1.30+ (via Docker Desktop)` | Container orchestration, auto-scaling, and health probes | Phase 1.5 |
| **GitHub Actions**| `CI/CD v4/v5` | Automated workflow running testing, linting & Docker builds | Phase 1.5 |
| **Git** | `2.x` | Source code version control | Phase 1 |

---

## 4. Phase-by-Phase Version Changelog

### Phase 0: Foundations & Architecture Setup
* Initialized isolated virtual environment (`.venv`) on **Python 3.11.1**.
* Confirmed hardware specifications (RTX 3050 6GB + Ryzen 6600H + Radeon 660M + WSL2).
* Configured `requirements.txt` and `configs/vaip_config.json`.
* Produced **`00_Initial_Setup_and_Architecture_Guide.pdf`** using ReportLab 5.1.0.

### Phase 1: AMD Ryzen AI Deployment Pipeline
* Installed **AMD Quark 0.13** (pre-built CPU wheel with native Windows C++ extensions).
* Integrated **PyTorch 2.15.0** and **ONNX 1.22.0**.
* Exported **MobileNetV2** (3.5M parameters) to ONNX with dynamic batching.
* Quantized graph to **INT8 QDQ** via `quark.onnx.quantize_static`.
* Deployed inference via **ONNX Runtime 1.32.0** with hardware provider fallback.
* Produced **`01_Phase1_Model_Export_Quantization_Inference_Guide.pdf`**.
* Pushed repository to GitHub: [`ryzen-ai-hetero-lab`](https://github.com/rahulbhati9965/ryzen-ai-hetero-lab).

### Phase 2: The GPU Acceleration Lab (CUDA & AMD HIP)
* Developed single-source portable C++ GPU kernel (`src/kernels/vector_add.hip`) compatible with both **AMD HIP** and **NVIDIA CUDA**.
* Built end-to-end Host/Device memory lifecycle demonstration in `src/kernels/main.cpp`.
* Compiled and executed native kernel using **GCC 13.3.0** via WSL2 with NVIDIA driver CUDA UMD 13.4.
* Built benchmark and thread-hierarchy simulator harness in `src/test_gpu_kernel.py` benchmarking 10,000,000 float32 elements:
  - Python Single-Core: ~977.74 ms (1.0x baseline, 0.12 GB/s)
  - Compiled Native C++: 104.14 ms (9.4x speedup, 1.15 GB/s)
  - NumPy SIMD (AVX): 15.19 ms (64.4x speedup, 7.90 GB/s)
  - PyTorch Multi-threaded Engine: 8.06 ms (121.3x speedup, 14.89 GB/s)
* Produced **`02_Phase2_CUDA_and_HIP_GPU_Programming_Guide.pdf`** using ReportLab.

