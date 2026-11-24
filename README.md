# AI & Heterogeneous GPU Engineering Lab 🚀

A comprehensive, hands-on learning repository exploring **Deep Learning Model Deployment (AMD Ryzen AI & ONNX)** and **Unified GPU Kernel Optimization (NVIDIA CUDA + AMD ROCm / HIP)**.

---

## 💻 Hardware Landscape

This lab is developed on a heterogeneous computing platform:
* **Dedicated GPU (dGPU)**: NVIDIA GeForce RTX 3050 6GB Laptop GPU (Ampere, CUDA Compute Capability 8.6, Tensor Cores)
* **CPU**: AMD Ryzen 5 6600H (6 Cores / 12 Threads, Zen 3+ AVX2)
* **Integrated GPU (iGPU)**: AMD Radeon Graphics 660M (RDNA 2, DirectML acceleration)
* **Linux Subsystem**: WSL2 (Ubuntu 24.04) with direct GPU passthrough

---

## 📁 Project Structure

```text
├── 00_Initial_Setup_and_Architecture_Guide.pdf   # Phase 0 setup & hardware guide
├── 01_Phase1_Model_Export_Quantization_Inference_Guide.pdf # Phase 1 deployment guide
├── configs/
│   └── vaip_config.json                         # AMD Vitis AI Execution Provider config
├── models/
│   ├── mobilenet_v2.onnx                        # Step 1: Exported FP32 ONNX graph
│   └── mobilenet_v2_quantized.onnx              # Step 2: AMD Quark INT8 quantized graph
├── src/
│   ├── step1_export_onnx.py                     # Pretrained PyTorch model to ONNX export
│   ├── step2_quark_quantize.py                  # Quantization with AMD Quark quantizer
│   └── step3_ort_inference.py                   # ONNX Runtime inference & benchmarking
├── requirements.txt                             # Core project dependencies
└── .gitignore                                   # Ignore .venv, __pycache__, and temporary files
```

---

## ⚡ Phase 1: AMD Ryzen AI Deployment Pipeline

Based on AMD's official Ryzen AI tutorials, models follow a 3-step deployment lifecycle:

1. **Step 1 — Model Conversion to ONNX (`src/step1_export_onnx.py`)**:
   * Converts dynamic PyTorch models into a static, hardware-agnostic ONNX computation graph.
   * Enables dynamic batching (`dynamic_axes`) for flexible batch sizes.
2. **Step 2 — Post-Training Quantization with AMD Quark (`src/step2_quark_quantize.py`)**:
   * Uses AMD's official `amd-quark` quantizer with a calibration data reader.
   * Compresses 32-bit floats (FP32) into 8-bit integers (INT8) with QDQ (Quantize-Dequantize) operators.
3. **Step 3 — Deployment with ONNX Runtime (`src/step3_ort_inference.py`)**:
   * Deploys the model with multi-provider execution fallback:
     1. `VitisAIExecutionProvider` (AMD Ryzen AI NPU)
     2. `CUDAExecutionProvider` (NVIDIA RTX 3050 6GB GPU)
     3. `DmlExecutionProvider` (AMD Radeon 660M iGPU via DirectML)
     4. `CPUExecutionProvider` (AMD Ryzen 6600H AVX2)

---

## 🚀 Quickstart

1. **Create & activate the virtual environment**:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

3. **Run the pipeline**:
   ```powershell
   python src/step1_export_onnx.py
   python src/step2_quark_quantize.py
   python src/step3_ort_inference.py
   ```
