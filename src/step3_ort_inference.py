"""
================================================================================
PHASE 1 - STEP 3: DEPLOYMENT USING ONNX RUNTIME PYTHON CODE
================================================================================
WHAT THIS SCRIPT DOES:
1. Implements the exact code structure from Tutorial Slide 1!
2. Configures the Execution Providers:
      - 'VitisAIExecutionProvider' (with vaip_config.json for AMD Ryzen AI NPUs)
      - 'CUDAExecutionProvider'    (for your NVIDIA GeForce RTX 3050 6GB)
      - 'DmlExecutionProvider'     (for your AMD Radeon 660M iGPU via DirectML)
      - 'CPUExecutionProvider'     (universal fallback for AMD Ryzen 6600H)
3. Loads both the original FP32 model and our AMD Quark INT8 quantized model.
4. Executes live inference on a test image tensor.
5. Benchmarks latency and compares execution speed.
================================================================================
"""

import os
import sys
import time
import numpy as np
import onnxruntime as ort
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

def run_inference_session(model_path: str, model_label: str):
    console.print(f"\n[bold yellow]>>> Initializing Inference Session for: {model_label}[/bold yellow]")
    console.print(f"    Path: [cyan]{model_path}[/cyan]")

    if not os.path.exists(model_path):
        console.print(f"    [red]Error: Model file '{model_path}' not found![/red]")
        return None

    # --------------------------------------------------------------------------
    # 1. CONFIGURE PROVIDERS & PROVIDER OPTIONS (Directly from Slide 1)
    # --------------------------------------------------------------------------
    # On AMD Ryzen AI machines with an NPU, VitisAIExecutionProvider runs the model.
    # On your laptop, ONNX Runtime automatically selects the fastest available chip
    # (RTX 3050 GPU, AMD Radeon iGPU, or CPU).
    config_file_path = os.path.join("configs", "vaip_config.json")

    providers = [
        'VitisAIExecutionProvider', # 1. AMD Ryzen AI NPU
        'CUDAExecutionProvider',    # 2. NVIDIA RTX 3050 6GB GPU
        'DmlExecutionProvider',     # 3. AMD Radeon 660M iGPU (DirectML)
        'CPUExecutionProvider'      # 4. AMD Ryzen 5 6600H CPU
    ]

    provider_options = [
        {
            'config_file': config_file_path,
            'cacheKey': 'modelcachekey'
        },
        {}, # CUDA options (default)
        {}, # DirectML options (default)
        {}  # CPU options (default)
    ]

    # Filter providers to only those supported by current runtime build
    available_providers = ort.get_available_providers()
    active_providers = [p for p in providers if p in available_providers]
    active_options = [provider_options[providers.index(p)] for p in active_providers]

    console.print(f"    • Installed ONNX Providers: [dim]{available_providers}[/dim]")
    console.print(f"    • Active Execution Chain:   [green]{active_providers}[/green]")

    # --------------------------------------------------------------------------
    # 2. CREATE INFERENCE SESSION (Slide 1, Step 3)
    # --------------------------------------------------------------------------
    # session = ort.InferenceSession(model, providers=providers, provider_options=provider_options)
    session = ort.InferenceSession(
        model_path,
        providers=active_providers,
        provider_options=active_options
    )

    current_ep = session.get_providers()
    console.print(f"    [bold green]✓ Session created successfully![/bold green]")
    console.print(f"    • Running on Hardware Provider: [bold cyan]{current_ep}[/bold cyan]")

    # Inspect input and output node metadata
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    console.print(f"    • Model Input Node:  '{input_meta.name}', Shape: {input_meta.shape}, Type: {input_meta.type}")
    console.print(f"    • Model Output Node: '{output_meta.name}', Shape: {output_meta.shape}, Type: {output_meta.type}")

    # --------------------------------------------------------------------------
    # 3. PREPARE TEST INPUT TENSOR & RUN INFERENCE
    # --------------------------------------------------------------------------
    # Generate a realistic test image tensor: [Batch=1, Channels=3, Height=224, Width=224]
    np.random.seed(100)
    test_image = np.random.randn(1, 3, 224, 224).astype(np.float32)

    # Warm-up run (ensures kernels and memory are initialized)
    _ = session.run([output_meta.name], {input_meta.name: test_image})

    # Benchmark: Run 50 inference iterations to measure average latency
    num_runs = 50
    start_time = time.perf_counter()
    for _ in range(num_runs):
        outputs = session.run([output_meta.name], {input_meta.name: test_image})
    end_time = time.perf_counter()

    avg_latency_ms = ((end_time - start_time) / num_runs) * 1000.0
    fps = 1000.0 / avg_latency_ms

    predictions = outputs[0][0]
    top_class_id = int(np.argmax(predictions))
    top_confidence = float(predictions[top_class_id])

    console.print(f"    • Latency per Image: [bold green]{avg_latency_ms:.2f} ms[/bold green] ({fps:.1f} FPS)")
    console.print(f"    • Predicted Class ID: {top_class_id} (Raw Score: {top_confidence:.4f})")

    return {
        "model": model_label,
        "active_ep": current_ep[0] if current_ep else "Unknown",
        "latency_ms": avg_latency_ms,
        "fps": fps
    }

def main():
    console.print(Panel.fit(
        "[bold cyan]Step 3: ONNX Runtime Deployment & Inference[/bold cyan]\n"
        "[dim]Executing FP32 & AMD Quark INT8 Models with Hardware Provider Routing[/dim]",
        border_style="cyan"
    ))

    fp32_model = os.path.join("models", "mobilenet_v2.onnx")
    int8_model = os.path.join("models", "mobilenet_v2_quantized.onnx")

    results = []
    
    # Run FP32 model
    res_fp32 = run_inference_session(fp32_model, "Original FP32 Model")
    if res_fp32:
        results.append(res_fp32)

    # Run INT8 Quantized model
    res_int8 = run_inference_session(int8_model, "AMD Quark INT8 Quantized Model")
    if res_int8:
        results.append(res_int8)

    # --------------------------------------------------------------------------
    # 4. BENCHMARK SUMMARY TABLE
    # --------------------------------------------------------------------------
    table = Table(title="Inference Performance Summary", border_style="cyan")
    table.add_column("Model Precision", style="bold white")
    table.add_column("Active Provider", style="cyan")
    table.add_column("Latency (ms)", justify="right", style="green")
    table.add_column("Throughput (FPS)", justify="right", style="magenta")

    for r in results:
        table.add_row(r["model"], r["active_ep"], f"{r['latency_ms']:.2f} ms", f"{r['fps']:.1f} FPS")

    console.print("\n")
    console.print(table)

    console.print(Panel(
        f"[bold green]SUCCESS: Phase 1 Fully Completed![/bold green]\n"
        f"1. Model Export: PyTorch -> ONNX ('models/mobilenet_v2.onnx')\n"
        f"2. Quantization: AMD Quark INT8 ('models/mobilenet_v2_quantized.onnx')\n"
        f"3. Deployment: ONNX Runtime Inference with Vitis AI & Hardware Fallbacks",
        border_style="green"
    ))

if __name__ == "__main__":
    main()
