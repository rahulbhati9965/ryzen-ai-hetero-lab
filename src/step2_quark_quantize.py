"""
================================================================================
PHASE 1 - STEP 2: QUANTIZATION USING AMD QUARK QUANTIZER
================================================================================
WHAT THIS SCRIPT DOES:
1. Loads our exported ONNX model ('models/mobilenet_v2.onnx').
2. Explains what Quantization is and why NPUs / edge devices require INT8.
3. Builds an image CalibrationDataReader (provides sample inputs so the quantizer
   can measure the dynamic range of activations).
4. Runs the official AMD Quark Quantizer ('quark.onnx') to convert 32-bit floating point
   (FP32) weights and activations into 8-bit integers (INT8 / XINT8).
5. Compares file sizes and verifies the quantized graph structure.

WHY DO WE QUANTIZE?
- Standard AI models store every weight as a 32-bit Float (FP32) = 4 bytes per number.
- High precision (FP32) requires huge memory bandwidth and draws significant power.
- Quantization maps FP32 numbers into 8-bit Integers (INT8) = 1 byte per number.
- RESULT:
    • Model file size drops by ~75% (e.g. 14 MB -> ~3.5 MB).
    • Memory bandwidth is cut by 4x.
    • AMD NPUs, GPUs, and Tensor Cores compute integer math 3x to 8x faster!
================================================================================
"""

import os
import sys
import numpy as np
import onnx
import onnxruntime
from onnxruntime.quantization.calibrate import CalibrationDataReader
import quark.onnx as q_onnx
from rich.console import Console
from rich.panel import Panel

console = Console()

class MobileNetCalibrationReader(CalibrationDataReader):
    """
    What is Calibration in AI Quantization?
    --------------------------------------
    When converting continuous decimals (e.g. 0.04512) into integers (e.g. 12),
    the quantizer needs to find the 'Scale' and 'Zero-point':
        Real_Value = Scale * (Quantized_Integer - Zero_Point)
    
    To calculate the optimal Scale without losing accuracy, the quantizer needs
    to observe real data flowing through the layers. This class feeds a small batch
    of sample images (calibration dataset) through the model.
    """
    def __init__(self, num_samples: int = 10, batch_size: int = 1):
        super().__init__()
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.current_sample = 0
        
        # Generate synthetic calibration data matching the model's expected shape:
        # Shape: [Batch, Channels, Height, Width] = [1, 3, 224, 224] (Normalized float32)
        np.random.seed(42)
        self.data = [
            {"input_image": np.random.randn(batch_size, 3, 224, 224).astype(np.float32)}
            for _ in range(num_samples)
        ]

    def get_next(self):
        """Yields the next dictionary of input tensors for calibration."""
        if self.current_sample < self.num_samples:
            batch = self.data[self.current_sample]
            self.current_sample += 1
            return batch
        return None

    def rewind(self):
        """Resets the reader back to the beginning."""
        self.current_sample = 0

def main():
    console.print(Panel.fit(
        "[bold cyan]Step 2: Quantization using AMD Quark Quantizer[/bold cyan]\n"
        "[dim]Compressing FP32 model to INT8 for AMD NPU & GPU Acceleration[/dim]",
        border_style="cyan"
    ))

    input_onnx_path = os.path.join("models", "mobilenet_v2.onnx")
    output_quant_path = os.path.join("models", "mobilenet_v2_quantized.onnx")

    if not os.path.exists(input_onnx_path):
        console.print(f"[bold red]Error: Input model '{input_onnx_path}' not found! Run step1_export_onnx.py first.[/bold red]")
        sys.exit(1)

    # --------------------------------------------------------------------------
    # 1. SETUP CALIBRATION DATA READER
    # --------------------------------------------------------------------------
    console.print("[bold yellow][1/4] Preparing calibration dataset (10 sample batches)...[/bold yellow]")
    calib_reader = MobileNetCalibrationReader(num_samples=10, batch_size=1)
    console.print("   [green]✓ Calibration reader ready (feeds sample inputs to monitor layer dynamic range).[/green]")

    # --------------------------------------------------------------------------
    # 2. RUN AMD QUARK QUANTIZER (Static INT8 Post-Training Quantization)
    # --------------------------------------------------------------------------
    # What parameters are we passing?
    # - 'model_input': The FP32 ONNX model from Step 1.
    # - 'model_output': Where to write the quantized INT8 model.
    # - 'calibration_data_reader': Feeds sample data to compute scale factors.
    # - 'activation_type': QuantType.QInt8 (8-bit signed integer activations).
    # - 'weight_type': QuantType.QInt8 (8-bit signed integer weights).
    # - 'quant_format': QDQ (Quantize-Dequantize nodes inserted into graph).
    console.print(f"[bold yellow][2/4] Running AMD Quark Quantizer on '{input_onnx_path}'...[/bold yellow]")
    
    try:
        q_onnx.quantize_static(
            model_input=input_onnx_path,
            model_output=output_quant_path,
            calibration_data_reader=calib_reader,
            quant_format=onnxruntime.quantization.QuantFormat.QDQ,
            activation_type=onnxruntime.quantization.QuantType.QInt8,
            weight_type=onnxruntime.quantization.QuantType.QInt8,
            per_channel=False,
            reduce_range=False,
            optimize_model=True
        )
        console.print(f"   [green]✓ Quantization succeeded! Output saved to: {output_quant_path}[/green]")
    except Exception as e:
        console.print(f"   [yellow]Static QDQ fallback: {e}[/yellow]")
        console.print("   [yellow]Running dynamic quantization fallback...[/yellow]")
        q_onnx.quantize_dynamic(
            model_input=input_onnx_path,
            model_output=output_quant_path,
            weight_type=onnxruntime.quantization.QuantType.QInt8
        )
        console.print(f"   [green]✓ Dynamic Quantization complete: {output_quant_path}[/green]")

    # --------------------------------------------------------------------------
    # 3. VERIFY THE QUANTIZED ONNX GRAPH
    # --------------------------------------------------------------------------
    console.print("[bold yellow][3/4] Verifying quantized model integrity...[/bold yellow]")
    quant_model = onnx.load(output_quant_path)
    onnx.checker.check_model(quant_model)
    console.print("   [green]✓ Quantized ONNX model is verified and valid.[/green]")

    # --------------------------------------------------------------------------
    # 4. MEASURE FILE SIZE COMPRESSION RATIO
    # --------------------------------------------------------------------------
    console.print("[bold yellow][4/4] Compression Analysis (FP32 vs INT8):[/bold yellow]")
    orig_size = os.path.getsize(input_onnx_path) / (1024 * 1024)
    quant_size = os.path.getsize(output_quant_path) / (1024 * 1024)
    reduction = ((orig_size - quant_size) / orig_size) * 100 if orig_size > 0 else 0

    console.print(f"   • Original FP32 Model:  [bold red]{orig_size:.2f} MB[/bold red]")
    console.print(f"   • Quantized INT8 Model: [bold green]{quant_size:.2f} MB[/bold green]")
    console.print(f"   • Size Reduction:       [bold cyan]{reduction:.1f}% smaller[/bold cyan]")

    console.print(Panel(
        f"[bold green]SUCCESS: Step 2 Complete![/bold green]\n"
        f"Your INT8 quantized model is ready: [bold]{output_quant_path}[/bold]\n"
        f"Ready for Step 3: Deployment & Inference using ONNX Runtime!",
        border_style="green"
    ))

if __name__ == "__main__":
    main()
