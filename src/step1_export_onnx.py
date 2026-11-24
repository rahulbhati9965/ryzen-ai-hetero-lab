"""
================================================================================
PHASE 1 - STEP 1: PRETRAINED MODEL CONVERSION TO ONNX
================================================================================
Welcome to Step 1 of your AI deployment journey!

WHAT THIS SCRIPT DOES:
1. Loads a real-world Convolutional Neural Network (MobileNetV2) using PyTorch.
2. Explains what Tensors, Weights, and Forward Passes are.
3. Traces the model using a 'Dummy Input Tensor'.
4. Exports the PyTorch model into ONNX (Open Neural Network Exchange) format.
5. Verifies the exported ONNX file to ensure zero mathematical errors.

WHY DO WE CONVERT PYTORCH TO ONNX?
- PyTorch is designed for researchers: it runs python code dynamically line-by-line.
- Hardware accelerators (like AMD NPUs, GPUs, and edge chips) need a STATIC graph:
  a frozen mathematical blueprint where all shapes, operations, and memory allocations
  are planned in advance.
- ONNX is that universal blueprint. It allows a model trained in PyTorch to run on
  AMD Ryzen AI, NVIDIA TensorRT, Apple CoreML, or Android devices!
================================================================================
"""

import os
import sys
import torch
import torchvision.models as models
import onnx
from rich.console import Console
from rich.panel import Panel

# Initialize rich console for beautiful, clear terminal logs
console = Console()

def main():
    console.print(Panel.fit(
        "[bold cyan]Step 1: Exporting Pretrained PyTorch Model to ONNX[/bold cyan]\n"
        "[dim]Converting dynamic PyTorch graph to static ONNX blueprint[/dim]",
        border_style="cyan"
    ))

    # --------------------------------------------------------------------------
    # 1. DEFINE PATHS & DIRECTORIES
    # --------------------------------------------------------------------------
    # In Python, os.path helps us create folder paths that work on both Windows & Linux.
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)  # Create 'models/' folder if it doesn't exist
    
    onnx_file_path = os.path.join(output_dir, "mobilenet_v2.onnx")

    # --------------------------------------------------------------------------
    # 2. LOAD A PRETRAINED NEURAL NETWORK (MobileNetV2)
    # --------------------------------------------------------------------------
    # What is MobileNetV2?
    # It is a state-of-the-art vision model designed specifically for edge devices,
    # mobile phones, and NPUs. It takes an image and classifies it into 1,000 categories
    # (e.g. golden retriever, sports car, coffee mug).
    console.print("[bold yellow][1/5] Loading pretrained MobileNetV2 model...[/bold yellow]")
    
    # We load the model with pretrained ImageNet weights
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # CRITICAL AI STEP: Set the model to EVALUATION mode (.eval())
    # Why? Neural networks have layers like 'Dropout' and 'BatchNorm' that behave
    # differently during training vs inference. .eval() freezes them for production!
    model.eval()
    console.print("   [green]✓ MobileNetV2 loaded successfully and set to eval mode.[/green]")

    # --------------------------------------------------------------------------
    # 3. CREATE A 'DUMMY INPUT TENSOR'
    # --------------------------------------------------------------------------
    # What is a Tensor?
    # A tensor is simply a multi-dimensional array of numbers (like a spreadsheet with layers).
    #
    # In Computer Vision, input tensors follow the (B, C, H, W) format:
    #   B = Batch Size (How many images we process at once, here = 1)
    #   C = Channels   (Color channels: 3 for Red, Green, Blue)
    #   H = Height     (Standard image height = 224 pixels)
    #   W = Width      (Standard image width = 224 pixels)
    #
    # PyTorch needs to 'see' an example input so it can trace every mathematical calculation
    # from the input layer all the way to the output predictions.
    console.print("[bold yellow][2/5] Generating dummy input tensor (Shape: [1, 3, 224, 224])...[/bold yellow]")
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)
    console.print("   [green]✓ Dummy tensor created (random noise representing a test image).[/green]")

    # --------------------------------------------------------------------------
    # 4. EXPORT TO ONNX (torch.onnx.export)
    # --------------------------------------------------------------------------
    # Here is where the magic happens:
    # - 'model': Our PyTorch neural network
    # - 'dummy_input': The sample tensor used to trace the flow of math
    # - 'onnx_file_path': Destination file where the .onnx blueprint will be saved
    # - 'export_params=True': Stores the trained weights inside the ONNX file
    # - 'opset_version=17': Standard modern ONNX operator set version
    # - 'input_names' & 'output_names': Human-readable names for the graph input/output
    # - 'dynamic_axes': Enables flexible batch sizing (process 1 image or 100 images!)
    console.print(f"[bold yellow][3/5] Tracing and exporting graph to '{onnx_file_path}'...[/bold yellow]")

    input_names = ["input_image"]
    output_names = ["class_probabilities"]

    # dynamic_axes tells the ONNX engine that Dimension 0 (Batch Size) can change at runtime
    dynamic_axes = {
        "input_image": {0: "batch_size"},
        "class_probabilities": {0: "batch_size"}
    }

    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True, # Optimizes constant math operations at export time
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )

    console.print(f"   [green]✓ Export complete! File saved at: {onnx_file_path}[/green]")

    # --------------------------------------------------------------------------
    # 5. VALIDATE THE EXPORTED ONNX BLUEPRINT
    # --------------------------------------------------------------------------
    # Before deploying to AMD Ryzen AI or GPUs, we must verify that the ONNX graph
    # is mathematically sound and has no broken connections or missing operators.
    console.print("[bold yellow][4/5] Verifying ONNX graph integrity with onnx.checker...[/bold yellow]")
    onnx_model = onnx.load(onnx_file_path)
    onnx.checker.check_model(onnx_model)
    console.print("   [green]✓ ONNX graph structure is valid and verified![/green]")

    # --------------------------------------------------------------------------
    # 6. INSPECT GRAPH METADATA
    # --------------------------------------------------------------------------
    console.print("[bold yellow][5/5] Inspecting Model Blueprint Metadata:[/bold yellow]")
    file_size_mb = os.path.getsize(onnx_file_path) / (1024 * 1024)
    
    graph = onnx_model.graph
    input_info = graph.input[0]
    output_info = graph.output[0]

    console.print(f"   • File Size:           [bold cyan]{file_size_mb:.2f} MB[/bold cyan]")
    console.print(f"   • Graph Input Name:    [bold cyan]{input_info.name}[/bold cyan]")
    console.print(f"   • Graph Output Name:   [bold cyan]{output_info.name}[/bold cyan]")
    console.print(f"   • Total Nodes/Layers:  [bold cyan]{len(graph.node)} operations[/bold cyan]")
    
    console.print(Panel(
        f"[bold green]SUCCESS: Step 1 Complete![/bold green]\n"
        f"Your model is now saved as a universal ONNX blueprint: [bold]{onnx_file_path}[/bold]\n"
        f"Ready for Step 2: Quantization with AMD Quark Quantizer!",
        border_style="green"
    ))

if __name__ == "__main__":
    main()