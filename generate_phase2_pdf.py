import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Dynamic page numbering and running headers/footers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Header (Pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "Phase 2: High-Performance GPU Acceleration (CUDA & AMD HIP)")
            self.drawRightString(558, 750, "Heterogeneous AI Lab")
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)
            
        # Footer
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(54, 45, 558, 45)
        self.drawString(54, 32, "AMD Ryzen AI + NVIDIA RTX Heterogeneous Computing Guide")
        self.drawRightString(558, 32, f"Page {self._pageNumber} of {total_pages}")
        self.restoreState()

def build_pdf(filename):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom palette
    primary = colors.HexColor("#0F172A")    # Deep slate
    secondary = colors.HexColor("#D97706")  # Amber / GPU Gold
    amd_red = colors.HexColor("#ED1C24")    # AMD Red
    nvidia_green = colors.HexColor("#76B900")# NVIDIA Green
    text_dark = colors.HexColor("#1E293B")
    text_muted = colors.HexColor("#64748B")
    code_bg = colors.HexColor("#F8FAFC")
    border_color = colors.HexColor("#CBD5E1")

    # Typography styles
    styles.add(ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=primary,
        spaceAfter=6
    ))

    styles.add(ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=secondary,
        spaceAfter=15
    ))

    styles.add(ParagraphStyle(
        'SectionHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=primary,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    ))

    styles.add(ParagraphStyle(
        'SubSectionHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=text_dark,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    ))

    styles.add(ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=text_dark,
        spaceAfter=6
    ))

    styles.add(ParagraphStyle(
        'CodeSnippet',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#0F172A")
    ))

    story = []

    # Title Banner
    story.append(Paragraph("Phase 2: High-Performance GPU Acceleration", styles['DocTitle']))
    story.append(Paragraph("Systems-Level Heterogeneous Computing: Single-Source C++ Kernels with AMD HIP & NVIDIA CUDA", styles['DocSubTitle']))
    story.append(HRFlowable(width="100%", thickness=2, color=secondary, spaceAfter=14))

    # Executive Overview
    story.append(Paragraph("1. Executive Summary & The GPU Computing Paradigm", styles['SectionHeader']))
    story.append(Paragraph(
        "Modern Artificial Intelligence workloads are bound not by raw CPU clock speed, but by memory bandwidth and "
        "massive parallel floating-point throughput. While a modern CPU (such as the AMD Ryzen 5 6600H) utilizes 6 high-frequency "
        "cores optimized for complex branch prediction and serial latency, a GPU (such as the NVIDIA RTX 3050) deploys thousands "
        "of lightweight arithmetic logic units (ALUs) designed for throughput.",
        styles['CustomBody']
    ))
    story.append(Paragraph(
        "In this phase, we unlock systems-level GPU programming. Rather than relying solely on black-box frameworks, "
        "we write, compile, and benchmark bare-metal GPU kernels using <b>AMD HIP (Heterogeneous-Compute Interface for Portability)</b>. "
        "HIP allows developers to write single-source C++ code that executes identically on both NVIDIA GPUs (via CUDA) and AMD GPUs (via ROCm).",
        styles['CustomBody']
    ))

    # Architecture Comparison Table
    story.append(Paragraph("2. Architectural Foundations: CPU vs. GPU", styles['SectionHeader']))
    
    arch_data = [
        [Paragraph("<b>Feature</b>", styles['CustomBody']),
         Paragraph("<b>CPU (AMD Ryzen 5 6600H)</b>", styles['CustomBody']),
         Paragraph("<b>GPU (NVIDIA RTX 3050 Laptop)</b>", styles['CustomBody'])],
        [Paragraph("<b>Core Philosophy</b>", styles['CustomBody']),
         Paragraph("Low latency, complex out-of-order execution, deep branch prediction.", styles['CustomBody']),
         Paragraph("Massive throughput, SIMT (Single Instruction Multiple Threads).", styles['CustomBody'])],
        [Paragraph("<b>Core Count</b>", styles['CustomBody']),
         Paragraph("6 Physical Cores / 12 Threads (Zen 3+)", styles['CustomBody']),
         Paragraph("2,048 CUDA Cores / 64 Tensor Cores (Ampere)", styles['CustomBody'])],
        [Paragraph("<b>Execution Unit</b>", styles['CustomBody']),
         Paragraph("Thread (managed by OS scheduler)", styles['CustomBody']),
         Paragraph("Warp (32 threads NVIDIA) / Wavefront (64 threads AMD)", styles['CustomBody'])],
        [Paragraph("<b>Memory Model</b>", styles['CustomBody']),
         Paragraph("Host RAM (DDR5, large capacity, ~40-60 GB/s)", styles['CustomBody']),
         Paragraph("Device VRAM (GDDR6, dedicated, ~192 GB/s)", styles['CustomBody'])],
        [Paragraph("<b>Interconnect</b>", styles['CustomBody']),
         Paragraph("Direct CPU Memory Controller bus", styles['CustomBody']),
         Paragraph("PCIe Gen4 x8 Interconnect (Host-to-Device Bottleneck)", styles['CustomBody'])],
    ]

    t_arch = Table(arch_data, colWidths=[110, 195, 195])
    t_arch.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F1F5F9")),
        ('GRID', (0,0), (-1,-1), 0.5, border_color),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_arch)
    story.append(Spacer(1, 10))

    # The Thread Hierarchy
    story.append(Paragraph("3. The SIMT Execution Model & Thread Hierarchy", styles['SectionHeader']))
    story.append(Paragraph(
        "In standard CPU programming, computing vector addition <code>C[i] = A[i] + B[i]</code> requires a sequential <code>for</code> loop. "
        "In GPU programming (SIMT), the GPU launches millions of lightweight threads simultaneously, where <b>every thread computes exactly one index</b>.",
        styles['CustomBody']
    ))
    story.append(Paragraph(
        "Threads are grouped into a hierarchical structure: "
        "<br/>&bull; <b>Thread:</b> The smallest unit of execution executing the kernel."
        "<br/>&bull; <b>Warp / Wavefront:</b> Hardware lock-step execution group (32 threads on NVIDIA, 64 on AMD)."
        "<br/>&bull; <b>Thread Block (Workgroup):</b> A cluster of threads (typically 256 or 512) sharing on-chip shared memory."
        "<br/>&bull; <b>Grid:</b> The entire array of blocks required to cover the full dataset size N.",
        styles['CustomBody']
    ))

    # Thread Equation Box
    eq_data = [[
        Paragraph(
            "<b>The Universal Thread Index Formula:</b><br/>"
            "<code>int global_tid = (blockDim.x * blockIdx.x) + threadIdx.x;</code><br/><br/>"
            "<i>Where:</i><br/>"
            "&bull; <code>threadIdx.x</code> = Local offset within the current block (0 to 255)<br/>"
            "&bull; <code>blockIdx.x</code> = Which block in the grid is running (0 to grid_size - 1)<br/>"
            "&bull; <code>blockDim.x</code> = Number of threads per block (e.g., 256)",
            styles['CustomBody']
        )
    ]]
    t_eq = Table(eq_data, colWidths=[500])
    t_eq.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#FEF3C7")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#F59E0B")),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_eq)
    story.append(Spacer(1, 10))

    # Portable C++ Kernel
    story.append(Paragraph("4. The Unified AMD HIP / NVIDIA CUDA Kernel", styles['SectionHeader']))
    story.append(Paragraph(
        "Below is our portable GPU kernel located at <code>src/kernels/vector_add.hip</code>. "
        "It uses preprocessor macros to support AMD ROCm (<code>hipcc</code>), NVIDIA CUDA (<code>nvcc</code>), "
        "and standard C++ CPU compilers seamlessly:",
        styles['CustomBody']
    ))

    code_text = (
        "// src/kernels/vector_add.hip\n"
        "GPU_KERNEL void vectorAddKernel(const float* __restrict__ A,\n"
        "                                const float* __restrict__ B,\n"
        "                                float* __restrict__ C,\n"
        "                                int N) {\n"
        "    // Calculate global thread ID\n"
        "    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n\n"
        "    // Boundary check: ensure thread does not access out-of-bounds memory\n"
        "    if (idx < N) {\n"
        "        C[idx] = A[idx] + B[idx];\n"
        "    }\n"
        "}"
    )
    t_code = Table([[Paragraph(code_text.replace("\n", "<br/>").replace(" ", "&nbsp;"), styles['CodeSnippet'])]], colWidths=[500])
    t_code.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), code_bg),
        ('BOX', (0,0), (-1,-1), 0.5, border_color),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_code)
    story.append(Spacer(1, 10))

    # Benchmark Results
    story.append(Paragraph("5. Performance Benchmarks: 10,000,000 Elements Vector Addition", styles['SectionHeader']))
    story.append(Paragraph(
        "We executed <code>src/test_gpu_kernel.py</code> across 4 paradigms on our local test machine. "
        "The task processed 10,000,000 single-precision floats (~114.4 MB total memory read/write):",
        styles['CustomBody']
    ))

    bench_data = [
        [Paragraph("<b>Execution Paradigm</b>", styles['CustomBody']),
         Paragraph("<b>Latency (ms)</b>", styles['CustomBody']),
         Paragraph("<b>Effective Throughput</b>", styles['CustomBody']),
         Paragraph("<b>Speedup Multiplier</b>", styles['CustomBody'])],
        [Paragraph("<b>Python Single-Core Loop</b>", styles['CustomBody']),
         Paragraph("977.74 ms", styles['CustomBody']),
         Paragraph("0.12 GB/s", styles['CustomBody']),
         Paragraph("<b>1.0x (Baseline)</b>", styles['CustomBody'])],
        [Paragraph("<b>Native Compiled C++ Kernel</b>", styles['CustomBody']),
         Paragraph("104.14 ms", styles['CustomBody']),
         Paragraph("1.15 GB/s", styles['CustomBody']),
         Paragraph("<b>9.4x faster</b>", styles['CustomBody'])],
        [Paragraph("<b>NumPy SIMD Vectorized (AVX)</b>", styles['CustomBody']),
         Paragraph("15.19 ms", styles['CustomBody']),
         Paragraph("7.90 GB/s", styles['CustomBody']),
         Paragraph("<b>64.4x faster</b>", styles['CustomBody'])],
        [Paragraph("<b>PyTorch Optimized Tensor Engine</b>", styles['CustomBody']),
         Paragraph("8.06 ms", styles['CustomBody']),
         Paragraph("14.89 GB/s", styles['CustomBody']),
         Paragraph("<b>121.3x faster</b>", styles['CustomBody'])],
    ]

    t_bench = Table(bench_data, colWidths=[170, 95, 115, 120])
    t_bench.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#EFF6FF")),
        ('GRID', (0,0), (-1,-1), 0.5, border_color),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_bench)
    story.append(Spacer(1, 10))

    # Key Architectural Takeaways
    story.append(Paragraph("6. Key Engineering Insights", styles['SectionHeader']))
    story.append(Paragraph(
        "<b>1. Eliminating Interpreter Overhead:</b> Pure Python interprets each instruction dynamically, generating massive overhead. "
        "Native C++ executes pre-compiled machine instructions directly, yielding an immediate ~10x speedup.<br/>"
        "<b>2. SIMD Vectorization:</b> AVX2/AVX-512 vector units in the AMD Ryzen 6600H process 8 float32 values per cycle per core, "
        "jumping from 1.15 GB/s to nearly 8 GB/s in NumPy.<br/>"
        "<b>3. Multi-Threaded Parallelism:</b> PyTorch distributes chunks across all 12 CPU hardware threads, saturating the dual-channel DDR5 memory bus at 14.89 GB/s.<br/>"
        "<b>4. The PCIe Transfer Penalty:</b> When transferring data between CPU RAM and GPU VRAM over the PCIe bus, data transfer time must be amortized by high compute intensity. "
        "For simple memory-bound kernels like vector add, minimizing host-device memory movement is critical.",
        styles['CustomBody']
    ))

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f">> PDF successfully generated: {filename}")

if __name__ == "__main__":
    out_pdf = "02_Phase2_CUDA_and_HIP_GPU_Programming_Guide.pdf"
    build_pdf(out_pdf)
