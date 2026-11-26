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
            self.drawString(54, 750, "Phase 3: Tiled Matrix Multiplication (GEMM) & Shared Memory Optimization")
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
    secondary = colors.HexColor("#2563EB")  # Royal Blue / AI Compute
    accent = colors.HexColor("#D97706")     # Amber / Memory Cache
    text_dark = colors.HexColor("#1E293B")
    code_bg = colors.HexColor("#F8FAFC")
    border_color = colors.HexColor("#CBD5E1")

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
    story.append(Paragraph("Phase 3: Tiled Matrix Multiplication (GEMM)", styles['DocTitle']))
    story.append(Paragraph("High-Performance GPU Computing: Shared Memory (LDS) Tiling in AMD HIP & NVIDIA CUDA", styles['DocSubTitle']))
    story.append(HRFlowable(width="100%", thickness=2, color=secondary, spaceAfter=14))

    # Executive Overview
    story.append(Paragraph("1. Why GEMM is the Mathematical Heart of Modern AI", styles['SectionHeader']))
    story.append(Paragraph(
        "General Matrix Multiplication (GEMM: <code>C = A &times; B</code>) is the computational backbone of modern deep learning. "
        "Over <b>90% to 99%</b> of all floating-point operations in neural networks—including Convolutional layers, Fully Connected / MLP layers, "
        "and the Multi-Head Attention mechanism in Large Language Models (Transformers)—are mathematically executed as GEMM operations.",
        styles['CustomBody']
    ))
    story.append(Paragraph(
        "In Phase 2, we implemented 1D Vector Addition (<code>C[i] = A[i] + B[i]</code>), which is <i>memory-bound</i> (1 add operation per 2 memory reads). "
        "In contrast, GEMM performs <b>2 &times; M &times; N &times; K operations</b> on <code>O(N&sup2;)</code> data. It is inherently <i>compute-bound</i>, "
        "making it the ultimate benchmark for hardware acceleration and memory hierarchy optimization.",
        styles['CustomBody']
    ))

    # 2D Thread Hierarchy
    story.append(Paragraph("2. The 2D GPU Grid & Block Coordinate System", styles['SectionHeader']))
    story.append(Paragraph(
        "While vector addition uses a 1D grid of threads, matrices are naturally two-dimensional. "
        "Modern GPU architectures allow launching 2D thread blocks and 2D grids using <code>dim3</code>:",
        styles['CustomBody']
    ))

    eq_data = [[
        Paragraph(
            "<b>2D Coordinate Mapping Formula:</b><br/>"
            "<code>int col = blockDim.x * blockIdx.x + threadIdx.x;  // Column in Matrix C</code><br/>"
            "<code>int row = blockDim.y * blockIdx.y + threadIdx.y;  // Row in Matrix C</code><br/><br/>"
            "<i>Grid Sizing:</i><br/>"
            "&bull; <code>dim3 block(TILE_WIDTH, TILE_WIDTH);  // (16, 16) = 256 threads per block</code><br/>"
            "&bull; <code>dim3 grid((N + 15) / 16, (M + 15) / 16); // Number of 2D blocks needed</code>",
            styles['CustomBody']
        )
    ]]
    t_eq = Table(eq_data, colWidths=[500])
    t_eq.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EFF6FF")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#3B82F6")),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_eq)
    story.append(Spacer(1, 10))

    # The Naive Bottleneck vs Tiling
    story.append(Paragraph("3. The Memory Bottleneck: Naive vs. Shared Memory Tiling", styles['SectionHeader']))
    story.append(Paragraph(
        "<b>The Naive Flaw:</b> In a naive kernel, thread (row, col) computes its dot product by looping through row <code>row</code> of A "
        "and column <code>col</code> of B directly from <b>Global VRAM</b>. Because neighboring threads read the exact same elements repeatedly, "
        "every single floating-point number is fetched from slow VRAM up to <b>N times</b>! This saturates the PCIe / memory bus.",
        styles['CustomBody']
    ))
    story.append(Paragraph(
        "<b>The Tiled Solution:</b> High-performance GPUs contain an ultra-fast on-chip SRAM cache called <b>Shared Memory</b> "
        "(in CUDA) or <b>Local Data Share (LDS)</b> in AMD HIP. It is over <b>50x faster</b> than Global VRAM.<br/>"
        "By dividing matrices into <code>16 &times; 16</code> tiles, all 256 threads in a block cooperatively load a single tile into Shared Memory once, "
        "synchronize with <code>__syncthreads()</code>, and compute the dot product entirely from high-speed cache. "
        "This reduces Global Memory traffic by a factor of <b>16x</b>!",
        styles['CustomBody']
    ))

    # Kernel Code Snippet
    story.append(Paragraph("4. The Tiled Shared Memory Kernel (src/kernels/matrix_multiply.hip)", styles['SectionHeader']))
    code_text = (
        "// Cooperative Tiled GEMM with Shared Memory (LDS)\n"
        "GPU_SHARED float s_A[TILE_WIDTH][TILE_WIDTH];\n"
        "GPU_SHARED float s_B[TILE_WIDTH][TILE_WIDTH];\n\n"
        "for (int phase = 0; phase < numTiles; phase++) {\n"
        "    // Cooperative load into on-chip cache\n"
        "    s_A[ty][tx] = A[row * K + (phase * TILE_WIDTH + tx)];\n"
        "    s_B[ty][tx] = B[(phase * TILE_WIDTH + ty) * N + col];\n"
        "    GPU_BARRIER(); // Synchronize block (__syncthreads)\n\n"
        "    // Compute dot product from ultra-fast SRAM\n"
        "    for (int k = 0; k < TILE_WIDTH; k++)\n"
        "        sum += s_A[ty][k] * s_B[k][tx];\n\n"
        "    GPU_BARRIER(); // Synchronize before loading next tile\n"
        "}\n"
        "C[row * N + col] = sum;"
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
    story.append(Paragraph("5. Performance Benchmarks: [512 x 512] Matrix Multiplication", styles['SectionHeader']))
    story.append(Paragraph(
        "We executed <code>src/test_gemm.py</code> on our AMD Ryzen 5 6600H system with NVIDIA RTX 3050 Laptop GPU. "
        "Workload: <code>512 &times; 512 &times; 512</code> = <b>268.4 Million FLOPs</b>:",
        styles['CustomBody']
    ))

    bench_data = [
        [Paragraph("<b>Execution Paradigm</b>", styles['CustomBody']),
         Paragraph("<b>Latency (ms)</b>", styles['CustomBody']),
         Paragraph("<b>Compute (GFLOP/s)</b>", styles['CustomBody']),
         Paragraph("<b>Speedup Multiplier</b>", styles['CustomBody'])],
        [Paragraph("<b>Pure Python Triply-Nested Loop</b>", styles['CustomBody']),
         Paragraph("13,567.54 ms", styles['CustomBody']),
         Paragraph("0.02 GFLOP/s", styles['CustomBody']),
         Paragraph("<b>1.0x (Baseline)</b>", styles['CustomBody'])],
        [Paragraph("<b>Native C++ Tiled Kernel (WSL2)</b>", styles['CustomBody']),
         Paragraph("313.15 ms", styles['CustomBody']),
         Paragraph("0.86 GFLOP/s", styles['CustomBody']),
         Paragraph("<b>43.3x faster</b>", styles['CustomBody'])],
        [Paragraph("<b>NumPy BLAS (OpenBLAS AVX2)</b>", styles['CustomBody']),
         Paragraph("2.51 ms", styles['CustomBody']),
         Paragraph("106.95 GFLOP/s", styles['CustomBody']),
         Paragraph("<b>5,405x faster</b>", styles['CustomBody'])],
        [Paragraph("<b>PyTorch Multi-Threaded Engine</b>", styles['CustomBody']),
         Paragraph("1.87 ms", styles['CustomBody']),
         Paragraph("143.17 GFLOP/s", styles['CustomBody']),
         Paragraph("<b>7,236x faster</b>", styles['CustomBody'])],
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

    # Architecture Takeaways
    story.append(Paragraph("6. Key Engineering Insights", styles['SectionHeader']))
    story.append(Paragraph(
        "<b>1. Algorithmic Complexity:</b> The triply nested loop scales at <code>O(N&sup3;)</code>. While vector addition scaled linearly, "
        "GEMM at 512x512 requires 268 million operations, rendering interpreted Python (~13.5 seconds) unusable for real-time AI.<br/>"
        "<b>2. Why PyTorch Hits 143 GFLOP/s:</b> PyTorch and optimized BLAS libraries combine multi-level cache tiling (L1, L2, L3 cache), "
        "loop unrolling, AVX2 fused multiply-add (FMA) instructions, and register blocking across all 12 CPU hardware threads.<br/>"
        "<b>3. Foundation for Tensor Cores & FlashAttention:</b> The exact shared memory tiling principles learned in this phase "
        "form the core foundation of modern AI breakthroughs, including NVIDIA Tensor Core WMMA / MMA instructions and Dao et al.'s FlashAttention.",
        styles['CustomBody']
    ))

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f">> PDF successfully generated: {filename}")

if __name__ == "__main__":
    out_pdf = "03_Phase3_Tiled_Matrix_Multiplication_and_Shared_Memory_Guide.pdf"
    build_pdf(out_pdf)
