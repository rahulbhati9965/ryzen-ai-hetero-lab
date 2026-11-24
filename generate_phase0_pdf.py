import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Adds page numbers and header/footer to each page dynamically."""
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
        
        # Draw header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "Phase 0: Environment Setup, Hardware Architecture & AI Foundations")
            self.drawRightString(558, 750, "AI Engineering Lab")
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)
            
        # Draw footer
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(54, 45, 558, 45)
        self.drawString(54, 32, "AMD Ryzen AI + CUDA / ROCm Engineering Guide")
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
    
    # Custom Palette
    C_PRIMARY = colors.HexColor("#0F172A")    # Deep Navy Slate
    C_SECONDARY = colors.HexColor("#2563EB")  # Modern Indigo Blue
    C_ACCENT = colors.HexColor("#DC2626")     # AMD Red accent
    C_BODY = colors.HexColor("#334155")       # Charcoal Body Text
    C_CODE_BG = colors.HexColor("#F8FAFC")    # Clean Code BG
    C_CODE_BORDER = colors.HexColor("#CBD5E1")# Code Border
    C_BOX_BG = colors.HexColor("#F1F5F9")     # Highlight Card BG

    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=C_PRIMARY,
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=C_SECONDARY,
        spaceAfter=20
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=C_PRIMARY,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=C_SECONDARY,
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=C_BODY,
        spaceAfter=7
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=C_BODY,
        leftIndent=15,
        spaceAfter=4
    )

    code_style = ParagraphStyle(
        'Code_Custom',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#0F172A")
    )

    story = []

    # Title Banner
    story.append(Paragraph("Phase 0: Environment Setup & Architecture Guide", title_style))
    story.append(Paragraph("A Complete Beginner's Foundation: Hardware Landscape, Python Systems, Command History, and AMD Ryzen AI Pipeline", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=C_SECONDARY, spaceBefore=0, spaceAfter=14))

    # SECTION 1: Hardware Landscape
    story.append(Paragraph("1. Hardware Landscape: Heterogeneous AI Accelerator", h1_style))
    story.append(Paragraph(
        "Modern deep learning workloads are distributed across specialized compute units. Before writing a single line of AI code, "
        "we inspected and documented every processing unit present on your physical machine:",
        body_style
    ))

    hw_data = [
        [Paragraph("<b>Component</b>", body_style), Paragraph("<b>Hardware Detected</b>", body_style), Paragraph("<b>Role in AI & Deployment</b>", body_style)],
        [
            Paragraph("<b>CPU</b>", body_style),
            Paragraph("AMD Ryzen 5 6600H<br/>(6 Cores / 12 Threads, Zen 3+)", body_style),
            Paragraph("Data loading, preprocessing, model orchestration, and fallback execution via AVX2 instructions.", body_style)
        ],
        [
            Paragraph("<b>Dedicated GPU (dGPU)</b>", body_style),
            Paragraph("NVIDIA GeForce RTX 3050 Laptop GPU<br/>(6 GB GDDR6 VRAM, Ampere)", body_style),
            Paragraph("Heavy deep learning training, high-throughput inference, Tensor Cores, and native CUDA execution.", body_style)
        ],
        [
            Paragraph("<b>Integrated GPU (iGPU)</b>", body_style),
            Paragraph("AMD Radeon Graphics 660M<br/>(RDNA 2, 512 MB shared memory)", body_style),
            Paragraph("Energy-efficient graphics acceleration and DirectML (DirectX 12) compute on Windows.", body_style)
        ],
        [
            Paragraph("<b>Linux Subsystem</b>", body_style),
            Paragraph("WSL2 (Ubuntu 24.04)<br/>Verified with NVIDIA GPU Passthrough", body_style),
            Paragraph("Compiling portable AMD ROCm and HIP C++ kernels natively without Windows driver limitations.", body_style)
        ]
    ]

    t_hw = Table(hw_data, colWidths=[110, 180, 214])
    t_hw.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E2E8F0")),
        ('TEXTCOLOR', (0,0), (-1,0), C_PRIMARY),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
    ]))
    story.append(t_hw)
    story.append(Spacer(1, 14))

    # SECTION 2: Python Fundamentals for Beginners
    story.append(Paragraph("2. Python Fundamentals: Absolute Beginner's Primer", h1_style))
    story.append(Paragraph(
        "If you have never written Python before, do not worry. Python is human-readable and works like an interactive notebook. "
        "Here are the core concepts used across every AI project:",
        body_style
    ))

    # Sub-item 2.1: Virtual Environments
    story.append(Paragraph("Concept 1: What is a Virtual Environment (.venv)?", h2_style))
    story.append(Paragraph(
        "Normally, installing software puts packages globally on your computer, which can create conflicts when two projects require "
        "different versions of the same library. A <b>virtual environment</b> is simply an isolated folder that holds its own Python binary and libraries. "
        "When you activate it, anything you install stays neatly inside this project directory.",
        body_style
    ))

    # Sub-item 2.2: Python 3.11 vs 3.14
    story.append(Paragraph("Concept 2: Why Python 3.11 instead of Python 3.14?", h2_style))
    story.append(Paragraph(
        "In AI engineering, we do <b>not</b> blindly pick the newest Python release. AI libraries like <b>PyTorch, ONNX Runtime, and AMD Quark</b> "
        "contain high-performance compiled C++ and CUDA binaries called <b>wheels</b>. Python 3.14 is currently an experimental pre-release without "
        "pre-compiled wheels. Attempting to install AI frameworks on 3.14 fails immediately. <b>Python 3.11</b> is the official production standard "
        "fully validated by AMD Ryzen AI, NVIDIA CUDA, and PyTorch.",
        body_style
    ))

    # Sub-item 2.3: Python Syntax Cheatsheet
    story.append(Paragraph("Concept 3: Core Python Syntax You Will See in AI Code", h2_style))
    story.append(Paragraph(
        "Review these foundational blocks—every script in our pipeline is constructed from them:",
        body_style
    ))

    def make_code_box(code_text):
        content = [Paragraph(f"<font face='Courier' size='8'>{line}</font>", code_style) for line in code_text.strip().split('\n')]
        box = Table([[content]], colWidths=[504])
        box.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), C_CODE_BG),
            ('BOX', (0,0), (-1,-1), 0.8, C_CODE_BORDER),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('LEFTPADDING', (0,0), (-1,-1), 10),
            ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ]))
        return box

    syntax_demo = """# 1. COMMENTS: Any line starting with '#' is ignored by Python (notes for humans).

# 2. IMPORTS: Bring in external libraries into your code
import torch              # PyTorch library for neural network math
import onnxruntime as ort # Give a short nickname 'ort' for convenience

# 3. VARIABLES & DATA TYPES: Storing values in memory
batch_size = 1            # An integer (whole number)
learning_rate = 0.001     # A float (decimal number)
model_name = "resnet18"   # A string (text inside quotes)

# 4. LISTS: An ordered collection of items enclosed in square brackets []
providers = ['VitisAIExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']

# 5. DICTIONARIES: Key-Value pairs enclosed in curly braces {}
provider_options = {
    'config_file': 'configs/vaip_config.json',
    'cacheKey': 'model_cache_v1'
}

# 6. FUNCTIONS: Reusable blocks of code that take inputs and return outputs
def run_inference(image_tensor):
    # 'def' defines the function name and its parameter
    output = model(image_tensor)
    return output"""

    story.append(make_code_box(syntax_demo))
    story.append(Spacer(1, 14))

    # SECTION 3: Step-by-Step Command Log
    story.append(Paragraph("3. Step-by-Step Shell Commands Executed", h1_style))
    story.append(Paragraph(
        "Every operation executed to establish our clean backend environment, explained step by step:",
        body_style
    ))

    cmd_steps = [
        ("Step 1: Inspect Hardware & Drivers", "nvidia-smi; Get-CimInstance Win32_VideoController", "Verified physical presence of NVIDIA RTX 3050 6GB Laptop GPU and AMD Radeon 660M."),
        ("Step 2: Create Isolated Virtual Environment", "python -m venv .venv", "Generates the isolated Python 3.11 sandbox directory '.venv' in our project folder."),
        ("Step 3: Upgrade Package Installer (pip)", ".\\.venv\\Scripts\\python.exe -m pip install --upgrade pip", "Ensures modern wheel caching and resolving capabilities."),
        ("Step 4: Install Core ONNX & Tooling", ".\\.venv\\Scripts\\pip.exe install onnx onnxruntime reportlab rich numpy", "Installs the ONNX computation graph engine, ONNX Runtime, and ReportLab PDF generator."),
        ("Step 5: Install PyTorch & Vision Toolkit", ".\\.venv\\Scripts\\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cpu", "Installs PyTorch to build, train, and trace deep learning model graphs."),
        ("Step 6: Install AMD Quark Quantizer", ".\\.venv\\Scripts\\pip.exe install amd-quark", "Installs AMD's official quantization toolkit for converting models to INT8 / XINT8 for NPUs.")
    ]

    for title, cmd, desc in cmd_steps:
        story.append(Paragraph(f"<b>{title}</b>", h2_style))
        story.append(Paragraph(f"<i>Purpose:</i> {desc}", body_style))
        story.append(make_code_box(f"> {cmd}"))
        story.append(Spacer(1, 4))

    story.append(Spacer(1, 10))

    # SECTION 4: Files Created So Far
    story.append(Paragraph("4. Project Files Created & Their Purpose", h1_style))
    story.append(Paragraph(
        "The project is structured according to enterprise AI engineering standards:",
        body_style
    ))

    file_data = [
        [Paragraph("<b>File / Folder</b>", body_style), Paragraph("<b>Contents / Purpose</b>", body_style)],
        [
            Paragraph("<b>.venv/</b>", body_style),
            Paragraph("The isolated virtual environment containing Python 3.11, pip, PyTorch, ONNX, and AMD Quark binaries.", body_style)
        ],
        [
            Paragraph("<b>requirements.txt</b>", body_style),
            Paragraph("Tracks exact project dependencies so any developer or CI/CD system can reproduce the environment with one command.", body_style)
        ],
        [
            Paragraph("<b>configs/vaip_config.json</b>", body_style),
            Paragraph("The mandatory runtime configuration required by <code>VitisAIExecutionProvider</code> when compiling model graphs for AMD NPUs/DPUs.", body_style)
        ],
        [
            Paragraph("<b>models/</b>", body_style),
            Paragraph("Dedicated storage directory where raw PyTorch weights (.pt), exported graphs (.onnx), and quantized models (.onnx) will be saved.", body_style)
        ],
        [
            Paragraph("<b>src/</b>", body_style),
            Paragraph("Modular source code directory housing our 3-step pipeline scripts and API services.", body_style)
        ]
    ]

    t_files = Table(file_data, colWidths=[160, 344])
    t_files.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E2E8F0")),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
    ]))
    story.append(t_files)
    story.append(Spacer(1, 14))

    # SECTION 5: The 3-Step AMD Ryzen AI Pipeline
    story.append(Paragraph("5. The 3-Step AMD Ryzen AI Deployment Pipeline", h1_style))
    story.append(Paragraph(
        "Connecting directly to the slides you uploaded, here is how each phase transforms the AI model:",
        body_style
    ))

    pipeline_steps = [
        ("Step 1: Pretrained Model Conversion to ONNX",
         "PyTorch models run dynamically in Python, which is flexible during research but slow for production hardware. "
         "By passing a dummy input tensor through the model, <code>torch.onnx.export()</code> records every mathematical operator into a universal "
         "static computation graph called <b>ONNX (Open Neural Network Exchange)</b>."),
        
        ("Step 2: Quantization using AMD Quark Quantizer",
         "Standard models store weights in 32-bit floating point (FP32). <b>AMD Quark</b> analyzes layer activations using calibration data and converts "
         "weights to 8-bit integers (INT8 / XINT8). This shrinks model size by <b>75%</b> and enables the AMD NPU to perform blazing-fast bit-shift arithmetic."),
         
        ("Step 3: Deployment using ONNX Runtime Python Code",
         "The quantized model is loaded into <code>onnxruntime.InferenceSession</code> with <code>VitisAIExecutionProvider</code> and <code>vaip_config.json</code>. "
         "If an NPU is absent, the execution provider chain automatically falls back to <b>CUDA</b> (your RTX 3050), <b>DirectML</b> (AMD Radeon iGPU), or <b>CPU</b>.")
    ]

    for p_title, p_desc in pipeline_steps:
        story.append(Paragraph(f"<b>{p_title}</b>", h2_style))
        story.append(Paragraph(p_desc, body_style))
        story.append(Spacer(1, 3))

    story.append(Spacer(1, 10))

    # SECTION 6: Next Steps
    story.append(Paragraph("6. What We Are Building Next in Phase 1", h1_style))
    story.append(Paragraph(
        "Now that our foundations are established, we will implement <b>Step 1 (Model Export to ONNX)</b> with a real computer vision model "
        "(MobileNetV2 / ResNet), add step-by-step commentary to teach you tensor operations, export the graph, verify it with ONNX checker, "
        "and generate the Phase 1 milestone report.",
        body_style
    ))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF at: {filename}")

if __name__ == '__main__':
    target_path = r"C:\Users\rahul\Desktop\amd ai projects\ai project learning\01_Initial_Setup_and_Architecture_Guide.pdf"
    build_pdf(target_path)

