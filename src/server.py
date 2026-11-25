"""
================================================================================
FASTAPI INFERENCE SERVICE FOR CONTAINERIZED DEPLOYMENT
================================================================================
This server exposes our ONNX Runtime deep learning engine over HTTP:
- GET  /health   -> Used by Kubernetes / Docker for health probes
- GET  /info     -> Returns model status, active hardware provider, and precision
- POST /predict  -> Runs inference on input image data, applies Softmax,
                    and returns human-readable labels + Top-5 probabilities!
================================================================================
"""

import os
import sys
import time
import io
import json
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
import onnxruntime as ort

app = FastAPI(
    title="Ryzen AI Heterogeneous Inference Service",
    description="Containerized ONNX Runtime inference engine with AMD Quark INT8 & Vitis AI fallbacks",
    version="1.1.0"
)

# Global session & label cache
SESSION = None
MODEL_TYPE = "INT8" # Default to quantized model
INPUT_NAME = None
OUTPUT_NAME = None
LABELS = []

def load_labels():
    global LABELS
    labels_path = os.path.join("configs", "imagenet_classes.json")
    if os.path.exists(labels_path):
        with open(labels_path, "r") as f:
            LABELS = json.load(f)
    else:
        LABELS = [f"Class {i}" for i in range(1000)]

def get_session():
    global SESSION, MODEL_TYPE, INPUT_NAME, OUTPUT_NAME
    if SESSION is not None:
        return SESSION

    load_labels()

    model_int8 = os.path.join("models", "mobilenet_v2_quantized.onnx")
    model_fp32 = os.path.join("models", "mobilenet_v2.onnx")

    if os.path.exists(model_int8):
        model_path = model_int8
        MODEL_TYPE = "AMD Quark INT8"
    elif os.path.exists(model_fp32):
        model_path = model_fp32
        MODEL_TYPE = "Original FP32"
    else:
        raise RuntimeError("No ONNX model found in models/ directory!")

    # Set up hardware execution providers
    config_path = os.path.join("configs", "vaip_config.json")
    providers = ['VitisAIExecutionProvider', 'CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']
    provider_options = [{'config_file': config_path, 'cacheKey': 'modelcachekey'}, {}, {}, {}]

    available = ort.get_available_providers()
    active_providers = [p for p in providers if p in available]
    active_options = [provider_options[providers.index(p)] for p in active_providers]

    SESSION = ort.InferenceSession(model_path, providers=active_providers, provider_options=active_options)
    INPUT_NAME = SESSION.get_inputs()[0].name
    OUTPUT_NAME = SESSION.get_outputs()[0].name
    return SESSION

@app.on_event("startup")
def startup_event():
    """Load model and labels into memory when container boots up."""
    get_session()
    print(f"[SERVER] Model initialized: {MODEL_TYPE} with {len(LABELS)} labels")

@app.get("/")
def root():
    return {
        "service": "Ryzen AI Hetero Inference API",
        "status": "online",
        "docs": "/docs"
    }

@app.get("/health")
def health():
    """Kubernetes liveness and readiness probe endpoint."""
    return {"status": "healthy", "model_loaded": SESSION is not None}

@app.get("/info")
def model_info():
    """Inspect active hardware provider and graph metadata."""
    sess = get_session()
    return {
        "model_type": MODEL_TYPE,
        "active_provider": sess.get_providers(),
        "input_node": sess.get_inputs()[0].name,
        "input_shape": sess.get_inputs()[0].shape,
        "output_node": sess.get_outputs()[0].name,
        "output_shape": sess.get_outputs()[0].shape,
        "num_classes": len(LABELS)
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file (JPEG/PNG), preprocesses it to [1, 3, 224, 224],
    runs ONNX Runtime inference, computes Softmax probabilities, and returns
    human-readable object names with a Top-5 ranking.
    """
    sess = get_session()
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess: resize to 224x224 and normalize
        image = image.resize((224, 224))
        img_np = np.array(image).astype(np.float32) / 255.0
        
        # Standard ImageNet normalization: (x - mean) / std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std
        
        # Transpose from (Height, Width, Channels) to (Channels, Height, Width)
        img_np = np.transpose(img_np, (2, 0, 1))
        # Add batch dimension: [1, 3, 224, 224]
        input_tensor = np.expand_dims(img_np, axis=0)
        
        # Run inference
        t0 = time.perf_counter()
        outputs = sess.run([OUTPUT_NAME], {INPUT_NAME: input_tensor})
        latency_ms = (time.perf_counter() - t0) * 1000.0
        
        raw_logits = outputs[0][0]
        
        # Softmax: convert raw numbers into clean 0.0 - 1.0 probabilities
        exp_logits = np.exp(raw_logits - np.max(raw_logits))
        probabilities = exp_logits / np.sum(exp_logits)
        
        # Get Top-5 predictions
        top5_indices = np.argsort(probabilities)[::-1][:5]
        top5 = [
            {
                "rank": rank + 1,
                "class_id": int(idx),
                "label": LABELS[idx] if idx < len(LABELS) else f"Class {idx}",
                "confidence_percent": f"{probabilities[idx] * 100:.2f}%"
            }
            for rank, idx in enumerate(top5_indices)
        ]
        
        return {
            "top_prediction": top5[0],
            "top_5": top5,
            "latency_ms": round(latency_ms, 2),
            "model_precision": MODEL_TYPE,
            "active_hardware_provider": sess.get_providers()[0]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.server:app", host="0.0.0.0", port=8000, reload=True)
