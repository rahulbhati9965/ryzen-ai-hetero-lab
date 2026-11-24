# ==============================================================================
# Multi-Stage Production Dockerfile for AI Inference Service
# ==============================================================================
FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install minimal OS dependencies for computer vision & native math
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy configuration, exported models, and application code
COPY configs/ ./configs/
COPY models/ ./models/
COPY src/ ./src/

# Expose HTTP port for API inference
EXPOSE 8000

# Container healthcheck (Kubernetes / Docker probe)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI inference server
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]
