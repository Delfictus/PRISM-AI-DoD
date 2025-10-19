# PRISM-AI GNN Training Docker Image
# Optimized for RunPod H100 deployment
# Base: NVIDIA CUDA 12.1 + PyTorch 2.1

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Metadata
LABEL maintainer="PRISM-AI Team"
LABEL description="Multi-Task GNN Training for Graph Coloring - World Record Attempt"
LABEL version="1.0"

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch Geometric and dependencies
# Using specific versions for H100 compatibility
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0

# Install PyTorch Geometric
RUN pip install --no-cache-dir \
    torch-geometric==2.4.0

# Install PyG dependencies from wheel
RUN pip install --no-cache-dir \
    pyg_lib \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Install ONNX and other dependencies
RUN pip install --no-cache-dir \
    onnx==1.15.0 \
    onnxruntime-gpu==1.16.3 \
    tensorboard==2.15.1 \
    numpy==1.24.3 \
    scipy==1.11.4 \
    tqdm==4.66.1 \
    matplotlib==3.8.2

# Copy training code
COPY dataset.py model.py train.py export_onnx.py ./
COPY run.sh ./
RUN chmod +x run.sh

# Create directories for outputs
RUN mkdir -p /workspace/checkpoints /workspace/logs /workspace/models

# Expose TensorBoard port
EXPOSE 6006

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Default command (can be overridden)
CMD ["/bin/bash"]

# Usage:
# docker build -t delfictus/prism-ai-world-record:latest .
# docker push delfictus/prism-ai-world-record:latest
#
# On RunPod:
# 1. Select "Custom Container"
# 2. Image: delfictus/prism-ai-world-record:latest
# 3. Upload training_data/ to /workspace/training_data
# 4. Run: bash run.sh
