#!/bin/bash
# Setup CUDA 13 environment

echo "Setting up CUDA 13.0 environment..."

export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "CUDA Environment configured:"
echo "  CUDA_HOME=$CUDA_HOME"
echo "  PATH includes: $CUDA_HOME/bin"
echo "  LD_LIBRARY_PATH includes: $CUDA_HOME/lib64"
echo ""
echo "NVCC version:"
nvcc --version

echo ""
echo "Add these to your ~/.bashrc to make permanent:"
echo "export CUDA_HOME=/usr/local/cuda-13.0"
echo "export PATH=\$CUDA_HOME/bin:\$PATH"
echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"