#!/bin/bash
# Build script for CUDA 13 support

echo "Setting up CUDA 13 environment..."
export CUDA_HOME=/usr/local/cuda-13.0
export CUDA_PATH=/usr/local/cuda-13.0
export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "CUDA Environment:"
echo "  CUDA_HOME=$CUDA_HOME"
echo "  PATH includes CUDA: $(echo $PATH | grep -o cuda-13.0)"
echo "  nvcc location: $(which nvcc)"
echo "  nvcc version:"
nvcc --version | head -1

echo ""
echo "Cleaning build cache..."
cargo clean

echo ""
echo "Building with CUDA 13 support..."
cargo build --features cuda

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build successful with CUDA 13!"
    echo ""
    echo "Rebuilding GPU runtime library..."
    cd src
    nvcc -Xcompiler -fPIC -shared -o libgpu_runtime.so gpu_runtime.cu
    if [ $? -eq 0 ]; then
        echo "✅ GPU runtime library rebuilt"
    else
        echo "❌ Failed to rebuild GPU runtime library"
    fi
    cd ..
else
    echo ""
    echo "❌ Build failed. Check errors above."
fi