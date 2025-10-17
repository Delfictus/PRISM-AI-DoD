#!/bin/bash
# Build script for GPU runtime library

echo "Building GPU runtime library..."

# Compile the GPU runtime
nvcc --shared -o src/libgpu_runtime.so src/gpu_runtime.cu -arch=sm_70 --compiler-options '-fPIC' 2>&1

if [ $? -eq 0 ]; then
    echo "✅ GPU runtime library built successfully: src/libgpu_runtime.so"

    # Set library path for Rust
    export LD_LIBRARY_PATH=$PWD/src:$LD_LIBRARY_PATH
    export LIBRARY_PATH=$PWD/src:$LIBRARY_PATH

    echo "✅ Library paths configured"

    # Build Rust with GPU support
    echo "Building Rust project with GPU support..."
    cargo build --release --features cuda

    if [ $? -eq 0 ]; then
        echo "✅ Build complete! GPU acceleration is ready."
    else
        echo "❌ Rust build failed"
        exit 1
    fi
else
    echo "❌ Failed to build GPU runtime library"
    exit 1
fi

echo ""
echo "To run with GPU support:"
echo "  export LD_LIBRARY_PATH=$PWD/src:\$LD_LIBRARY_PATH"
echo "  cargo run --release --features cuda"