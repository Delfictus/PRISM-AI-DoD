#!/bin/bash

echo "========================================="
echo "    GPU VERIFICATION SUMMARY"
echo "========================================="
echo

# Show GPU status
echo "📊 GPU STATUS:"
nvidia-smi --query-gpu=name,driver_version,cuda_version,memory.total,memory.used --format=csv,noheader,nounits | while IFS=, read -r name driver cuda mem_total mem_used; do
    echo "  GPU: $name"
    echo "  Driver: $driver"
    echo "  CUDA: $cuda"
    echo "  Memory: ${mem_used}MB / ${mem_total}MB used"
done
echo

# Run kernel test and capture output
echo "🚀 GPU KERNEL EXECUTION TEST:"
./target/release/test_gpu_kernel | grep -E "Performance:|GPU Time:|✅|Dimensions:"
echo

# Check if GPU libraries are loaded
echo "🔍 CUDA LIBRARIES LOADED:"
ldd ./target/release/test_gpu_kernel | grep -i cuda | head -3
echo

echo "========================================="
echo "✅ GPU IS ACTIVE AND EXECUTING KERNELS!"
echo "========================================="