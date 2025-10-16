#!/bin/bash

echo "=== PRISM-AI GPU vs CPU Performance Benchmark ==="
echo "Date: $(date)"
echo "GPU: RTX 5070 Laptop GPU"
echo ""

# Test 1: Matrix Multiplication Performance
echo "1. Matrix Multiplication (1024x1024) Performance Test"
echo "---------------------------------------------------"

# GPU Test
echo -n "GPU: "
time cargo run --release --features cuda --example matrix_multiply_gpu 2>/dev/null || echo "N/A - example not found"

# CPU Test
echo -n "CPU: "
time cargo run --release --example matrix_multiply_cpu 2>/dev/null || echo "N/A - example not found"

echo ""

# Test 2: Transfer Entropy Computation
echo "2. Transfer Entropy Computation Performance Test"
echo "------------------------------------------------"

# GPU Test
echo "GPU Test:"
cargo test --release --features cuda test_transfer_entropy_gpu --lib 2>&1 | grep -E "test .* ok" | head -5

# CPU Test
echo "CPU Test:"
cargo test --release test_transfer_entropy --lib 2>&1 | grep -E "test .* ok" | head -5

echo ""

# Test 3: Quantum Simulation
echo "3. Quantum Simulation Performance Test"
echo "--------------------------------------"

# GPU Test
echo "GPU Test:"
cargo test --release --features cuda test_quantum_evolution --lib 2>&1 | grep -E "test .* ok|time" | head -5

# CPU Test
echo "CPU Test:"
cargo test --release test_quantum_evolution --lib 2>&1 | grep -E "test .* ok|time" | head -5

echo ""

# Test 4: Neural Network Training
echo "4. Neural Network Training Performance Test"
echo "-------------------------------------------"

# GPU Test
echo "GPU Test:"
cargo test --release --features cuda test_neuromorphic_gpu --lib 2>&1 | grep -E "test .* ok|time" | head -5

# CPU Test
echo "CPU Test:"
cargo test --release test_neuromorphic --lib 2>&1 | grep -E "test .* ok|time" | head -5

echo ""

# Test 5: Overall Test Suite Performance
echo "5. Overall Test Suite Performance"
echo "---------------------------------"

echo "Running GPU test suite (sample)..."
start_gpu=$(date +%s)
cargo test --release --features cuda --lib 2>&1 | tail -3
end_gpu=$(date +%s)
gpu_time=$((end_gpu - start_gpu))
echo "GPU Total Time: ${gpu_time}s"

echo ""
echo "Running CPU test suite (sample)..."
start_cpu=$(date +%s)
cargo test --release --lib 2>&1 | tail -3
end_cpu=$(date +%s)
cpu_time=$((end_cpu - start_cpu))
echo "CPU Total Time: ${cpu_time}s"

echo ""
echo "=== Performance Summary ==="
echo "GPU Test Time: ${gpu_time}s"
echo "CPU Test Time: ${cpu_time}s"

if [ $gpu_time -gt 0 ] && [ $cpu_time -gt 0 ]; then
    speedup=$(echo "scale=2; $cpu_time / $gpu_time" | bc)
    echo "Speedup Factor: ${speedup}x"
fi

echo ""
echo "=== GPU Acceleration Status ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "GPU monitoring not available"