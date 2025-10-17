# GPU Troubleshooting Guide

**Author**: Worker 2 (GPU Infrastructure)
**Date**: 2025-10-13
**Status**: Production Ready

## Overview

This guide provides comprehensive troubleshooting for common GPU issues in PRISM-AI. It covers initialization failures, runtime errors, performance degradation, and integration problems.

## Table of Contents

1. [Quick Diagnostic Steps](#quick-diagnostic-steps)
2. [Initialization Failures](#initialization-failures)
3. [Runtime Errors](#runtime-errors)
4. [Performance Issues](#performance-issues)
5. [Integration Problems](#integration-problems)
6. [Advanced Debugging](#advanced-debugging)

---

## Quick Diagnostic Steps

### Step 1: Verify CUDA Installation

```bash
# Check NVIDIA driver
nvidia-smi

# Expected output:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 535.104.05   Driver Version: 535.104.05   CUDA Version: 12.2     |
# +-----------------------------------------------------------------------------+

# Check nvcc compiler
nvcc --version

# Expected output:
# nvcc: NVIDIA (R) Cuda compiler driver
# Cuda compilation tools, release 12.2, V12.2.140
```

**If nvidia-smi fails**:
- Driver not installed: `sudo apt install nvidia-driver-535`
- Driver crashed: `sudo nvidia-smi -r` (reset GPU)
- GPU not detected: Check PCIe connection, BIOS settings

**If nvcc not found**:
- CUDA toolkit not installed: `sudo apt install cuda-toolkit-12-2`
- PATH not set: Add to `~/.bashrc`: `export PATH=/usr/local/cuda/bin:$PATH`

### Step 2: Verify Rust Build

```bash
# Clean build
cargo clean
cargo build --features cuda --release

# Expected output:
# Compiling prism-ai v0.1.0 (/path/to/PRISM-Worker-2/src-new)
# Finished release [optimized] target(s) in X.XXs
```

**If compilation fails**:
- See [Initialization Failures](#initialization-failures)

### Step 3: Run Smoke Tests

```bash
# Basic functionality test
cargo test --test gpu_kernel_smoke_test --features cuda -- --ignored

# Expected output:
# running 6 tests
# test test_smoke_ar_forecast ... ok
# test test_smoke_lstm_cell ... ok
# ...
# test result: ok. 6 passed; 0 failed
```

**If tests fail**:
- See [Runtime Errors](#runtime-errors)

---

## Initialization Failures

### Issue 1: `failed to get global executor`

**Error Message**:
```
Error: failed to get global executor
Caused by: Failed to initialize CUDA device
```

**Symptoms**:
- Occurs on first GPU operation
- Application cannot start
- `nvidia-smi` shows GPU available

**Root Causes**:
1. GPU already in use by another process
2. Insufficient GPU memory
3. cudarc initialization failure

**Diagnosis**:

```bash
# Check GPU usage
nvidia-smi

# Look for other processes using GPU
# If column "GPU-Memory-Usage" shows high usage:
# - Identify process with PID in "Processes" section
# - Kill process: kill <PID>
# - Or wait for process to complete

# Check available memory
nvidia-smi --query-gpu=memory.free --format=csv

# Minimum required: 512 MB
# Recommended: 2 GB+
```

**Solutions**:

**Solution 1**: Free GPU memory
```bash
# Kill other GPU processes
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill

# Restart application
cargo run --example gpu_production_profiler --features cuda
```

**Solution 2**: Reduce memory allocation
```rust
// In src/gpu/kernel_executor.rs
// Reduce default buffer sizes
const DEFAULT_BUFFER_SIZE: usize = 64 * 1024 * 1024; // 64 MB instead of 512 MB
```

**Solution 3**: Use different GPU (multi-GPU systems)
```bash
# Select GPU 1 instead of GPU 0
export CUDA_VISIBLE_DEVICES=1

# Run application
cargo run --example gpu_production_profiler --features cuda
```

### Issue 2: `error: linking with cc failed`

**Error Message**:
```
error: linking with `cc` failed: exit status: 1
  = note: /usr/bin/ld: cannot find -lcudart
```

**Symptoms**:
- Compilation fails at link stage
- Missing CUDA runtime library

**Root Causes**:
- CUDA toolkit not installed
- LD_LIBRARY_PATH not set

**Solutions**:

```bash
# Install CUDA toolkit
sudo apt install cuda-toolkit-12-2

# Add to ~/.bashrc
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH

# Reload
source ~/.bashrc

# Verify
ldconfig -p | grep cuda
```

### Issue 3: `PTX file not found`

**Error Message**:
```
Error: PTX file not found: /path/to/target/debug/build/.../tensor_core_matmul.ptx
```

**Symptoms**:
- Occurs when using Tensor Core kernels
- Build succeeds but runtime fails

**Root Causes**:
- build.rs didn't compile CUDA kernels
- nvcc not found
- Incorrect compute capability

**Solutions**:

```bash
# Force rebuild with verbose output
cargo clean
cargo build --features cuda --verbose 2>&1 | grep -i "nvcc\|ptx\|cuda"

# Check build.rs output
# Expected:
# Compiling CUDA kernels with nvcc: /usr/local/cuda/bin/nvcc
# Detected Compute 12.0, using sm_90
# Successfully compiled Tensor Core kernels to PTX

# If nvcc not found:
export PATH=/usr/local/cuda/bin:$PATH
cargo build --features cuda

# If wrong compute capability:
# Edit src-new/build.rs
# Change: let sm_arch = "sm_90"; // to your GPU's compute capability
# RTX 4090: sm_90 (Compute 12.0 → sm_90)
# RTX 3090: sm_86 (Compute 8.6 → sm_86)
# RTX 2080: sm_75 (Compute 7.5 → sm_75)
```

---

## Runtime Errors

### Issue 1: `kernel launch failed`

**Error Message**:
```
Error: kernel launch failed
Caused by: CUDA error: invalid configuration argument
```

**Symptoms**:
- Specific kernel fails
- Other kernels work fine
- Occurs with large inputs

**Root Causes**:
1. Block/grid size exceeds GPU limits
2. Shared memory too large
3. Register count too high

**Diagnosis**:

```bash
# Check GPU limits
nvidia-smi --query-gpu=compute_cap --format=csv

# Get detailed GPU info
nvidia-smi -q

# Look for:
# - Max threads per block: 1024
# - Max blocks per grid: 2147483647
# - Max shared memory per block: 48 KB (sm_90)
```

**Solutions**:

**Solution 1**: Reduce block size
```rust
// In kernel call, reduce threads_per_block
// BEFORE:
let threads_per_block = 1024;

// AFTER:
let threads_per_block = 256; // More conservative
```

**Solution 2**: Enable auto-tuning
```rust
use prism_ai::gpu::kernel_autotuner::GpuKernelAutotuner;

let mut tuner = GpuKernelAutotuner::new();

// Auto-tune problematic kernel
let config = tuner.tune_kernel("pixel_entropy", |cfg| {
    executor.pixel_entropy_with_config(&image, h, w, window, cfg)
}, data_size)?;

println!("Optimal config: threads={}, blocks={}",
         config.threads_per_block, config.blocks_per_grid);
```

### Issue 2: `out of memory`

**Error Message**:
```
Error: out of memory
Caused by: CUDA error: out of memory
```

**Symptoms**:
- Occurs with large workloads
- GPU memory usage at 100%
- Inconsistent failure (works sometimes)

**Root Causes**:
1. Allocation too large for GPU
2. Memory leak (allocations not freed)
3. Fragmentation

**Diagnosis**:

```bash
# Monitor GPU memory during execution
watch -n 0.5 nvidia-smi

# Check memory usage programmatically
```

```rust
use prism_ai::gpu::gpu_monitoring::get_global_monitor;

let monitor = get_global_monitor()?;
let report = monitor.lock().unwrap().get_report();
println!("{}", report);

// Look for:
// - Current Memory: X MB
// - Peak Memory: Y MB
// - If Peak > 80% of total: memory pressure
```

**Solutions**:

**Solution 1**: Enable memory pooling
```rust
use prism_ai::gpu::active_memory_pool::ActiveMemoryPool;

let pool = ActiveMemoryPool::with_defaults();

// Wrap allocations
let id = pool.register_allocation(size_bytes);
// ... use buffer ...
pool.register_deallocation(id);

// Check if pooling helps
let stats = pool.get_stats();
println!("Pool hit rate: {:.1}%", stats.hit_rate());
// If hit rate >50%, pooling is helping
```

**Solution 2**: Process in batches
```rust
// BEFORE: Process entire 1024x1024 image at once
let entropy = executor.pixel_entropy(&large_image, 1024, 1024, 16)?;

// AFTER: Process in 512x512 tiles
let mut entropy_full = vec![0.0; 1024 * 1024];
for y_tile in 0..2 {
    for x_tile in 0..2 {
        let tile = extract_tile(&large_image, x_tile, y_tile, 512, 512);
        let tile_entropy = executor.pixel_entropy(&tile, 512, 512, 16)?;
        merge_tile(&mut entropy_full, &tile_entropy, x_tile, y_tile);
    }
}
```

**Solution 3**: Reduce precision
```rust
// Use FP16 instead of FP32 (halves memory)
let fp16_data = executor.fp32_to_fp16(&fp32_data)?;
let fp16_result = executor.matrix_multiply_fp16(&fp16_data, ...)?;
let fp32_result = executor.fp16_to_fp32(&fp16_result)?;
```

### Issue 3: `computation took too long`

**Error Message**:
```
Error: computation took too long
Caused by: CUDA error: device kernel timeout
```

**Symptoms**:
- Long-running kernels timeout
- Display driver resets
- System freezes momentarily

**Root Causes**:
1. Kernel too slow (>2 seconds on display GPU)
2. Infinite loop in kernel
3. Display driver watchdog

**Solutions**:

**Solution 1**: Use dedicated compute GPU (not display GPU)
```bash
# Identify GPUs
nvidia-smi -L

# GPU 0: NVIDIA GeForce RTX 4090 (Display)
# GPU 1: NVIDIA GeForce RTX 3090 (Compute)

# Use compute GPU
export CUDA_VISIBLE_DEVICES=1
```

**Solution 2**: Increase watchdog timeout (Linux)
```bash
# Edit /etc/modprobe.d/nvidia.conf
sudo nano /etc/modprobe.d/nvidia.conf

# Add:
options nvidia NVreg_RegistryDwords="RM_GSP_ENABLE_TIMER_CALLBACK=0"

# Reboot
sudo reboot
```

**Solution 3**: Break kernel into smaller chunks
```rust
// BEFORE: One large kernel call
let result = executor.pixel_entropy(&image, 1024, 1024, 16)?;

// AFTER: Multiple smaller calls
let chunk_size = 256;
for i in 0..(1024 / chunk_size) {
    let chunk = &image[i * chunk_size * 1024..(i + 1) * chunk_size * 1024];
    let chunk_result = executor.pixel_entropy(chunk, chunk_size, 1024, 16)?;
    // ... merge results ...
}
```

---

## Performance Issues

### Issue 1: `kernel too slow`

**Symptom**: Kernel takes >10× expected time

**Diagnosis**:

Run profiler:
```bash
cargo run --example gpu_production_profiler --features cuda
```

Look for:
- Mean time >>baseline (see GPU_PERFORMANCE_PROFILING_GUIDE.md)
- High coefficient of variation (>20%)

**Solutions**:

See [GPU_PERFORMANCE_PROFILING_GUIDE.md](GPU_PERFORMANCE_PROFILING_GUIDE.md) for detailed optimization strategies.

**Quick fixes**:
1. Enable memory pooling (15-35% speedup)
2. Enable auto-tuning (5-20% speedup)
3. Reduce data size if possible

### Issue 2: `GPU underutilized`

**Symptom**: nvidia-smi shows <50% GPU utilization

**Diagnosis**:

```bash
# Monitor GPU utilization
nvidia-smi dmon -s u

# Expected: GPU >70% for compute-heavy workloads
# If <50%: GPU is waiting (CPU bottleneck or memory transfer)
```

**Root Causes**:
1. CPU bottleneck (data preparation)
2. PCIe transfer bottleneck
3. Small kernels (launch overhead dominates)

**Solutions**:

**Solution 1**: Batch operations
```rust
// BEFORE: 1000 small kernel launches
for i in 0..1000 {
    let result = executor.vector_add(&a[i], &b[i])?;
}

// AFTER: 1 large batched kernel
let results = executor.vector_add_batch(&a_batch, &b_batch)?;
```

**Solution 2**: Overlap transfers and compute (future enhancement)
```rust
// Requires async support (not yet implemented)
// This is the target architecture:
let stream = CudaStream::new()?;
stream.copy_to_device_async(&host_data)?;
stream.launch_kernel_async("pixel_entropy", &args)?;
stream.copy_to_host_async(&result)?;
stream.synchronize()?;
```

### Issue 3: `high variability`

**Symptom**: Kernel time varies >20% between runs

**Diagnosis**:

```bash
# Check GPU temperature
nvidia-smi --query-gpu=temperature.gpu --format=csv -l 1

# Thermal throttling if >83°C (RTX 4090)
```

```bash
# Check GPU clock
nvidia-smi --query-gpu=clocks.current.graphics --format=csv -l 1

# Clock drops indicate throttling
```

**Solutions**:

**Solution 1**: Improve cooling
- Check case airflow
- Clean dust filters
- Increase fan speed: `nvidia-smi -i 0 -pm 1` (persistence mode)

**Solution 2**: Reduce power limit
```bash
# Reduce power to avoid thermal throttling
# RTX 4090: 450W max, reduce to 400W for stability
sudo nvidia-smi -i 0 -pl 400
```

**Solution 3**: Use auto-tuning to adapt
```rust
// Auto-tuner adjusts to current GPU state
let tuner = GpuKernelAutotuner::new();
tuner.set_retune_interval(100); // Re-tune every 100 operations
```

---

## Integration Problems

### Issue 1: `method not found on KernelExecutor`

**Error Message**:
```
error[E0599]: no method named `pixel_entropy` found for struct `KernelExecutor`
```

**Symptoms**:
- Compilation fails
- Method clearly exists in documentation

**Root Causes**:
1. Missing `--features cuda` flag
2. Outdated dependency
3. Wrong import

**Solutions**:

**Solution 1**: Add cuda feature
```bash
# BEFORE:
cargo build

# AFTER:
cargo build --features cuda
```

**Solution 2**: Update dependencies
```bash
cargo update
cargo build --features cuda
```

**Solution 3**: Check import
```rust
// Correct import
use prism_ai::gpu::kernel_executor::get_global_executor;

let executor = get_global_executor()?;
let executor = executor.lock().unwrap();

executor.pixel_entropy(&image, h, w, window)?; // ✅ Works
```

### Issue 2: `borrow checker errors with executor`

**Error Message**:
```
error[E0502]: cannot borrow `executor` as mutable because it is also borrowed as immutable
```

**Symptoms**:
- Executor wrapped in Arc<Mutex<>>
- Borrow checker rejects valid-looking code

**Solutions**:

**Solution 1**: Lock once, use multiple times
```rust
// BEFORE: Multiple locks
let result1 = executor.lock().unwrap().vector_add(&a, &b)?; // Lock 1
let result2 = executor.lock().unwrap().vector_add(&c, &d)?; // Lock 2

// AFTER: Single lock
let executor = executor.lock().unwrap();
let result1 = executor.vector_add(&a, &b)?;
let result2 = executor.vector_add(&c, &d)?;
```

**Solution 2**: Limit lock scope
```rust
// BEFORE: Lock held too long
let executor = executor.lock().unwrap();
let result1 = executor.vector_add(&a, &b)?;
// ... lots of CPU work ...
let result2 = executor.vector_add(&c, &d)?; // Lock still held

// AFTER: Release lock between operations
{
    let executor = executor.lock().unwrap();
    let result1 = executor.vector_add(&a, &b)?;
} // Lock released

// ... CPU work ...

{
    let executor = executor.lock().unwrap();
    let result2 = executor.vector_add(&c, &d)?;
} // Lock released
```

### Issue 3: `test failures with --ignored`

**Error Message**:
```
test test_worker1_ar_forecasting_realistic ... FAILED
```

**Symptoms**:
- Integration tests fail
- Smoke tests pass
- Only occurs with `--ignored` tests

**Root Causes**:
1. GPU not available in CI environment
2. Test data too large
3. Timing assumptions incorrect

**Solutions**:

**Solution 1**: Run tests on GPU-enabled machine
```bash
# Skip --ignored tests in CI (no GPU)
cargo test --features cuda

# Run --ignored tests on development machine (has GPU)
cargo test --test cross_worker_integration --features cuda -- --ignored
```

**Solution 2**: Reduce test data size
```rust
// In tests/cross_worker_integration.rs
// BEFORE:
let height = 512;
let width = 512;

// AFTER: Smaller for faster tests
let height = 128;
let width = 128;
```

---

## Advanced Debugging

### Using CUDA-MEMCHECK

Detect memory errors:

```bash
# Run application with memory checker
cuda-memcheck cargo run --example gpu_production_profiler --features cuda

# Look for:
# - Invalid memory access
# - Uninitialized memory read
# - Memory leaks
```

### Using Nsight Systems

Profile GPU timeline:

```bash
# Install Nsight Systems
sudo apt install nsight-systems

# Profile application
nsys profile --stats=true cargo run --example gpu_production_profiler --features cuda

# View report
nsys-ui report1.nsys-rep

# Look for:
# - Kernel launch overhead
# - Memory transfer bottlenecks
# - CPU-GPU sync points
```

### Using Nsight Compute

Profile kernel performance:

```bash
# Install Nsight Compute
sudo apt install nsight-compute

# Profile specific kernel
ncu --target-processes all cargo run --example gpu_production_profiler --features cuda

# View report
ncu-ui report1.ncu-rep

# Analyze:
# - Occupancy
# - Memory throughput
# - Instruction throughput
```

### Enabling Rust Backtrace

```bash
# Full backtrace
RUST_BACKTRACE=full cargo run --example gpu_production_profiler --features cuda

# Look for:
# - Stack trace showing error origin
# - Function call chain
```

---

## Common Error Messages Reference

| Error | Likely Cause | Solution |
|-------|-------------|----------|
| `failed to get global executor` | GPU busy or no memory | Free GPU memory, use different GPU |
| `cannot find -lcudart` | CUDA not installed | Install CUDA toolkit |
| `PTX file not found` | build.rs failed | Check nvcc, verify PATH |
| `kernel launch failed` | Invalid config | Reduce block size, enable auto-tuning |
| `out of memory` | Too large allocation | Enable pooling, process in batches |
| `computation took too long` | Kernel timeout | Use compute GPU, break into chunks |
| `no method named X` | Missing cuda feature | Add `--features cuda` |
| `cannot borrow executor` | Borrow checker issue | Lock once, limit scope |

---

## Getting Help

### Check Documentation

1. [GPU_PERFORMANCE_PROFILING_GUIDE.md](GPU_PERFORMANCE_PROFILING_GUIDE.md) - Performance optimization
2. [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) - Full documentation map
3. [GPU_KERNEL_INTEGRATION_GUIDE.md](GPU_KERNEL_INTEGRATION_GUIDE.md) - Integration examples

### Debug Checklist

Before asking for help, verify:

- [ ] `nvidia-smi` shows GPU available
- [ ] `nvcc --version` shows CUDA toolkit installed
- [ ] `cargo build --features cuda` succeeds
- [ ] `cargo test --features cuda` passes (non-ignored tests)
- [ ] GPU memory usage <80% (`nvidia-smi`)
- [ ] GPU temperature <85°C (`nvidia-smi`)
- [ ] Other GPU processes stopped (`nvidia-smi`)

### Collect Diagnostic Info

```bash
# System info
nvidia-smi -q > nvidia_info.txt
nvcc --version > cuda_version.txt
cargo --version > cargo_version.txt

# Build log
cargo clean
cargo build --features cuda --verbose 2>&1 | tee build.log

# Test log
cargo test --features cuda 2>&1 | tee test.log

# Attach these files when requesting help
```

---

**Document Version**: 1.0
**Last Updated**: 2025-10-13
**Maintained By**: Worker 2 (GPU Infrastructure)
