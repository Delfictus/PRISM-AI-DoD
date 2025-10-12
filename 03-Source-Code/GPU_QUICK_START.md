# 🚀 GPU Implementation Quick Start Guide
## Immediate Action Items for RTX 5070 Integration

---

## 🎯 IMMEDIATE NEXT STEPS (Do These First!)

### Step 1: Create GPU Memory Manager (30 mins)
```rust
// Create: src/gpu/memory_manager.rs
use cudarc::driver::{CudaContext, CudaSlice, DevicePtr};
use std::sync::Arc;
use anyhow::Result;

pub struct GpuMemoryPool {
    ctx: Arc<CudaContext>,
    allocated_buffers: Vec<CudaSlice<f32>>,
}

impl GpuMemoryPool {
    pub fn new() -> Result<Self> {
        let ctx = CudaContext::new(0)?;
        Ok(Self {
            ctx: Arc::new(ctx),
            allocated_buffers: Vec::new(),
        })
    }

    pub fn allocate_f32(&mut self, size: usize) -> Result<CudaSlice<f32>> {
        self.ctx.alloc_zeros(size)
    }

    pub fn transfer_to_gpu(&self, data: &[f32]) -> Result<CudaSlice<f32>> {
        self.ctx.htod_sync_copy(data)
    }

    pub fn transfer_from_gpu(&self, gpu_data: &CudaSlice<f32>) -> Result<Vec<f32>> {
        self.ctx.dtoh_sync_copy(gpu_data)
    }
}
```

### Step 2: Create First GPU Kernel (30 mins)
```cuda
// Create: src/kernels/cuda/matrix_ops.cu
extern "C" __global__
void matrix_multiply(
    const float* a,
    const float* b,
    float* c,
    int m, int n, int k
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

extern "C" __global__
void softmax_forward(
    const float* input,
    float* output,
    int batch_size,
    int num_classes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        // Find max for numerical stability
        float max_val = -INFINITY;
        for (int i = 0; i < num_classes; i++) {
            max_val = fmaxf(max_val, input[idx * num_classes + i]);
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            output[idx * num_classes + i] = expf(input[idx * num_classes + i] - max_val);
            sum += output[idx * num_classes + i];
        }

        // Normalize
        for (int i = 0; i < num_classes; i++) {
            output[idx * num_classes + i] /= sum;
        }
    }
}
```

### Step 3: Compile CUDA Kernels (5 mins)
```bash
# Run these commands:
cd src/kernels/cuda
nvcc -ptx matrix_ops.cu -o matrix_ops.ptx --gpu-architecture=sm_89
# Note: sm_89 for Ada Lovelace (RTX 5070)
```

### Step 4: Create GPU-Accelerated Linear Layer (45 mins)
```rust
// Create: src/gpu/layers/linear.rs
use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig};
use std::sync::Arc;
use anyhow::Result;

pub struct GpuLinear {
    ctx: Arc<CudaContext>,
    weight: CudaSlice<f32>,
    bias: CudaSlice<f32>,
    in_features: usize,
    out_features: usize,
    module: Arc<CudaModule>,
}

impl GpuLinear {
    pub fn new(
        ctx: Arc<CudaContext>,
        in_features: usize,
        out_features: usize,
    ) -> Result<Self> {
        // Initialize weights on GPU
        let weight_size = in_features * out_features;
        let weight = ctx.alloc_zeros(weight_size)?;
        let bias = ctx.alloc_zeros(out_features)?;

        // Load PTX module
        let ptx = std::fs::read("src/kernels/cuda/matrix_ops.ptx")?;
        let module = ctx.load_ptx(ptx, "matrix_ops", &["matrix_multiply"])?;

        Ok(Self {
            ctx,
            weight,
            bias,
            in_features,
            out_features,
            module: Arc::new(module),
        })
    }

    pub fn forward(&self, input: &CudaSlice<f32>, batch_size: usize) -> Result<CudaSlice<f32>> {
        let output_size = batch_size * self.out_features;
        let output = self.ctx.alloc_zeros(output_size)?;

        // Launch kernel
        let grid = (
            (self.out_features + 15) / 16,
            (batch_size + 15) / 16,
            1
        );
        let block = (16, 16, 1);

        unsafe {
            self.module.launch(
                "matrix_multiply",
                grid,
                block,
                &[
                    &input.as_device_ptr(),
                    &self.weight.as_device_ptr(),
                    &output.as_device_ptr(),
                    &batch_size,
                    &self.out_features,
                    &self.in_features,
                ],
            )?;
        }

        // Add bias (implement bias kernel similarly)

        Ok(output)
    }
}
```

---

## 📋 TESTING CHECKLIST

### 1. Test GPU Memory Manager
```rust
#[test]
fn test_gpu_memory() {
    let mut pool = GpuMemoryPool::new().unwrap();

    // Test allocation
    let gpu_mem = pool.allocate_f32(1024).unwrap();

    // Test transfer
    let data = vec![1.0_f32; 1024];
    let gpu_data = pool.transfer_to_gpu(&data).unwrap();
    let result = pool.transfer_from_gpu(&gpu_data).unwrap();

    assert_eq!(result, data);
}
```

### 2. Benchmark GPU vs CPU
```rust
fn benchmark_linear_layer() {
    let batch_size = 64;
    let in_features = 1024;
    let out_features = 512;

    // CPU timing
    let start = Instant::now();
    cpu_linear_forward(/* ... */);
    let cpu_time = start.elapsed();

    // GPU timing
    let start = Instant::now();
    gpu_linear_forward(/* ... */);
    let gpu_time = start.elapsed();

    println!("CPU: {:?}, GPU: {:?}, Speedup: {:.2}x",
        cpu_time, gpu_time, cpu_time.as_secs_f32() / gpu_time.as_secs_f32());
}
```

---

## 🔥 PRIORITY MODULES TO GPU-ACCELERATE

### 1. PWSA Active Inference (Highest Impact)
**File**: `src/pwsa/active_inference_classifier.rs`
- [ ] Replace CPU Linear with GpuLinear
- [ ] Replace CPU softmax with GPU kernel
- [ ] Add batch processing

### 2. Transfer Entropy (High Compute)
**File**: `src/cma/transfer_entropy_gpu.rs`
- [ ] Port k-NN search to GPU
- [ ] Parallel entropy calculation
- [ ] Batch multiple time series

### 3. Neural Quantum State (Heavy Compute)
**File**: `src/cma/neural/neural_quantum.rs`
- [ ] Monte Carlo sampling on GPU
- [ ] ResNet forward pass
- [ ] Parallel Metropolis-Hastings

---

## ⚡ OPTIMIZATION TIPS FOR RTX 5070

### 1. Use Tensor Cores
```cuda
// Use wmma API for matrix multiply
#include <mma.h>
using namespace nvcuda;

// In kernel:
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
```

### 2. Optimize Memory Access
- **Coalesced access**: Threads access consecutive memory
- **Shared memory**: Use for frequently accessed data
- **Constant memory**: Use for read-only parameters

### 3. Maximize Occupancy
```bash
# Use nvcc occupancy calculator
nvcc --ptxas-options=-v matrix_ops.cu
# Look for: "Used X registers, Y bytes smem"
```

---

## 🚧 COMMON ISSUES & SOLUTIONS

### Issue 1: "Out of memory"
```rust
// Solution: Free unused buffers
ctx.synchronize()?;  // Ensure kernels complete
drop(old_buffer);    // Explicitly free
```

### Issue 2: "Kernel launch failed"
```rust
// Check last error
if let Err(e) = ctx.synchronize() {
    println!("CUDA error: {:?}", e);
}
```

### Issue 3: "Wrong results"
```cuda
// Add bounds checking in kernels
if (threadIdx.x >= size) return;
```

---

## 📊 EXPECTED PERFORMANCE GAINS

| Module | CPU Time | GPU Time | Speedup |
|--------|----------|----------|---------|
| Linear Layer (1024→512) | 2.5ms | 0.1ms | 25x |
| Softmax (64×1000) | 1.2ms | 0.05ms | 24x |
| Transfer Entropy | 500ms | 10ms | 50x |
| Neural Quantum State | 1000ms | 20ms | 50x |
| Full Forward Pass | 100ms | 5ms | 20x |

---

## 🎯 TODAY'S GOALS

1. **Hour 1**: Create GPU memory manager
2. **Hour 2**: Write and compile first CUDA kernel
3. **Hour 3**: Integrate with PWSA classifier
4. **Hour 4**: Benchmark and validate results
5. **Hour 5**: Document and plan next module

---

## ✅ VALIDATION SCRIPT

```rust
// Create: examples/validate_gpu_speedup.rs
fn main() {
    println!("=== GPU Acceleration Validation ===\n");

    // 1. Test memory transfer
    test_memory_transfer();

    // 2. Test matrix multiply
    test_matrix_multiply();

    // 3. Test neural network layer
    test_neural_layer();

    // 4. Compare accuracies
    validate_numerical_accuracy();

    println!("\n✅ All GPU tests passed!");
}
```

---

**START HERE**: Implement the GPU Memory Manager first, then proceed with the Linear layer GPU kernel. This will give you immediate, measurable GPU acceleration!