# GPU API Fix Plan - Enabling Actual Kernel Execution

## Executive Summary

Current blocker: cudarc API doesn't expose expected methods for PTX loading and kernel execution. This plan outlines multiple approaches to resolve this and achieve actual GPU computation.

## Current Situation

### What Works ✅
- GPU context creation (`CudaContext::new(0)`)
- PTX compilation (`compile_ptx_with_opts()`)
- GPU detection and enablement
- Basic memory allocation via streams

### What's Blocked ❌
- PTX module loading (`load_ptx` not found)
- Kernel function retrieval (`get_func` not found)
- Direct memory operations (`htod_sync_copy` not found)
- Kernel launching

## Phase 1: Investigation & Discovery (2-4 hours)

### Task 1.1: Deep Dive into cudarc Source
```bash
# Clone and examine cudarc repository
git clone https://github.com/coreylowman/cudarc.git
cd cudarc

# Search for PTX-related functionality
grep -r "load_ptx\|module\|kernel\|launch" --include="*.rs"
grep -r "CudaFunction\|CudaModule" --include="*.rs"

# Check examples
find examples -name "*.rs" -exec grep -l "ptx\|kernel\|launch" {} \;
```

**Expected Findings:**
- Hidden module loading API
- Different naming conventions
- Required feature flags

### Task 1.2: Analyze cudarc Tests
```bash
# Check test files for API usage patterns
find tests -name "*.rs" -exec grep -l "compile_ptx\|launch" {} \;
cargo test --features cuda --doc
```

### Task 1.3: Check cudarc Dependencies
```toml
# Examine Cargo.toml for sub-features
[dependencies.cudarc]
features = ["cuda-13000", "driver", "nvrtc", "cublas", "curand", "f16", "cufft", "nccl"]
```

## Phase 2: API Mapping (4-6 hours)

### Task 2.1: Create API Translation Layer

```rust
// src/gpu/cudarc_wrapper.rs

use cudarc::driver::{CudaContext, CudaStream, CudaSlice};
use cudarc::nvrtc::{compile_ptx_with_opts, Ptx};
use std::sync::Arc;

pub struct GpuDevice {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    modules: HashMap<String, CudaModule>,
}

impl GpuDevice {
    pub fn new(device_id: usize) -> Result<Self> {
        let context = CudaContext::new(device_id)?;
        let stream = context.stream()?;

        Ok(Self {
            context,
            stream,
            modules: HashMap::new(),
        })
    }

    // Wrapper for memory operations
    pub fn upload<T>(&self, data: &[T]) -> Result<CudaSlice<T>> {
        self.stream.htod_copy(data)
    }

    pub fn download<T>(&self, buffer: &CudaSlice<T>) -> Result<Vec<T>> {
        self.stream.dtoh_copy(buffer)
    }
}
```

### Task 2.2: Find Actual Method Names

Based on neuromorphic module patterns:
```rust
// What we see working:
stream.memcpy_htod()  // Host to device
stream.memcpy_dtoh()  // Device to host
stream.alloc_zeros()  // Allocation

// Likely kernel API:
context.load_module()  // Or similar
module.get_function()  // Or similar
function.launch()      // Or similar
```

## Phase 3: Implementation Options (8-12 hours)

### Option A: Direct FFI Approach

```rust
// src/gpu/cuda_ffi.rs

use std::ffi::{CString, c_void};
use std::ptr;

#[link(name = "cuda")]
extern "C" {
    fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;
    fn cuModuleGetFunction(func: *mut CUfunction, module: CUmodule, name: *const c_char) -> CUresult;
    fn cuLaunchKernel(
        f: CUfunction,
        gridDimX: u32, gridDimY: u32, gridDimZ: u32,
        blockDimX: u32, blockDimY: u32, blockDimZ: u32,
        sharedMemBytes: u32,
        stream: CUstream,
        kernelParams: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult;
}

pub struct CudaModule {
    handle: CUmodule,
}

impl CudaModule {
    pub fn load_ptx(ptx: &str) -> Result<Self> {
        let mut module = ptr::null_mut();
        let ptx_cstr = CString::new(ptx)?;

        unsafe {
            check_cuda(cuModuleLoadData(&mut module, ptx_cstr.as_ptr() as *const c_void))?;
        }

        Ok(Self { handle: module })
    }

    pub fn get_function(&self, name: &str) -> Result<CudaFunction> {
        let mut func = ptr::null_mut();
        let name_cstr = CString::new(name)?;

        unsafe {
            check_cuda(cuModuleGetFunction(&mut func, self.handle, name_cstr.as_ptr()))?;
        }

        Ok(CudaFunction { handle: func })
    }
}
```

### Option B: Use cust Crate Instead

```toml
[dependencies]
cust = "0.3"
```

```rust
use cust::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA
    cust::init()?;

    // Create context
    let device = Device::get_device(0)?;
    let _ctx = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

    // Load PTX
    let ptx = CString::new(include_str!("kernel.ptx"))?;
    let module = Module::load_from_string(&ptx)?;
    let kernel = module.get_function("my_kernel")?;

    // Launch kernel
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    unsafe {
        launch!(kernel<<<grid, block, 0, stream>>>(params))?;
    }

    Ok(())
}
```

### Option C: Use RustaCUDA

```toml
[dependencies]
rustacuda = "0.1"
rustacuda_core = "0.1"
```

```rust
use rustacuda::prelude::*;
use rustacuda::memory::DeviceBox;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize
    rustacuda::init(CudaFlags::empty())?;

    let device = Device::get_device(0)?;
    let _context = Context::create_and_push(
        ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO,
        device,
    )?;

    // Compile and load
    let ptx = compile_kernel()?;
    let module = Module::load_from_string(&ptx)?;
    let kernel = module.get_function(&CString::new("kernel_name")?)?;

    // Execute
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    unsafe {
        launch!(kernel<<<1, 1, 0, stream>>>(args))?;
    }

    Ok(())
}
```

## Phase 4: Hybrid Solution (Recommended) - 12-16 hours

### Step 1: Keep cudarc for What Works
```rust
// Memory management and basic operations
use cudarc::driver::CudaContext;
use cudarc::nvrtc::compile_ptx_with_opts;

// Keep using cudarc for:
// - Context creation
// - PTX compilation
// - Memory allocation via streams
```

### Step 2: Add FFI for Kernel Execution
```rust
// src/gpu/kernel_launcher_ffi.rs

pub struct KernelLauncher {
    context: Arc<CudaContext>,
    modules: HashMap<String, CudaModuleFFI>,
}

impl KernelLauncher {
    pub fn load_and_launch_ptx(
        &mut self,
        ptx_code: &str,
        kernel_name: &str,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        params: &[*mut c_void],
    ) -> Result<()> {
        // Use FFI to load PTX
        let module = CudaModuleFFI::load_ptx(ptx_code)?;
        let kernel = module.get_function(kernel_name)?;

        // Launch using FFI
        kernel.launch(grid, block, params)?;

        Ok(())
    }
}
```

### Step 3: Create High-Level API
```rust
// src/gpu/gpu_compute.rs

pub struct GpuCompute {
    device: GpuDevice,
    launcher: KernelLauncher,
    kernels: HashMap<String, CompiledKernel>,
}

impl GpuCompute {
    pub fn new() -> Result<Self> {
        let device = GpuDevice::new(0)?;
        let launcher = KernelLauncher::new(&device)?;

        Ok(Self {
            device,
            launcher,
            kernels: HashMap::new(),
        })
    }

    pub fn compile_kernel(&mut self, name: &str, code: &str) -> Result<()> {
        let ptx = compile_ptx_with_opts(code, Default::default())?;
        self.kernels.insert(name.to_string(), ptx);
        Ok(())
    }

    pub fn run_kernel(
        &mut self,
        name: &str,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: Vec<GpuArg>,
    ) -> Result<()> {
        let ptx = self.kernels.get(name)
            .ok_or_else(|| anyhow!("Kernel not found"))?;

        self.launcher.load_and_launch_ptx(ptx, name, grid, block, &args)
    }
}
```

## Phase 5: Testing & Validation (4-6 hours)

### Test 1: Basic Kernel Execution
```rust
#[test]
fn test_vector_add() {
    let mut gpu = GpuCompute::new().unwrap();

    let kernel_code = r#"
    extern "C" __global__ void vector_add(float* a, float* b, float* c, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            c[idx] = a[idx] + b[idx];
        }
    }
    "#;

    gpu.compile_kernel("vector_add", kernel_code).unwrap();

    // Prepare data
    let a = vec![1.0f32; 1024];
    let b = vec![2.0f32; 1024];
    let mut c = vec![0.0f32; 1024];

    // Upload to GPU
    let a_gpu = gpu.device.upload(&a).unwrap();
    let b_gpu = gpu.device.upload(&b).unwrap();
    let c_gpu = gpu.device.allocate(1024).unwrap();

    // Run kernel
    gpu.run_kernel(
        "vector_add",
        (4, 1, 1),
        (256, 1, 1),
        vec![a_gpu, b_gpu, c_gpu, 1024],
    ).unwrap();

    // Download result
    gpu.device.download(&c_gpu, &mut c).unwrap();

    // Verify
    assert!(c.iter().all(|&x| x == 3.0));
}
```

### Test 2: Matrix Multiplication
```rust
#[test]
fn test_matmul_gpu() {
    // Test actual GPU matrix multiplication
    let result = benchmark_matmul_gpu_vs_cpu(1024);
    assert!(result.gpu_speedup > 10.0);
}
```

## Phase 6: Integration (6-8 hours)

### Update Existing Modules
```rust
// src/gpu/simple_gpu.rs
impl SimpleGpuTensor {
    pub fn matmul(&self, other: &SimpleGpuTensor) -> Result<SimpleGpuTensor> {
        // Replace CPU fallback with:
        GPU_COMPUTE.run_kernel("matmul", grid, block, args)?;
    }
}
```

### Update PWSA Classifier
```rust
// src/pwsa/gpu_classifier.rs
impl GpuActiveInferenceClassifier {
    fn forward_gpu(&self, input: &GpuTensor) -> Result<GpuTensor> {
        // Use actual GPU kernels
        GPU_COMPUTE.run_kernel("neural_forward", ...)?;
    }
}
```

## Timeline & Priority

| Phase | Priority | Time | Outcome |
|-------|---------|------|---------|
| 1. Investigation | HIGH | 2-4h | Understand actual API |
| 2. API Mapping | HIGH | 4-6h | Create wrapper design |
| 3. Implementation | CRITICAL | 8-12h | Working kernel execution |
| 4. Hybrid Solution | HIGH | 12-16h | Production-ready API |
| 5. Testing | HIGH | 4-6h | Validated GPU ops |
| 6. Integration | MEDIUM | 6-8h | System-wide GPU |

**Total Estimated Time: 36-52 hours**

## Success Metrics

1. **Kernel Execution**: Successfully launch custom CUDA kernels
2. **Performance**: >10x speedup on matrix operations
3. **Memory**: Efficient host-device transfers
4. **Integration**: All modules use GPU kernels
5. **Reliability**: Automatic fallback on errors

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| cudarc API incomplete | Use FFI or alternative crate |
| CUDA 13 compatibility | Test with CUDA 12 fallback |
| Kernel compilation fails | Pre-compile to .cubin |
| Performance regression | Keep CPU fallback path |

## Immediate Next Steps

1. **Hour 1-2**: Clone cudarc repo and search for actual API
2. **Hour 3-4**: Test cust or rustacuda as alternatives
3. **Hour 5-8**: Implement minimal FFI for kernel launch
4. **Hour 9-12**: Create working vector_add example
5. **Hour 13-16**: Integrate into one module for testing

## Fallback Plan

If all else fails:
1. Use cuBLAS for matrix operations (proven to work)
2. Use cuDNN for neural networks
3. Pre-compile kernels to .cubin files
4. Use PyTorch C++ API with Rust bindings

---

*Plan Created: 2025-10-11*
*Priority: CRITICAL*
*Blocking: Full GPU acceleration*