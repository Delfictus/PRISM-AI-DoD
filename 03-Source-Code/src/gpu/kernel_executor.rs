//! GPU Kernel Executor with actual kernel execution capabilities
//!
//! This module provides the infrastructure to compile, load and execute
//! actual GPU kernels using the correct cudarc API.

use anyhow::{Result, Context as AnyhowContext};
use cudarc::{
    driver::{CudaContext, LaunchConfig, PushKernelArg, CudaModule, CudaFunction},
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};
use std::collections::HashMap;
use std::sync::Arc;

/// Common GPU kernels used across the system
pub mod kernels {
    pub const VECTOR_ADD: &str = r#"
    extern "C" __global__ void vector_add(float* a, float* b, float* c, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            c[idx] = a[idx] + b[idx];
        }
    }
    "#;

    pub const MATRIX_MUL: &str = r#"
    extern "C" __global__ void matmul(float* a, float* b, float* c, int m, int k, int n) {
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
    "#;

    pub const RELU: &str = r#"
    extern "C" __global__ void relu(float* data, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] = fmaxf(0.0f, data[idx]);
        }
    }
    "#;

    pub const SOFTMAX: &str = r#"
    extern "C" __global__ void softmax(float* data, int batch_size, int num_classes) {
        int batch_idx = blockIdx.x;
        if (batch_idx >= batch_size) return;

        float* row = data + batch_idx * num_classes;

        // Find max for numerical stability
        float max_val = row[0];
        for (int i = 1; i < num_classes; i++) {
            max_val = fmaxf(max_val, row[i]);
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            row[i] = expf(row[i] - max_val);
            sum += row[i];
        }

        // Normalize
        for (int i = 0; i < num_classes; i++) {
            row[i] /= sum;
        }
    }
    "#;

    pub const SIGMOID: &str = r#"
    extern "C" __global__ void sigmoid(float* data, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] = 1.0f / (1.0f + expf(-data[idx]));
        }
    }
    "#;

    pub const TANH: &str = r#"
    extern "C" __global__ void tanh_activation(float* data, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] = tanhf(data[idx]);
        }
    }
    "#;

    pub const BATCH_NORM: &str = r#"
    extern "C" __global__ void batch_norm(
        float* data, float* gamma, float* beta,
        float* mean, float* var,
        int batch_size, int features, float epsilon
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_elements = batch_size * features;

        if (idx < total_elements) {
            int feature_idx = idx % features;
            float normalized = (data[idx] - mean[feature_idx]) /
                              sqrtf(var[feature_idx] + epsilon);
            data[idx] = gamma[feature_idx] * normalized + beta[feature_idx];
        }
    }
    "#;
}

/// GPU Kernel Executor that manages kernel compilation and execution
pub struct GpuKernelExecutor {
    context: Arc<CudaContext>,
    modules: HashMap<String, Arc<CudaModule>>,
    kernels: HashMap<String, Arc<CudaFunction>>,
}

impl GpuKernelExecutor {
    /// Create a new kernel executor
    pub fn new(device_id: usize) -> Result<Self> {
        let context = CudaContext::new(device_id)
            .context("Failed to create CUDA context")?;

        println!("✅ GPU Kernel Executor initialized on device {}", device_id);

        Ok(Self {
            context, // Already Arc<CudaContext>
            modules: HashMap::new(),
            kernels: HashMap::new(),
        })
    }

    /// Compile and register a kernel
    pub fn register_kernel(&mut self, name: &str, code: &str) -> Result<()> {
        // Check if already registered
        if self.kernels.contains_key(name) {
            return Ok(());
        }

        println!("  Compiling kernel: {}", name);

        // Compile PTX
        let ptx = compile_ptx_with_opts(code, CompileOptions::default())
            .with_context(|| format!("Failed to compile kernel: {}", name))?;

        // Load module - already returns Arc<CudaModule>
        let module = self.context.load_module(ptx)
            .with_context(|| format!("Failed to load PTX module for: {}", name))?;

        // Get function
        let func = module.load_function(name)
            .with_context(|| format!("Failed to load function: {}", name))?;

        // Store (module is already Arc wrapped)
        self.modules.insert(name.to_string(), module);
        self.kernels.insert(name.to_string(), Arc::new(func));

        println!("    ✅ Kernel '{}' registered", name);
        Ok(())
    }

    /// Register all standard kernels
    pub fn register_standard_kernels(&mut self) -> Result<()> {
        println!("Registering standard GPU kernels...");

        self.register_kernel("vector_add", kernels::VECTOR_ADD)?;
        self.register_kernel("matmul", kernels::MATRIX_MUL)?;
        self.register_kernel("relu", kernels::RELU)?;
        self.register_kernel("softmax", kernels::SOFTMAX)?;
        self.register_kernel("sigmoid", kernels::SIGMOID)?;
        self.register_kernel("tanh_activation", kernels::TANH)?;
        self.register_kernel("batch_norm", kernels::BATCH_NORM)?;

        println!("✅ All standard kernels registered");
        Ok(())
    }

    /// Get a kernel function
    pub fn get_kernel(&self, name: &str) -> Result<&Arc<CudaFunction>> {
        self.kernels.get(name)
            .ok_or_else(|| anyhow::anyhow!("Kernel '{}' not found", name))
    }

    /// Get the CUDA context
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Execute vector addition
    pub fn vector_add(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        let n = a.len();
        anyhow::ensure!(b.len() == n, "Vector dimensions must match");

        let stream = self.context.default_stream();
        let kernel = self.get_kernel("vector_add")?;

        // Upload data
        let a_dev = stream.memcpy_stod(a)?;
        let b_dev = stream.memcpy_stod(b)?;
        let mut c_dev = stream.alloc_zeros::<f32>(n)?;

        // Launch kernel
        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe {
            stream.launch_builder(kernel)
                .arg(&a_dev)
                .arg(&b_dev)
                .arg(&mut c_dev)
                .arg(&(n as i32))
                .launch(cfg)?;
        }

        // Download result
        let result = stream.memcpy_dtov(&c_dev)?;
        Ok(result)
    }

    /// Execute matrix multiplication
    pub fn matrix_multiply(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        anyhow::ensure!(a.len() == m * k, "Matrix A dimensions incorrect");
        anyhow::ensure!(b.len() == k * n, "Matrix B dimensions incorrect");

        let stream = self.context.default_stream();
        let kernel = self.get_kernel("matmul")?;

        // Upload data
        let a_dev = stream.memcpy_stod(a)?;
        let b_dev = stream.memcpy_stod(b)?;
        let mut c_dev = stream.alloc_zeros::<f32>(m * n)?;

        // Launch with 2D grid
        let block_size = 16;
        let grid_x = (n as u32 + block_size - 1) / block_size;
        let grid_y = (m as u32 + block_size - 1) / block_size;

        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (block_size, block_size, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream.launch_builder(kernel)
                .arg(&a_dev)
                .arg(&b_dev)
                .arg(&mut c_dev)
                .arg(&(m as i32))
                .arg(&(k as i32))
                .arg(&(n as i32))
                .launch(cfg)?;
        }

        // Download result
        let result = stream.memcpy_dtov(&c_dev)?;
        Ok(result)
    }

    /// Apply ReLU activation in-place
    pub fn relu_inplace(&self, data: &mut [f32]) -> Result<()> {
        let n = data.len();
        let stream = self.context.default_stream();
        let kernel = self.get_kernel("relu")?;

        // Upload data
        let mut data_dev = stream.memcpy_stod(data)?;

        // Launch kernel
        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe {
            stream.launch_builder(kernel)
                .arg(&mut data_dev)
                .arg(&(n as i32))
                .launch(cfg)?;
        }

        // Download result
        let result = stream.memcpy_dtov(&data_dev)?;
        data.copy_from_slice(&result);
        Ok(())
    }

    /// Apply softmax activation
    pub fn softmax(&self, data: &mut [f32], batch_size: usize, num_classes: usize) -> Result<()> {
        anyhow::ensure!(
            data.len() == batch_size * num_classes,
            "Data dimensions must match batch_size * num_classes"
        );

        let stream = self.context.default_stream();
        let kernel = self.get_kernel("softmax")?;

        // Upload data
        let mut data_dev = stream.memcpy_stod(data)?;

        // Launch with one block per batch
        let cfg = LaunchConfig {
            grid_dim: (batch_size as u32, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream.launch_builder(kernel)
                .arg(&mut data_dev)
                .arg(&(batch_size as i32))
                .arg(&(num_classes as i32))
                .launch(cfg)?;
        }

        // Download result
        let result = stream.memcpy_dtov(&data_dev)?;
        data.copy_from_slice(&result);
        Ok(())
    }

    /// Apply sigmoid activation
    pub fn sigmoid_inplace(&self, data: &mut [f32]) -> Result<()> {
        let n = data.len();
        let stream = self.context.default_stream();
        let kernel = self.get_kernel("sigmoid")?;

        // Upload data
        let mut data_dev = stream.memcpy_stod(data)?;

        // Launch kernel
        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe {
            stream.launch_builder(kernel)
                .arg(&mut data_dev)
                .arg(&(n as i32))
                .launch(cfg)?;
        }

        // Download result
        let result = stream.memcpy_dtov(&data_dev)?;
        data.copy_from_slice(&result);
        Ok(())
    }

    /// Apply tanh activation
    pub fn tanh_inplace(&self, data: &mut [f32]) -> Result<()> {
        let n = data.len();
        let stream = self.context.default_stream();
        let kernel = self.get_kernel("tanh_activation")?;

        // Upload data
        let mut data_dev = stream.memcpy_stod(data)?;

        // Launch kernel
        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe {
            stream.launch_builder(kernel)
                .arg(&mut data_dev)
                .arg(&(n as i32))
                .launch(cfg)?;
        }

        // Download result
        let result = stream.memcpy_dtov(&data_dev)?;
        data.copy_from_slice(&result);
        Ok(())
    }
}

/// Global kernel executor instance (lazy initialized)
pub fn get_global_executor() -> Result<&'static std::sync::Mutex<GpuKernelExecutor>> {
    use std::sync::{Mutex, OnceLock};

    static EXECUTOR: OnceLock<Mutex<GpuKernelExecutor>> = OnceLock::new();

    let executor = EXECUTOR.get_or_init(|| {
        let mut exec = GpuKernelExecutor::new(0)
            .expect("Failed to create GPU kernel executor");
        exec.register_standard_kernels()
            .expect("Failed to register standard kernels");
        Mutex::new(exec)
    });

    Ok(executor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_executor() -> Result<()> {
        let mut executor = GpuKernelExecutor::new(0)?;
        executor.register_standard_kernels()?;

        // Test vector addition
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c = executor.vector_add(&a, &b)?;

        assert_eq!(c.len(), 4);
        assert!((c[0] - 6.0).abs() < 1e-6);
        assert!((c[3] - 12.0).abs() < 1e-6);

        // Test ReLU
        let mut data = vec![-1.0, 0.0, 1.0, -0.5, 2.0];
        executor.relu_inplace(&mut data)?;

        assert_eq!(data, vec![0.0, 0.0, 1.0, 0.0, 2.0]);

        Ok(())
    }

    #[test]
    fn test_matrix_multiply() -> Result<()> {
        let mut executor = GpuKernelExecutor::new(0)?;
        executor.register_kernel("matmul", kernels::MATRIX_MUL)?;

        // 2x3 * 3x2 = 2x2
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2
        let c = executor.matrix_multiply(&a, &b, 2, 3, 2)?;

        // Expected:
        // [1,2,3] * [[7,8],[9,10],[11,12]] = [58, 64]
        // [4,5,6] * [[7,8],[9,10],[11,12]] = [139, 154]

        assert_eq!(c.len(), 4);
        assert!((c[0] - 58.0).abs() < 1e-5);
        assert!((c[1] - 64.0).abs() < 1e-5);
        assert!((c[2] - 139.0).abs() < 1e-5);
        assert!((c[3] - 154.0).abs() < 1e-5);

        Ok(())
    }
}