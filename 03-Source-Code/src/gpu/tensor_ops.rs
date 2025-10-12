//! GPU Tensor Operations
//!
//! Provides GPU-accelerated tensor operations for neural networks

use anyhow::{Result, Context};
use std::sync::Arc;

use super::memory_manager::{GpuMemoryPool, GpuBuffer};
use super::kernel_launcher::GpuKernelLauncher;

/// GPU Tensor representation
pub struct GpuTensor {
    /// Underlying data buffer
    #[cfg(feature = "cuda")]
    data: GpuBuffer<f32>,

    #[cfg(not(feature = "cuda"))]
    data: Vec<f32>,

    /// Shape of the tensor
    shape: Vec<usize>,

    /// Memory pool reference
    pub pool: Arc<GpuMemoryPool>,

    /// Kernel launcher for GPU operations
    #[cfg(feature = "cuda")]
    kernel_launcher: Arc<GpuKernelLauncher>,
}

impl GpuTensor {
    /// Create new GPU tensor from CPU data
    pub fn from_cpu(data: Vec<f32>, shape: Vec<usize>, pool: Arc<GpuMemoryPool>) -> Result<Self> {
        // Validate shape matches data
        let total_size: usize = shape.iter().product();
        if total_size != data.len() {
            anyhow::bail!("Shape {:?} doesn't match data length {}", shape, data.len());
        }

        #[cfg(feature = "cuda")]
        {
            let gpu_data = pool.cpu_to_gpu(&data)?;
            let kernel_launcher = Arc::new(GpuKernelLauncher::new(pool.device.context.clone())?);

            Ok(Self {
                data: gpu_data,
                shape,
                pool,
                kernel_launcher,
            })
        }

        #[cfg(not(feature = "cuda"))]
        {
            Ok(Self {
                data,
                shape,
                pool,
            })
        }
    }

    /// Create zeros tensor on GPU
    pub fn zeros(shape: Vec<usize>, pool: Arc<GpuMemoryPool>) -> Result<Self> {
        let total_size: usize = shape.iter().product();

        #[cfg(feature = "cuda")]
        {
            let data = pool.allocate_f32(total_size)?;
            let kernel_launcher = Arc::new(GpuKernelLauncher::new(pool.device.context.clone())?);

            Ok(Self {
                data,
                shape,
                pool,
                kernel_launcher,
            })
        }

        #[cfg(not(feature = "cuda"))]
        {
            let data = vec![0.0f32; total_size];
            Ok(Self {
                data,
                shape,
                pool,
            })
        }
    }

    /// Get tensor shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get total number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Transfer tensor to CPU
    pub fn to_cpu(&self) -> Result<Vec<f32>> {
        #[cfg(feature = "cuda")]
        return self.pool.gpu_to_cpu(&self.data);

        #[cfg(not(feature = "cuda"))]
        Ok(self.data.clone())
    }

    /// Reshape tensor (doesn't copy data)
    pub fn reshape(&mut self, new_shape: Vec<usize>) -> Result<()> {
        let new_size: usize = new_shape.iter().product();
        if new_size != self.numel() {
            anyhow::bail!("Cannot reshape from {:?} to {:?}", self.shape, new_shape);
        }
        self.shape = new_shape;
        Ok(())
    }

    /// Get raw data buffer (for kernel operations)
    #[cfg(feature = "cuda")]
    pub fn data_ptr(&self) -> &GpuBuffer<f32> {
        &self.data
    }

    #[cfg(not(feature = "cuda"))]
    pub fn data_ptr(&self) -> &Vec<f32> {
        &self.data
    }

    /// Matrix multiply (GPU-accelerated)
    pub fn matmul(&self, other: &GpuTensor) -> Result<GpuTensor> {
        // Validate shapes for matrix multiplication
        if self.shape.len() != 2 || other.shape.len() != 2 {
            anyhow::bail!("matmul requires 2D tensors");
        }

        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];

        if k != other.shape[0] {
            anyhow::bail!("Shape mismatch: [{}, {}] x [{}, {}]", m, k, other.shape[0], n);
        }

        // Create output tensor
        let output_shape = vec![m, n];
        let mut output = GpuTensor::zeros(output_shape.clone(), self.pool.clone())?;

        #[cfg(feature = "cuda")]
        {
            // Use GPU kernel for matrix multiplication
            self.kernel_launcher.matmul(
                &self.data,
                &other.data,
                &mut output.data,
                m as i32,
                k as i32,
                n as i32,
            )?;
        }

        #[cfg(not(feature = "cuda"))]
        {
            // CPU fallback
            self.matmul_cpu(&other, &mut output)?;
        }

        Ok(output)
    }

    /// CPU fallback for matrix multiplication
    fn matmul_cpu(&self, other: &GpuTensor, output: &mut GpuTensor) -> Result<()> {
        let a_data = self.to_cpu()?;
        let b_data = other.to_cpu()?;

        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];

        let mut c_data = vec![0.0f32; m * n];

        // Simple matrix multiplication
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a_data[i * k + l] * b_data[l * n + j];
                }
                c_data[i * n + j] = sum;
            }
        }

        // Update output
        #[cfg(feature = "cuda")]
        {
            output.data = output.pool.cpu_to_gpu(&c_data)?;
        }

        #[cfg(not(feature = "cuda"))]
        {
            output.data = c_data;
        }

        Ok(())
    }

    /// Add bias to tensor
    pub fn add_bias(&mut self, bias: &GpuTensor) -> Result<()> {
        if self.shape.len() != 2 {
            anyhow::bail!("add_bias requires 2D tensor");
        }

        let batch_size = self.shape[0];
        let features = self.shape[1];

        if bias.shape != vec![features] && bias.shape != vec![1, features] {
            anyhow::bail!("Bias shape {:?} incompatible with tensor shape {:?}",
                        bias.shape, self.shape);
        }

        #[cfg(feature = "cuda")]
        {
            // Use GPU kernel for bias addition
            self.kernel_launcher.add_bias(
                &mut self.data,
                &bias.data,
                batch_size as i32,
                features as i32,
            )?;
        }

        #[cfg(not(feature = "cuda"))]
        {
            // CPU implementation
            let bias_data = bias.to_cpu()?;
            for b in 0..batch_size {
                for f in 0..features {
                    self.data[b * features + f] += bias_data[f % bias_data.len()];
                }
            }
        }

        Ok(())
    }

    /// Apply ReLU activation
    pub fn relu(&mut self) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            // Use GPU kernel for ReLU
            self.kernel_launcher.relu(&mut self.data, self.numel() as i32)?;
        }

        #[cfg(not(feature = "cuda"))]
        {
            // CPU implementation
            for x in &mut self.data {
                *x = x.max(0.0);
            }
        }

        Ok(())
    }

    /// Apply softmax (for classification)
    pub fn softmax(&mut self, dim: usize) -> Result<()> {
        if dim != 1 || self.shape.len() != 2 {
            anyhow::bail!("Softmax currently only supports dim=1 on 2D tensors");
        }

        let batch_size = self.shape[0];
        let num_classes = self.shape[1];

        #[cfg(feature = "cuda")]
        {
            // Create temporary output buffer for softmax
            let mut output = self.pool.allocate_f32(batch_size * num_classes)?;

            // Use GPU kernel for softmax
            self.kernel_launcher.softmax(
                &self.data,
                &mut output,
                batch_size as i32,
                num_classes as i32,
            )?;

            // Replace data with softmax result
            self.data = output;
        }

        #[cfg(not(feature = "cuda"))]
        {
            // CPU implementation
            for b in 0..batch_size {
                let offset = b * num_classes;

                // Find max for numerical stability
                let max_val = self.data[offset..offset + num_classes]
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);

                // Compute exp and sum
                let mut sum = 0.0f32;
                for i in 0..num_classes {
                    self.data[offset + i] = (self.data[offset + i] - max_val).exp();
                    sum += self.data[offset + i];
                }

                // Normalize
                for i in 0..num_classes {
                    self.data[offset + i] /= sum;
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        if let Ok(pool) = GpuMemoryPool::new() {
            let pool = Arc::new(pool);
            let data = vec![1.0f32; 64];
            let tensor = GpuTensor::from_cpu(data.clone(), vec![8, 8], pool);

            assert!(tensor.is_ok());
            if let Ok(t) = tensor {
                assert_eq!(t.shape(), &[8, 8]);
                assert_eq!(t.numel(), 64);
            }
        }
    }

    #[test]
    fn test_matmul() {
        if let Ok(pool) = GpuMemoryPool::new() {
            let pool = Arc::new(pool);

            // Create two matrices
            let a_data = vec![1.0f32; 6];  // 2x3
            let b_data = vec![2.0f32; 12]; // 3x4

            let a = GpuTensor::from_cpu(a_data, vec![2, 3], pool.clone()).unwrap();
            let b = GpuTensor::from_cpu(b_data, vec![3, 4], pool.clone()).unwrap();

            let c = a.matmul(&b);
            assert!(c.is_ok());

            if let Ok(result) = c {
                assert_eq!(result.shape(), &[2, 4]);

                let cpu_data = result.to_cpu().unwrap();
                // Each element should be 3*1*2 = 6
                assert_eq!(cpu_data[0], 6.0);
            }
        }
    }

    #[test]
    fn test_softmax() {
        if let Ok(pool) = GpuMemoryPool::new() {
            let pool = Arc::new(pool);

            let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            let mut tensor = GpuTensor::from_cpu(data, vec![2, 3], pool).unwrap();

            tensor.softmax(1).unwrap();

            let result = tensor.to_cpu().unwrap();

            // Check that each row sums to 1
            let sum1: f32 = result[0..3].iter().sum();
            let sum2: f32 = result[3..6].iter().sum();

            assert!((sum1 - 1.0).abs() < 1e-6);
            assert!((sum2 - 1.0).abs() < 1e-6);
        }
    }
}