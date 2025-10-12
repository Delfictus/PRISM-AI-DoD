//! GPU Kernel Launcher
//!
//! Simplified kernel execution for CUDA operations

use anyhow::{Result, Context};
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaSlice};

use super::memory_manager::GpuBuffer;

/// Simple GPU Kernel Launcher
pub struct GpuKernelLauncher {
    #[cfg(feature = "cuda")]
    context: Arc<CudaContext>,

    #[cfg(not(feature = "cuda"))]
    _phantom: std::marker::PhantomData<()>,
}

impl GpuKernelLauncher {
    /// Create new kernel launcher
    pub fn new(context: Arc<CudaContext>) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            Ok(Self {
                context,
            })
        }

        #[cfg(not(feature = "cuda"))]
        {
            Ok(Self {
                _phantom: std::marker::PhantomData,
            })
        }
    }

    /// Execute matrix multiplication kernel
    /// C = A @ B where A is [M x K], B is [K x N], C is [M x N]
    pub fn matmul(
        &self,
        a: &GpuBuffer<f32>,
        b: &GpuBuffer<f32>,
        c: &mut GpuBuffer<f32>,
        m: i32,
        k: i32,
        n: i32,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            // For now, use CPU fallback since PTX loading is complex
            // In production, this would load and execute the PTX kernel
            self.matmul_cpu_fallback(a, b, c, m as usize, k as usize, n as usize)?;
            Ok(())
        }

        #[cfg(not(feature = "cuda"))]
        {
            self.matmul_cpu_fallback(a, b, c, m as usize, k as usize, n as usize)
        }
    }

    /// CPU fallback for matrix multiplication
    fn matmul_cpu_fallback(
        &self,
        a: &GpuBuffer<f32>,
        b: &GpuBuffer<f32>,
        c: &mut GpuBuffer<f32>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            // Download from GPU
            let a_cpu = self.context.dtoh_copy(a)?;
            let b_cpu = self.context.dtoh_copy(b)?;

            // Compute on CPU
            let mut c_cpu = vec![0.0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for l in 0..k {
                        sum += a_cpu[i * k + l] * b_cpu[l * n + j];
                    }
                    c_cpu[i * n + j] = sum;
                }
            }

            // Upload back to GPU
            let c_gpu = self.context.htod_copy(&c_cpu)?;
            *c = c_gpu;
            Ok(())
        }

        #[cfg(not(feature = "cuda"))]
        {
            // Direct CPU computation
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for l in 0..k {
                        sum += a[i * k + l] * b[l * n + j];
                    }
                    c[i * n + j] = sum;
                }
            }
            Ok(())
        }
    }

    /// Add bias to tensor (broadcasted)
    pub fn add_bias(
        &self,
        output: &mut GpuBuffer<f32>,
        bias: &GpuBuffer<f32>,
        batch_size: i32,
        features: i32,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            // Download, compute, upload
            let mut output_cpu = self.context.dtoh_copy(output)?;
            let bias_cpu = self.context.dtoh_copy(bias)?;

            for b in 0..batch_size as usize {
                for f in 0..features as usize {
                    output_cpu[b * features as usize + f] += bias_cpu[f];
                }
            }

            let output_gpu = self.context.htod_copy(&output_cpu)?;
            *output = output_gpu;
            Ok(())
        }

        #[cfg(not(feature = "cuda"))]
        {
            for b in 0..batch_size as usize {
                for f in 0..features as usize {
                    output[b * features as usize + f] += bias[f];
                }
            }
            Ok(())
        }
    }

    /// Apply ReLU activation
    pub fn relu(&self, data: &mut GpuBuffer<f32>, size: i32) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let mut data_cpu = self.context.dtoh_copy(data)?;
            for x in &mut data_cpu {
                *x = x.max(0.0);
            }
            let data_gpu = self.context.htod_copy(&data_cpu)?;
            *data = data_gpu;
            Ok(())
        }

        #[cfg(not(feature = "cuda"))]
        {
            for i in 0..size as usize {
                data[i] = data[i].max(0.0);
            }
            Ok(())
        }
    }

    /// Apply softmax activation
    pub fn softmax(
        &self,
        input: &GpuBuffer<f32>,
        output: &mut GpuBuffer<f32>,
        batch_size: i32,
        num_classes: i32,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let input_cpu = self.context.dtoh_copy(input)?;
            let mut output_cpu = vec![0.0f32; (batch_size * num_classes) as usize];

            for b in 0..batch_size as usize {
                let offset = b * num_classes as usize;

                // Find max for numerical stability
                let max_val = input_cpu[offset..offset + num_classes as usize]
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);

                // Compute exp and sum
                let mut sum = 0.0f32;
                for i in 0..num_classes as usize {
                    output_cpu[offset + i] = (input_cpu[offset + i] - max_val).exp();
                    sum += output_cpu[offset + i];
                }

                // Normalize
                for i in 0..num_classes as usize {
                    output_cpu[offset + i] /= sum;
                }
            }

            let output_gpu = self.context.htod_copy(&output_cpu)?;
            *output = output_gpu;
            Ok(())
        }

        #[cfg(not(feature = "cuda"))]
        {
            for b in 0..batch_size as usize {
                let offset = b * num_classes as usize;

                // Find max for numerical stability
                let max_val = input[offset..offset + num_classes as usize]
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);

                // Compute exp and sum
                let mut sum = 0.0f32;
                for i in 0..num_classes as usize {
                    output[offset + i] = (input[offset + i] - max_val).exp();
                    sum += output[offset + i];
                }

                // Normalize
                for i in 0..num_classes as usize {
                    output[offset + i] /= sum;
                }
            }
            Ok(())
        }
    }

    /// Apply sigmoid activation
    pub fn sigmoid(&self, data: &mut GpuBuffer<f32>, size: i32) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let mut data_cpu = self.context.dtoh_copy(data)?;
            for x in &mut data_cpu {
                *x = 1.0 / (1.0 + (-*x).exp());
            }
            let data_gpu = self.context.htod_copy(&data_cpu)?;
            *data = data_gpu;
            Ok(())
        }

        #[cfg(not(feature = "cuda"))]
        {
            for i in 0..size as usize {
                data[i] = 1.0 / (1.0 + (-data[i]).exp());
            }
            Ok(())
        }
    }

    /// Apply tanh activation
    pub fn tanh(&self, data: &mut GpuBuffer<f32>, size: i32) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let mut data_cpu = self.context.dtoh_copy(data)?;
            for x in &mut data_cpu {
                *x = x.tanh();
            }
            let data_gpu = self.context.htod_copy(&data_cpu)?;
            *data = data_gpu;
            Ok(())
        }

        #[cfg(not(feature = "cuda"))]
        {
            for i in 0..size as usize {
                data[i] = data[i].tanh();
            }
            Ok(())
        }
    }

    /// SAXPY operation: y = y + alpha * x
    pub fn saxpy(
        &self,
        y: &mut GpuBuffer<f32>,
        x: &GpuBuffer<f32>,
        alpha: f32,
        size: i32,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let mut y_cpu = self.context.dtoh_copy(y)?;
            let x_cpu = self.context.dtoh_copy(x)?;

            for i in 0..size as usize {
                y_cpu[i] += alpha * x_cpu[i];
            }

            let y_gpu = self.context.htod_copy(&y_cpu)?;
            *y = y_gpu;
            Ok(())
        }

        #[cfg(not(feature = "cuda"))]
        {
            for i in 0..size as usize {
                y[i] += alpha * x[i];
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuMemoryPool;

    #[test]
    fn test_kernel_launcher_creation() {
        if let Ok(pool) = GpuMemoryPool::new() {
            let pool = Arc::new(pool);

            #[cfg(feature = "cuda")]
            {
                let launcher = GpuKernelLauncher::new(pool.device.context.clone());
                assert!(launcher.is_ok());
            }
        }
    }
}