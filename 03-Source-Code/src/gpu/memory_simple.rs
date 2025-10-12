//! Simplified GPU Memory Manager that actually works with cudarc
//!
//! Uses the actual cudarc API correctly

use anyhow::{Result, Context};
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::{
    driver::{CudaContext, CudaSlice},
};

/// Simple GPU memory operations
pub struct SimpleGpuMemory {
    #[cfg(feature = "cuda")]
    pub context: Arc<CudaContext>,

    #[cfg(not(feature = "cuda"))]
    _phantom: std::marker::PhantomData<()>,
}

impl SimpleGpuMemory {
    /// Create new GPU memory context
    pub fn new() -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            let context = CudaContext::new(0)
                .context("Failed to create CUDA context")?;
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

    /// Allocate GPU memory for f32
    #[cfg(feature = "cuda")]
    pub fn allocate_f32(&self, size: usize) -> Result<CudaSlice<f32>> {
        // Create a buffer of zeros
        let zeros = vec![0.0f32; size];

        // Use cudarc's actual API to transfer to GPU
        let gpu_buffer = self.context.alloc(size)
            .context("Failed to allocate GPU memory")?;

        // Copy zeros to GPU
        self.context.htod_copy(zeros.as_slice(), &gpu_buffer)
            .context("Failed to copy to GPU")?;

        Ok(gpu_buffer)
    }

    #[cfg(not(feature = "cuda"))]
    pub fn allocate_f32(&self, size: usize) -> Result<Vec<f32>> {
        Ok(vec![0.0f32; size])
    }

    /// Transfer CPU data to GPU
    #[cfg(feature = "cuda")]
    pub fn cpu_to_gpu(&self, data: &[f32]) -> Result<CudaSlice<f32>> {
        let gpu_buffer = self.context.alloc(data.len())
            .context("Failed to allocate GPU memory")?;

        self.context.htod_copy(data, &gpu_buffer)
            .context("Failed to copy to GPU")?;

        Ok(gpu_buffer)
    }

    #[cfg(not(feature = "cuda"))]
    pub fn cpu_to_gpu(&self, data: &[f32]) -> Result<Vec<f32>> {
        Ok(data.to_vec())
    }

    /// Transfer GPU data to CPU
    #[cfg(feature = "cuda")]
    pub fn gpu_to_cpu(&self, buffer: &CudaSlice<f32>) -> Result<Vec<f32>> {
        let mut result = vec![0.0f32; buffer.len()];

        self.context.dtoh_copy(buffer, &mut result)
            .context("Failed to copy from GPU")?;

        Ok(result)
    }

    #[cfg(not(feature = "cuda"))]
    pub fn gpu_to_cpu(&self, data: &Vec<f32>) -> Result<Vec<f32>> {
        Ok(data.clone())
    }

    /// Check if GPU is available
    pub fn is_available() -> bool {
        #[cfg(feature = "cuda")]
        {
            CudaContext::new(0).is_ok()
        }

        #[cfg(not(feature = "cuda"))]
        false
    }
}

// Type aliases for easier use
#[cfg(feature = "cuda")]
pub type GpuBuffer<T> = CudaSlice<T>;

#[cfg(not(feature = "cuda"))]
pub type GpuBuffer<T> = Vec<T>;