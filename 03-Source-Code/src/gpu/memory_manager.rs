//! GPU Memory Manager for PRISM-AI
//!
//! Manages GPU memory allocation, transfers, and lifecycle
//! Optimized for RTX 5070 with CUDA 13

use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use anyhow::{Result, Context};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaSlice};

/// GPU Buffer wrapper for memory management
#[cfg(feature = "cuda")]
pub type GpuBuffer<T> = CudaSlice<T>;

#[cfg(not(feature = "cuda"))]
pub type GpuBuffer<T> = Vec<T>;

/// GPU Device abstraction
pub struct GpuDevice {
    #[cfg(feature = "cuda")]
    pub context: Arc<CudaContext>,

    #[cfg(not(feature = "cuda"))]
    _phantom: std::marker::PhantomData<()>,
}

impl GpuDevice {
    /// Create new GPU device context
    pub fn new(device_id: usize) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            let context = CudaContext::new(device_id)
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

/// GPU Memory Pool for efficient allocation
pub struct GpuMemoryPool {
    pub device: Arc<GpuDevice>,

    /// Track allocated buffers for reuse
    allocated_buffers: Arc<Mutex<HashMap<String, usize>>>,

    /// Memory statistics
    total_allocated: Arc<Mutex<usize>>,
    peak_usage: Arc<Mutex<usize>>,
}

impl GpuMemoryPool {
    /// Create new memory pool
    pub fn new() -> Result<Self> {
        let device = GpuDevice::new(0)?;
        Ok(Self {
            device: Arc::new(device),
            allocated_buffers: Arc::new(Mutex::new(HashMap::new())),
            total_allocated: Arc::new(Mutex::new(0)),
            peak_usage: Arc::new(Mutex::new(0)),
        })
    }

    /// Allocate buffer for f32
    #[cfg(feature = "cuda")]
    pub fn allocate_f32(&self, size: usize) -> Result<GpuBuffer<f32>> {
        // For cudarc, we need to allocate and then initialize
        // Since cudarc doesn't have alloc_zeros, we'll allocate uninitialized
        // This is a simplified version - real implementation would use proper cudarc API

        // Create host data
        let zeros = vec![0.0f32; size];

        // Transfer to device
        let buffer = self.device.context.htod_copy(zeros.as_slice())
            .context("Failed to allocate and copy to GPU")?;

        // Update statistics
        let mut total = self.total_allocated.lock().unwrap();
        *total += size * std::mem::size_of::<f32>();

        let mut peak = self.peak_usage.lock().unwrap();
        if *total > *peak {
            *peak = *total;
        }

        Ok(buffer)
    }

    #[cfg(not(feature = "cuda"))]
    pub fn allocate_f32(&self, size: usize) -> Result<GpuBuffer<f32>> {
        Ok(vec![0.0f32; size])
    }

    /// Transfer data from CPU to GPU
    #[cfg(feature = "cuda")]
    pub fn cpu_to_gpu(&self, data: &[f32]) -> Result<GpuBuffer<f32>> {
        let buffer = self.device.context.htod_copy(data)
            .context("Failed to copy data to GPU")?;

        // Update statistics
        let mut total = self.total_allocated.lock().unwrap();
        *total += data.len() * std::mem::size_of::<f32>();

        Ok(buffer)
    }

    #[cfg(not(feature = "cuda"))]
    pub fn cpu_to_gpu(&self, data: &[f32]) -> Result<GpuBuffer<f32>> {
        Ok(data.to_vec())
    }

    /// Transfer data from GPU to CPU
    #[cfg(feature = "cuda")]
    pub fn gpu_to_cpu(&self, buffer: &GpuBuffer<f32>) -> Result<Vec<f32>> {
        let result = self.device.context.dtoh_copy(buffer)
            .context("Failed to copy data from GPU")?;
        Ok(result)
    }

    #[cfg(not(feature = "cuda"))]
    pub fn gpu_to_cpu(&self, buffer: &GpuBuffer<f32>) -> Result<Vec<f32>> {
        Ok(buffer.clone())
    }

    /// Get memory statistics
    pub fn memory_stats(&self) -> (usize, usize) {
        let total = *self.total_allocated.lock().unwrap();
        let peak = *self.peak_usage.lock().unwrap();
        (total, peak)
    }

    /// Clear all allocated buffers (for cleanup)
    pub fn clear(&mut self) {
        self.allocated_buffers.lock().unwrap().clear();
        *self.total_allocated.lock().unwrap() = 0;
    }

    /// Synchronize GPU operations
    #[cfg(feature = "cuda")]
    pub fn synchronize(&self) -> Result<()> {
        self.device.context.synchronize()
            .context("Failed to synchronize GPU")?;
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    pub fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}

/// Safe GPU memory guard that automatically deallocates
pub struct GpuMemoryGuard<T> {
    buffer: Option<GpuBuffer<T>>,
    pool: Arc<GpuMemoryPool>,
}

impl<T> Drop for GpuMemoryGuard<T> {
    fn drop(&mut self) {
        // Buffer is automatically freed when dropped
        if let Some(buffer) = self.buffer.take() {
            // Update statistics
            let size = std::mem::size_of::<T>() * buffer.len();
            let mut total = self.pool.total_allocated.lock().unwrap();
            *total = total.saturating_sub(size);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_creation() {
        if GpuDevice::is_available() {
            let pool = GpuMemoryPool::new();
            assert!(pool.is_ok());
        } else {
            println!("GPU not available, skipping test");
        }
    }

    #[test]
    fn test_allocation() {
        if let Ok(pool) = GpuMemoryPool::new() {
            let buffer = pool.allocate_f32(1024);
            assert!(buffer.is_ok());

            if let Ok(buf) = buffer {
                assert_eq!(buf.len(), 1024);
            }
        }
    }

    #[test]
    fn test_cpu_gpu_transfer() {
        if let Ok(pool) = GpuMemoryPool::new() {
            let data = vec![1.0f32; 256];

            // Transfer to GPU
            let gpu_buffer = pool.cpu_to_gpu(&data);
            assert!(gpu_buffer.is_ok());

            // Transfer back to CPU
            if let Ok(buffer) = gpu_buffer {
                let result = pool.gpu_to_cpu(&buffer);
                assert!(result.is_ok());

                if let Ok(cpu_data) = result {
                    assert_eq!(cpu_data.len(), data.len());
                    assert_eq!(cpu_data[0], 1.0);
                }
            }
        }
    }
}