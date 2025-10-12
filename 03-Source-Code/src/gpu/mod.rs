//! GPU acceleration module for PRISM-AI
//!
//! Provides GPU memory management and operations for neural network acceleration
//! using cudarc with CUDA 13 support for RTX 5070

// Temporarily disabled until cudarc API is fixed
// pub mod memory_manager;
// pub mod tensor_ops;
// pub mod kernel_launcher;

// This module doesn't import the problematic modules
pub mod layers;
pub mod simple_gpu;
// pub mod simple_gpu_v2;  // Disabled - using gpu_enabled instead
// pub mod gpu_real;  // Disabled - API issues
pub mod gpu_enabled;
pub mod gpu_executor;

// Use simple implementation for now to ensure compilation
pub use simple_gpu::{
    SimpleGpuContext as GpuMemoryPool,
    SimpleGpuTensor as GpuTensor,
    SimpleGpuLinear as GpuLinear,
    SimpleGpuBuffer as GpuBuffer,
};

// Also export the layers
pub use layers::GpuLinear as GpuLinearLayer;