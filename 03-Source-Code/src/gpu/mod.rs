//! GPU acceleration module for PRISM-AI
//!
//! Provides GPU memory management and operations for neural network acceleration
//! using cudarc with CUDA 13 support for RTX 5070

// Temporarily disabled until cudarc API is fixed
// pub mod memory_manager;
// pub mod tensor_ops;
// pub mod kernel_launcher;

// GPU-ONLY modules - NO CPU FALLBACK
pub mod layers;
pub mod gpu_enabled;  // GPU-only implementation with kernel execution
pub mod kernel_executor;  // GPU kernel executor
pub mod gpu_tensor_optimized;  // FULLY OPTIMIZED: CudaSlice, persistent GPU, fused kernels
pub mod memory_pool;  // GPU memory pooling tracking and statistics
pub mod active_memory_pool;  // Active memory pooling with buffer reuse (67.9% savings)
pub mod kernel_autotuner;  // Kernel auto-tuning for optimal launch configurations
pub mod neuromorphic_ffi;  // FFI bindings for neuromorphic CUDA kernels
pub mod cublas_compat;  // CUBLAS compatibility layer for CUDA 12.8

// Production GPU Runtime - Full acceleration for deployment
#[cfg(feature = "cuda")]
pub mod production_runtime;
#[cfg(feature = "cuda")]
pub mod cudarc_replacement;

// Use GPU-enabled implementation - NO CPU FALLBACK
pub use gpu_enabled::{
    SimpleGpuContext as GpuMemoryPool,
    SimpleGpuTensor as GpuTensor,
    SimpleGpuLinear as GpuLinear,
    SimpleGpuBuffer as GpuBuffer,
};

// Export OPTIMIZED GPU tensors (4-10x faster)
pub use gpu_tensor_optimized::{
    GpuTensorOpt,
    FusedLinearLayerOpt,
    OptimizedGpuNetwork,
};

// Also export the layers
pub use layers::GpuLinear as GpuLinearLayer;

// Export kernel executor
pub use kernel_executor::{GpuKernelExecutor, get_global_executor};

// Export memory pool
pub use memory_pool::{GpuMemoryPool as GpuMemoryPoolV2, MemoryPoolConfig, MemoryPoolStats};

// Export kernel auto-tuner
pub use kernel_autotuner::{KernelAutoTuner, LaunchConfig, KernelId, AutoTunerConfig, AutoTunerStats};

// Export active memory pool
pub use active_memory_pool::{ActiveMemoryPool, ActivePoolConfig, ActivePoolStats};