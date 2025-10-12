//! GPU Executor with Automatic Fallback
//!
//! Provides transparent GPU acceleration with CPU fallback when:
//! 1. GPU is not available (no CUDA driver)
//! 2. GPU operations fail (out of memory, etc.)
//! 3. Kernels are not compiled (missing PTX)
//!
//! The executor tracks performance and automatically chooses the best backend.

use anyhow::{Result, Context};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::time::{Instant, Duration};

/// Execution backend preference
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Backend {
    /// Prefer GPU, fallback to CPU on failure
    PreferGpu,
    /// Always use CPU
    CpuOnly,
    /// Always use GPU, fail if not available
    GpuOnly,
    /// Automatically choose based on performance history
    Auto,
}

/// Operation performance statistics
#[derive(Debug, Clone)]
struct OpStats {
    cpu_time: Duration,
    gpu_time: Duration,
    cpu_runs: u64,
    gpu_runs: u64,
    last_gpu_failure: Option<Instant>,
}

impl Default for OpStats {
    fn default() -> Self {
        Self {
            cpu_time: Duration::ZERO,
            gpu_time: Duration::ZERO,
            cpu_runs: 0,
            gpu_runs: 0,
            last_gpu_failure: None,
        }
    }
}

/// GPU Executor with automatic fallback
pub struct GpuExecutor {
    backend: Backend,
    gpu_available: bool,
    stats: Arc<Mutex<HashMap<String, OpStats>>>,
    retry_threshold: Duration,
}

impl GpuExecutor {
    /// Create new executor with automatic backend selection
    pub fn new(backend: Backend) -> Result<Self> {
        let gpu_available = Self::check_gpu_available();

        if backend == Backend::GpuOnly && !gpu_available {
            anyhow::bail!("GPU required but not available");
        }

        Ok(Self {
            backend,
            gpu_available,
            stats: Arc::new(Mutex::new(HashMap::new())),
            retry_threshold: Duration::from_secs(30), // Retry GPU after 30 seconds
        })
    }

    /// Check if GPU is available
    fn check_gpu_available() -> bool {
        #[cfg(feature = "cuda")]
        {
            // Try to create a CUDA context
            if let Ok(_) = crate::gpu::simple_gpu::SimpleGpuContext::new() {
                return true;
            }
        }
        false
    }

    /// Execute operation with automatic fallback
    pub fn execute<T, F, G>(
        &self,
        op_name: &str,
        gpu_op: F,
        cpu_op: G,
    ) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
        G: FnOnce() -> Result<T>,
    {
        let start = Instant::now();

        // Decide which backend to use
        let use_gpu = match self.backend {
            Backend::CpuOnly => false,
            Backend::GpuOnly => true,
            Backend::PreferGpu => self.should_use_gpu(op_name),
            Backend::Auto => self.auto_select_backend(op_name),
        };

        // Execute with chosen backend
        let result = if use_gpu && self.gpu_available {
            match gpu_op() {
                Ok(val) => {
                    self.record_success(op_name, true, start.elapsed());
                    Ok(val)
                }
                Err(gpu_err) => {
                    self.record_failure(op_name);

                    // Fallback to CPU unless GPU-only mode
                    if self.backend == Backend::GpuOnly {
                        Err(gpu_err).context("GPU operation failed (GPU-only mode)")
                    } else {
                        eprintln!("GPU operation '{}' failed, using CPU fallback: {}",
                                 op_name, gpu_err);
                        let cpu_result = cpu_op()?;
                        self.record_success(op_name, false, start.elapsed());
                        Ok(cpu_result)
                    }
                }
            }
        } else {
            // Use CPU directly
            let cpu_result = cpu_op()?;
            self.record_success(op_name, false, start.elapsed());
            Ok(cpu_result)
        };

        result
    }

    /// Check if GPU should be used for operation
    fn should_use_gpu(&self, op_name: &str) -> bool {
        let stats = self.stats.lock().unwrap();

        if let Some(op_stats) = stats.get(op_name) {
            // Don't retry GPU immediately after failure
            if let Some(last_failure) = op_stats.last_gpu_failure {
                if last_failure.elapsed() < self.retry_threshold {
                    return false;
                }
            }
        }

        true // Default to trying GPU
    }

    /// Auto-select backend based on performance history
    fn auto_select_backend(&self, op_name: &str) -> bool {
        let stats = self.stats.lock().unwrap();

        if let Some(op_stats) = stats.get(op_name) {
            // Need at least 5 runs of each to make decision
            if op_stats.cpu_runs >= 5 && op_stats.gpu_runs >= 5 {
                let avg_cpu = op_stats.cpu_time / op_stats.cpu_runs as u32;
                let avg_gpu = op_stats.gpu_time / op_stats.gpu_runs as u32;

                // Use GPU if it's at least 20% faster
                return avg_gpu < avg_cpu.mul_f32(0.8);
            }

            // Check for recent failures
            if let Some(last_failure) = op_stats.last_gpu_failure {
                if last_failure.elapsed() < self.retry_threshold {
                    return false; // Use CPU
                }
            }
        }

        // Default: try GPU if available
        self.gpu_available
    }

    /// Record successful operation
    fn record_success(&self, op_name: &str, used_gpu: bool, duration: Duration) {
        let mut stats = self.stats.lock().unwrap();
        let op_stats = stats.entry(op_name.to_string()).or_default();

        if used_gpu {
            op_stats.gpu_time += duration;
            op_stats.gpu_runs += 1;
        } else {
            op_stats.cpu_time += duration;
            op_stats.cpu_runs += 1;
        }
    }

    /// Record GPU failure
    fn record_failure(&self, op_name: &str) {
        let mut stats = self.stats.lock().unwrap();
        let op_stats = stats.entry(op_name.to_string()).or_default();
        op_stats.last_gpu_failure = Some(Instant::now());
    }

    /// Get performance report
    pub fn get_performance_report(&self) -> String {
        let stats = self.stats.lock().unwrap();
        let mut report = String::from("\n=== GPU Executor Performance Report ===\n");

        for (op_name, op_stats) in stats.iter() {
            report.push_str(&format!("\nOperation: {}\n", op_name));

            if op_stats.cpu_runs > 0 {
                let avg_cpu = op_stats.cpu_time / op_stats.cpu_runs as u32;
                report.push_str(&format!(
                    "  CPU: {} runs, avg {:.3}ms\n",
                    op_stats.cpu_runs,
                    avg_cpu.as_secs_f64() * 1000.0
                ));
            }

            if op_stats.gpu_runs > 0 {
                let avg_gpu = op_stats.gpu_time / op_stats.gpu_runs as u32;
                report.push_str(&format!(
                    "  GPU: {} runs, avg {:.3}ms\n",
                    op_stats.gpu_runs,
                    avg_gpu.as_secs_f64() * 1000.0
                ));

                if op_stats.cpu_runs > 0 {
                    let avg_cpu = op_stats.cpu_time / op_stats.cpu_runs as u32;
                    let speedup = avg_cpu.as_secs_f64() / avg_gpu.as_secs_f64();
                    report.push_str(&format!("  Speedup: {:.1}x\n", speedup));
                }
            }

            if let Some(last_failure) = op_stats.last_gpu_failure {
                report.push_str(&format!(
                    "  Last GPU failure: {:.1}s ago\n",
                    last_failure.elapsed().as_secs_f64()
                ));
            }
        }

        report
    }
}

/// Global executor instance for convenience
static GLOBAL_EXECUTOR: once_cell::sync::Lazy<Arc<GpuExecutor>> =
    once_cell::sync::Lazy::new(|| {
        Arc::new(GpuExecutor::new(Backend::Auto).expect("Failed to create GPU executor"))
    });

/// Get global GPU executor
pub fn global_executor() -> Arc<GpuExecutor> {
    GLOBAL_EXECUTOR.clone()
}

/// Matrix multiplication with automatic GPU/CPU selection
pub fn matmul_auto(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>> {
    let executor = global_executor();

    executor.execute(
        "matmul",
        // GPU operation
        || {
            #[cfg(feature = "cuda")]
            {
                use crate::gpu::simple_gpu::{SimpleGpuContext, SimpleGpuTensor};
                let ctx = SimpleGpuContext::new()?;
                let a_tensor = SimpleGpuTensor::from_cpu(a.to_vec(), vec![m, k])?;
                let b_tensor = SimpleGpuTensor::from_cpu(b.to_vec(), vec![k, n])?;
                let c_tensor = a_tensor.matmul(&b_tensor)?;
                c_tensor.to_cpu()
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("CUDA not available")
            }
        },
        // CPU fallback
        || {
            let mut c = vec![0.0; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for l in 0..k {
                        sum += a[i * k + l] * b[l * n + j];
                    }
                    c[i * n + j] = sum;
                }
            }
            Ok(c)
        },
    )
}

/// ReLU activation with automatic GPU/CPU selection
pub fn relu_auto(data: &[f32]) -> Result<Vec<f32>> {
    let executor = global_executor();

    executor.execute(
        "relu",
        // GPU operation
        || {
            #[cfg(feature = "cuda")]
            {
                use crate::gpu::simple_gpu::{SimpleGpuContext, SimpleGpuTensor};
                let ctx = SimpleGpuContext::new()?;
                let mut tensor = SimpleGpuTensor::from_cpu(data.to_vec(), vec![data.len()])?;
                tensor.relu()?;
                tensor.to_cpu()
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("CUDA not available")
            }
        },
        // CPU fallback
        || {
            Ok(data.iter().map(|&x| x.max(0.0)).collect())
        },
    )
}

/// Softmax with automatic GPU/CPU selection
pub fn softmax_auto(data: &[f32], dim: usize) -> Result<Vec<f32>> {
    let executor = global_executor();

    executor.execute(
        "softmax",
        // GPU operation
        || {
            #[cfg(feature = "cuda")]
            {
                use crate::gpu::simple_gpu::{SimpleGpuContext, SimpleGpuTensor};
                let ctx = SimpleGpuContext::new()?;
                let mut tensor = SimpleGpuTensor::from_cpu(
                    data.to_vec(),
                    vec![data.len() / dim, dim]
                )?;
                tensor.softmax(1)?;
                tensor.to_cpu()
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("CUDA not available")
            }
        },
        // CPU fallback
        || {
            let mut result = data.to_vec();
            let rows = data.len() / dim;

            for i in 0..rows {
                let start = i * dim;
                let end = start + dim;
                let row = &data[start..end];

                // Find max for numerical stability
                let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

                // Compute exp and sum
                let mut exp_sum = 0.0;
                for j in 0..dim {
                    result[start + j] = (row[j] - max_val).exp();
                    exp_sum += result[start + j];
                }

                // Normalize
                for j in 0..dim {
                    result[start + j] /= exp_sum;
                }
            }

            Ok(result)
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_fallback() {
        let executor = GpuExecutor::new(Backend::CpuOnly).unwrap();

        let result = executor.execute(
            "test_op",
            || anyhow::bail!("GPU should not be called"),
            || Ok(42),
        );

        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_auto_fallback() {
        let executor = GpuExecutor::new(Backend::PreferGpu).unwrap();

        // Should fallback to CPU when GPU fails
        let result = executor.execute(
            "test_op",
            || anyhow::bail!("GPU operation failed"),
            || Ok(100),
        );

        assert_eq!(result.unwrap(), 100);
    }

    #[test]
    fn test_matmul_auto() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2
        let c = matmul_auto(&a, &b, 2, 2, 2).unwrap();

        // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        //         = [[19, 22], [43, 50]]
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_relu_auto() {
        let data = vec![-1.0, 0.0, 1.0, -0.5, 2.0];
        let result = relu_auto(&data).unwrap();
        assert_eq!(result, vec![0.0, 0.0, 1.0, 0.0, 2.0]);
    }

    #[test]
    fn test_softmax_auto() {
        let data = vec![1.0, 2.0, 3.0];
        let result = softmax_auto(&data, 3).unwrap();

        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(result[2] > result[1] && result[1] > result[0]);
    }
}