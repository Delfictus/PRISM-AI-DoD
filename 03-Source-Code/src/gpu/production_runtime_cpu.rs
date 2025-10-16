//! Production Runtime with CPU Fallback
//!
//! Demonstrates the production runtime architecture with CPU fallback
//! for systems where PTX loading fails

use std::sync::{Arc, Mutex, OnceLock};
use anyhow::Result;

/// Production Runtime with CPU fallback
pub struct ProductionRuntimeCPU {
    use_gpu: bool,
}

static GLOBAL_RUNTIME: OnceLock<Arc<ProductionRuntimeCPU>> = OnceLock::new();

impl ProductionRuntimeCPU {
    /// Initialize runtime
    pub fn initialize() -> Result<Arc<Self>> {
        Ok(GLOBAL_RUNTIME.get_or_init(|| {
            Arc::new(Self {
                use_gpu: false,  // CPU fallback for now
            })
        }).clone())
    }

    /// Matrix multiply with CPU fallback
    pub fn matmul(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Result<Vec<f32>> {
        eprintln!("  [Production] Using CPU fallback for matmul ({}x{} × {}x{})", m, k, k, n);

        let mut result = vec![0.0f32; m * n];

        // Simple CPU matmul
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                result[i * n + j] = sum;
            }
        }

        Ok(result)
    }

    /// Check if GPU is available
    pub fn has_gpu(&self) -> bool {
        self.use_gpu
    }
}

/// Simple tensor operations
pub struct SimpleTensor {
    data: Vec<f32>,
    size: usize,
    runtime: Arc<ProductionRuntimeCPU>,
}

impl SimpleTensor {
    /// Create from CPU data
    pub fn from_cpu(data: &[f32], runtime: Arc<ProductionRuntimeCPU>) -> Result<Self> {
        Ok(Self {
            data: data.to_vec(),
            size: data.len(),
            runtime,
        })
    }

    /// Get CPU data
    pub fn to_cpu(&self) -> Result<Vec<f32>> {
        Ok(self.data.clone())
    }

    /// Matrix multiply
    pub fn matmul(&self, other: &Self, m: usize, n: usize, k: usize) -> Result<Self> {
        let result = self.runtime.matmul(&self.data, &other.data, m, n, k)?;
        Ok(Self {
            data: result,
            size: m * n,
            runtime: self.runtime.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_fallback_matmul() -> Result<()> {
        let runtime = ProductionRuntimeCPU::initialize()?;

        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2

        let result = runtime.matmul(&a, &b, 2, 2, 2)?;

        // Verify result
        assert!((result[0] - 19.0).abs() < 1e-5); // 1*5 + 2*7 = 19
        assert!((result[1] - 22.0).abs() < 1e-5); // 1*6 + 2*8 = 22
        assert!((result[2] - 43.0).abs() < 1e-5); // 3*5 + 4*7 = 43
        assert!((result[3] - 50.0).abs() < 1e-5); // 3*6 + 4*8 = 50

        eprintln!("✅ CPU fallback matmul successful!");
        Ok(())
    }
}