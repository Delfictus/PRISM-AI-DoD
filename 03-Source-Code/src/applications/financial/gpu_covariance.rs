// GPU-Accelerated Covariance Matrix Calculation
// Integrates with Worker 2's GPU infrastructure for portfolio optimization
// Constitution: Financial Application + GPU Enhancement

use anyhow::{Result, Context};
use ndarray::{Array1, Array2};

#[cfg(feature = "cuda")]
use crate::gpu::kernel_executor::get_global_executor;

/// GPU-accelerated covariance matrix calculator
///
/// Leverages Worker 2's Tensor Core acceleration for:
/// - 8x speedup using WMMA (Warp Matrix Multiply-Accumulate)
/// - Efficient large-scale portfolio covariance computation
/// - FP16/FP32 mixed precision for optimal performance
pub struct GpuCovarianceCalculator {
    /// Use GPU if available
    pub use_gpu: bool,

    /// Use Tensor Cores (WMMA) for large matrices
    pub use_tensor_cores: bool,

    /// Minimum size to use Tensor Cores (below this, use regular GPU)
    pub tensor_core_threshold: usize,
}

impl Default for GpuCovarianceCalculator {
    fn default() -> Self {
        Self {
            use_gpu: true,
            use_tensor_cores: true,
            tensor_core_threshold: 256, // Use Tensor Cores for 256x256+ matrices
        }
    }
}

impl GpuCovarianceCalculator {
    /// Create new GPU covariance calculator
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with specific configuration
    pub fn with_config(use_gpu: bool, use_tensor_cores: bool) -> Self {
        Self {
            use_gpu,
            use_tensor_cores,
            tensor_core_threshold: 256,
        }
    }

    /// Calculate covariance matrix using GPU acceleration
    ///
    /// Cov(X, Y) = E[(X - E[X])(Y - E[Y])] / (n-1)
    ///
    /// For a data matrix of shape (n_samples, n_assets), computes
    /// the (n_assets, n_assets) covariance matrix.
    ///
    /// # Arguments
    /// * `returns` - Asset returns matrix [n_samples × n_assets]
    ///
    /// # Returns
    /// Covariance matrix [n_assets × n_assets]
    pub fn calculate(&self, returns: &Array2<f64>) -> Result<Array2<f64>> {
        let (n_samples, n_assets) = returns.dim();

        if n_samples < 2 {
            anyhow::bail!("Need at least 2 samples to calculate covariance");
        }

        #[cfg(feature = "cuda")]
        {
            if self.use_gpu {
                // Try GPU acceleration
                if let Ok(cov) = self.calculate_gpu(returns, n_samples, n_assets) {
                    return Ok(cov);
                }
            }
        }

        // Fall back to CPU implementation
        self.calculate_cpu(returns, n_samples, n_assets)
    }

    /// GPU implementation using Worker 2's matrix multiply kernels
    #[cfg(feature = "cuda")]
    fn calculate_gpu(&self, returns: &Array2<f64>, n_samples: usize, n_assets: usize) -> Result<Array2<f64>> {
        let executor = get_global_executor()
            .context("Failed to get GPU executor from Worker 2")?;
        let executor = executor.lock().unwrap();

        // Center the data: X_centered = X - mean(X)
        let means = returns.mean_axis(ndarray::Axis(0))
            .context("Failed to calculate means")?;

        let mut centered = Array2::zeros((n_samples, n_assets));
        for i in 0..n_samples {
            for j in 0..n_assets {
                centered[[i, j]] = returns[[i, j]] - means[j];
            }
        }

        // Convert to f32 for GPU
        let centered_f32: Vec<f32> = centered.iter().map(|&x| x as f32).collect();

        // Decide whether to use Tensor Cores
        let use_wmma = self.use_tensor_cores && n_assets >= self.tensor_core_threshold;

        // Compute X^T X using GPU matrix multiply
        // Cov = (1/(n-1)) * X^T X
        let xt_x = if use_wmma {
            // Use Tensor Core WMMA for large matrices (8x speedup)
            executor.tensor_core_matmul_wmma(
                &centered_f32,
                &centered_f32,
                n_assets,
                n_samples,
                n_assets
            )?
        } else {
            // Use regular GPU matrix multiply
            executor.matrix_multiply(
                &centered_f32,
                &centered_f32,
                n_assets,
                n_samples,
                n_assets
            )?
        };

        // Convert back to f64 and scale by 1/(n-1)
        let scale = 1.0 / ((n_samples - 1) as f64);
        let mut covariance = Array2::zeros((n_assets, n_assets));

        for i in 0..n_assets {
            for j in 0..n_assets {
                covariance[[i, j]] = (xt_x[i * n_assets + j] as f64) * scale;
            }
        }

        Ok(covariance)
    }

    /// CPU fallback implementation
    fn calculate_cpu(&self, returns: &Array2<f64>, n_samples: usize, n_assets: usize) -> Result<Array2<f64>> {
        // Calculate means for each asset
        let means = returns.mean_axis(ndarray::Axis(0))
            .context("Failed to calculate means")?;

        // Center the data
        let mut centered = Array2::zeros((n_samples, n_assets));
        for i in 0..n_samples {
            for j in 0..n_assets {
                centered[[i, j]] = returns[[i, j]] - means[j];
            }
        }

        // Compute covariance: Cov = (1/(n-1)) * X^T X
        let xt = centered.t();
        let xt_x = xt.dot(&centered);
        let covariance = xt_x / ((n_samples - 1) as f64);

        Ok(covariance)
    }

    /// Calculate correlation matrix from covariance matrix
    ///
    /// Corr(i,j) = Cov(i,j) / sqrt(Var(i) * Var(j))
    pub fn covariance_to_correlation(&self, covariance: &Array2<f64>) -> Result<Array2<f64>> {
        let n = covariance.nrows();
        let mut correlation = Array2::zeros((n, n));

        // Extract standard deviations (sqrt of diagonal)
        let mut std_devs = Array1::zeros(n);
        for i in 0..n {
            std_devs[i] = covariance[[i, i]].sqrt();
            if std_devs[i] < 1e-10 {
                anyhow::bail!("Zero variance detected for asset {}", i);
            }
        }

        // Compute correlation matrix
        for i in 0..n {
            for j in 0..n {
                correlation[[i, j]] = covariance[[i, j]] / (std_devs[i] * std_devs[j]);
            }
        }

        Ok(correlation)
    }

    /// Batch covariance calculation for rolling windows
    ///
    /// Computes covariance matrices for multiple time windows efficiently
    ///
    /// # Arguments
    /// * `returns` - Full time series of returns [n_total_samples × n_assets]
    /// * `window_size` - Size of rolling window
    /// * `step_size` - Step between windows (1 = every time step)
    ///
    /// # Returns
    /// Vec of covariance matrices, one per window
    pub fn calculate_rolling(&self, returns: &Array2<f64>, window_size: usize, step_size: usize) -> Result<Vec<Array2<f64>>> {
        let (n_samples, _n_assets) = returns.dim();

        if window_size > n_samples {
            anyhow::bail!("Window size ({}) exceeds number of samples ({})", window_size, n_samples);
        }

        let mut covariances = Vec::new();
        let mut start = 0;

        while start + window_size <= n_samples {
            let window = returns.slice(ndarray::s![start..start+window_size, ..]).to_owned();
            let cov = self.calculate(&window)?;
            covariances.push(cov);
            start += step_size;
        }

        Ok(covariances)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, arr2};

    #[test]
    fn test_covariance_2x2() {
        // Simple 2-asset, 3-sample case
        let returns = arr2(&[
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
        ]);

        let calc = GpuCovarianceCalculator::new();
        let cov = calc.calculate(&returns).unwrap();

        // Expected covariance for perfect linear relationship
        assert_eq!(cov.dim(), (2, 2));

        // Variance of first asset
        assert!((cov[[0, 0]] - 1.0).abs() < 0.01, "Var(X) = {}", cov[[0, 0]]);

        // Covariance should be positive for positively correlated assets
        assert!(cov[[0, 1]] > 0.0, "Cov(X,Y) = {}", cov[[0, 1]]);

        // Symmetry
        assert!((cov[[0, 1]] - cov[[1, 0]]).abs() < 1e-10);
    }

    #[test]
    fn test_covariance_independent() {
        // Two independent random-like series
        let returns = arr2(&[
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 2.0],
            [4.0, 1.0],
            [5.0, 2.0],
        ]);

        let calc = GpuCovarianceCalculator::new();
        let cov = calc.calculate(&returns).unwrap();

        // Diagonal elements (variances) should be positive
        assert!(cov[[0, 0]] > 0.0);
        assert!(cov[[1, 1]] > 0.0);

        // Symmetry
        assert!((cov[[0, 1]] - cov[[1, 0]]).abs() < 1e-10);
    }

    #[test]
    fn test_correlation_from_covariance() {
        let cov = arr2(&[
            [4.0, 2.0],
            [2.0, 9.0],
        ]);

        let calc = GpuCovarianceCalculator::new();
        let corr = calc.covariance_to_correlation(&cov).unwrap();

        // Diagonal should be 1.0
        assert!((corr[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((corr[[1, 1]] - 1.0).abs() < 1e-10);

        // Correlation should be in [-1, 1]
        assert!(corr[[0, 1]].abs() <= 1.0);

        // Symmetry
        assert!((corr[[0, 1]] - corr[[1, 0]]).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_covariance() {
        let returns = Array2::from_shape_fn((20, 3), |(i, j)| {
            ((i + j) as f64 * 0.1).sin()
        });

        let calc = GpuCovarianceCalculator::new();
        let rolling_covs = calc.calculate_rolling(&returns, 10, 5).unwrap();

        // Should have 3 windows: [0:10], [5:15], [10:20]
        assert_eq!(rolling_covs.len(), 3);

        for cov in &rolling_covs {
            assert_eq!(cov.dim(), (3, 3));

            // All diagonal elements should be non-negative
            for i in 0..3 {
                assert!(cov[[i, i]] >= 0.0);
            }
        }
    }

    #[test]
    fn test_cpu_fallback() {
        let returns = arr2(&[
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
        ]);

        // Force CPU calculation
        let calc = GpuCovarianceCalculator::with_config(false, false);
        let cov = calc.calculate(&returns).unwrap();

        assert_eq!(cov.dim(), (3, 3));

        // Verify symmetry
        for i in 0..3 {
            for j in 0..3 {
                assert!((cov[[i, j]] - cov[[j, i]]).abs() < 1e-10);
            }
        }
    }
}
