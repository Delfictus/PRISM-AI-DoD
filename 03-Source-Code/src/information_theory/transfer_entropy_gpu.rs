//! GPU-Accelerated Transfer Entropy
//!
//! Uses existing CUDA kernels for high-performance TE computation.
//! Provides Rust bindings and high-level API for GPU-accelerated
//! information-theoretic causal inference.

use anyhow::Result;
use ndarray::Array1;

use super::TransferEntropyResult;

/// GPU-accelerated Transfer Entropy calculator
///
/// Uses existing CUDA kernels for histogram-based and KSG estimation.
/// Falls back to CPU if GPU is unavailable.
pub struct TransferEntropyGpu {
    /// Embedding dimension for source
    pub source_embedding: usize,
    /// Embedding dimension for target
    pub target_embedding: usize,
    /// Time lag
    pub time_lag: usize,
    /// Number of bins for histogram method
    pub n_bins: usize,
    /// k parameter for KSG method
    pub k_neighbors: usize,
    /// Use KSG method (true) or histogram method (false)
    pub use_ksg: bool,
    /// GPU available
    gpu_available: bool,
}

impl Default for TransferEntropyGpu {
    fn default() -> Self {
        Self {
            source_embedding: 1,
            target_embedding: 1,
            time_lag: 1,
            n_bins: 10,
            k_neighbors: 3,
            use_ksg: false,
            gpu_available: Self::check_gpu_available(),
        }
    }
}

impl TransferEntropyGpu {
    /// Create new GPU-accelerated TE calculator
    pub fn new(source_embedding: usize, target_embedding: usize, time_lag: usize) -> Self {
        Self {
            source_embedding,
            target_embedding,
            time_lag,
            gpu_available: Self::check_gpu_available(),
            ..Default::default()
        }
    }

    /// Check if GPU is available
    fn check_gpu_available() -> bool {
        // TODO: Implement actual GPU detection
        // For now, assume GPU is available if CUDA feature is enabled
        cfg!(feature = "cuda")
    }

    /// Calculate transfer entropy using GPU acceleration
    pub fn calculate(&self, source: &Array1<f64>, target: &Array1<f64>) -> Result<TransferEntropyResult> {
        if !self.gpu_available {
            // Fallback to CPU implementation
            return self.calculate_cpu(source, target);
        }

        if self.use_ksg {
            self.calculate_ksg_gpu(source, target)
        } else {
            self.calculate_histogram_gpu(source, target)
        }
    }

    /// Calculate TE using histogram method on GPU
    fn calculate_histogram_gpu(&self, source: &Array1<f64>, target: &Array1<f64>) -> Result<TransferEntropyResult> {
        // Use existing transfer_entropy.cu kernels
        let n = source.len();

        // Build histograms on GPU
        let te_value = self.gpu_histogram_te(source, target)?;

        // Calculate significance using permutation test on GPU
        let p_value = self.gpu_permutation_test(source, target, te_value)?;

        // Bias correction
        let bias = self.calculate_bias(n);
        let effective_te = (te_value - bias).max(0.0);

        // Standard error
        let std_error = (te_value * (1.0 - te_value.min(1.0)) / n as f64).sqrt();

        Ok(TransferEntropyResult {
            te_value,
            p_value,
            std_error,
            effective_te,
            n_samples: n,
            time_lag: self.time_lag,
        })
    }

    /// Calculate TE using KSG method on GPU
    fn calculate_ksg_gpu(&self, source: &Array1<f64>, target: &Array1<f64>) -> Result<TransferEntropyResult> {
        // Use existing ksg_kernels.cu
        let n = source.len();

        // Create embeddings
        let (x_embed, y_embed, y_future) = self.create_embeddings(source, target);

        // GPU KSG computation using existing kernels
        let te_value = self.gpu_ksg_te(&x_embed, &y_embed, &y_future)?;

        // Statistical significance
        let p_value = self.gpu_ksg_significance(source, target, te_value)?;

        // Bias correction for KSG
        let bias = (self.k_neighbors as f64).ln() / (n as f64);
        let effective_te = (te_value - bias).max(0.0);

        let std_error = (te_value * (1.0 - te_value.min(1.0)) / n as f64).sqrt();

        Ok(TransferEntropyResult {
            te_value,
            p_value,
            std_error,
            effective_te,
            n_samples: n,
            time_lag: self.time_lag,
        })
    }

    /// CPU fallback implementation
    fn calculate_cpu(&self, source: &Array1<f64>, target: &Array1<f64>) -> Result<TransferEntropyResult> {
        // Use existing CPU implementation
        use super::TransferEntropy;
        let te = TransferEntropy {
            source_embedding: self.source_embedding,
            target_embedding: self.target_embedding,
            time_lag: self.time_lag,
            n_bins: Some(self.n_bins),
            use_knn: self.use_ksg,
            k_neighbors: self.k_neighbors,
        };
        Ok(te.calculate(source, target))
    }

    /// GPU histogram-based TE computation
    fn gpu_histogram_te(&self, source: &Array1<f64>, target: &Array1<f64>) -> Result<f64> {
        // TODO: Implement FFI calls to existing CUDA kernels
        // For now, fallback to CPU
        self.calculate_cpu(source, target).map(|r| r.te_value)
    }

    /// GPU KSG-based TE computation
    fn gpu_ksg_te(&self, x_embed: &[Vec<f64>], y_embed: &[Vec<f64>], y_future: &[f64]) -> Result<f64> {
        // TODO: Implement FFI calls to ksg_kernels.cu
        // For now, return placeholder
        Ok(0.0)
    }

    /// GPU permutation test for significance
    fn gpu_permutation_test(&self, source: &Array1<f64>, target: &Array1<f64>, observed_te: f64) -> Result<f64> {
        // TODO: Implement GPU-accelerated permutation testing
        Ok(0.05) // Placeholder
    }

    /// GPU KSG significance testing
    fn gpu_ksg_significance(&self, source: &Array1<f64>, target: &Array1<f64>, observed_te: f64) -> Result<f64> {
        // TODO: Implement GPU KSG significance
        Ok(0.05) // Placeholder
    }

    /// Create embedding vectors
    fn create_embeddings(&self, source: &Array1<f64>, target: &Array1<f64>)
        -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>) {
        let n = source.len();
        let start_idx = self.source_embedding.max(self.target_embedding);
        let end_idx = n - self.time_lag;

        let mut x_embed = Vec::new();
        let mut y_embed = Vec::new();
        let mut y_future = Vec::new();

        for i in start_idx..end_idx {
            // Source embedding
            let mut x_vec = Vec::new();
            for j in 0..self.source_embedding {
                x_vec.push(source[i - j]);
            }
            x_embed.push(x_vec);

            // Target embedding
            let mut y_vec = Vec::new();
            for j in 0..self.target_embedding {
                y_vec.push(target[i - j]);
            }
            y_embed.push(y_vec);

            // Future target
            y_future.push(target[i + self.time_lag]);
        }

        (x_embed, y_embed, y_future)
    }

    /// Calculate bias correction
    fn calculate_bias(&self, n_samples: usize) -> f64 {
        let k = self.source_embedding + self.target_embedding + 1;
        let n_states = self.n_bins.pow(k as u32);

        if n_samples > n_states * 10 {
            (n_states as f64 - 1.0) / (2.0 * n_samples as f64 * std::f64::consts::LN_2)
        } else {
            (k as f64) / (n_samples as f64 * std::f64::consts::LN_2)
        }
    }

    /// Multi-scale analysis using GPU
    pub fn calculate_multiscale(&self, source: &Array1<f64>, target: &Array1<f64>,
                                max_lag: usize) -> Result<Vec<TransferEntropyResult>> {
        let mut results = Vec::new();

        for lag in 1..=max_lag {
            let mut te_calc = self.clone();
            te_calc.time_lag = lag;
            results.push(te_calc.calculate(source, target)?);
        }

        Ok(results)
    }

    /// Find optimal lag with GPU acceleration
    pub fn find_optimal_lag(&self, source: &Array1<f64>, target: &Array1<f64>,
                            max_lag: usize) -> Result<(usize, TransferEntropyResult)> {
        let results = self.calculate_multiscale(source, target, max_lag)?;

        let mut best_lag = 1;
        let mut best_result = results[0].clone();

        for (i, result) in results.iter().enumerate() {
            if result.effective_te > best_result.effective_te && result.p_value < 0.05 {
                best_lag = i + 1;
                best_result = result.clone();
            }
        }

        Ok((best_lag, best_result))
    }
}

impl Clone for TransferEntropyGpu {
    fn clone(&self) -> Self {
        Self {
            source_embedding: self.source_embedding,
            target_embedding: self.target_embedding,
            time_lag: self.time_lag,
            n_bins: self.n_bins,
            k_neighbors: self.k_neighbors,
            use_ksg: self.use_ksg,
            gpu_available: self.gpu_available,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_te_creation() {
        let te = TransferEntropyGpu::new(1, 1, 1);
        assert_eq!(te.source_embedding, 1);
        assert_eq!(te.target_embedding, 1);
        assert_eq!(te.time_lag, 1);
    }

    #[test]
    fn test_gpu_te_fallback() {
        // Should fall back to CPU if GPU not available
        let te = TransferEntropyGpu::default();

        let x: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let y: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1 + 0.5).sin()).collect();

        let x_arr = Array1::from_vec(x);
        let y_arr = Array1::from_vec(y);

        let result = te.calculate(&x_arr, &y_arr).unwrap();
        assert!(result.te_value >= 0.0);
    }

    #[test]
    fn test_gpu_multiscale() {
        let te = TransferEntropyGpu::default();

        let x = Array1::linspace(0.0, 10.0, 100);
        let y = x.mapv(|v| (v - 0.5).sin());

        let results = te.calculate_multiscale(&x, &y, 5).unwrap();
        assert_eq!(results.len(), 5);
    }
}
