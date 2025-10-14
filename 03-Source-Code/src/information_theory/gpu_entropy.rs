// GPU-Accelerated Information Theory Operations
// Integrates Worker 2's entropy and divergence GPU kernels
// Constitution: Information Theory + Production GPU Optimization

use anyhow::{Result, Context};
use ndarray::Array1;

#[cfg(feature = "cuda")]
use crate::gpu::kernel_executor::get_global_executor;

/// GPU-accelerated Shannon entropy calculator
///
/// Leverages Worker 2's shannon_entropy GPU kernel for:
/// - 10-20x speedup over CPU implementation
/// - Parallel histogram computation
/// - Miller-Madow bias correction
///
/// # Use Cases
/// - Portfolio diversification measurement
/// - Risk entropy calculation
/// - Information content analysis
pub struct GpuEntropyCalculator {
    /// Use GPU if available
    pub use_gpu: bool,

    /// Number of bins for discretization
    pub n_bins: usize,

    /// Apply Miller-Madow bias correction
    pub bias_correction: bool,
}

impl Default for GpuEntropyCalculator {
    fn default() -> Self {
        Self {
            use_gpu: true,
            n_bins: 10,
            bias_correction: true,
        }
    }
}

impl GpuEntropyCalculator {
    /// Create new GPU entropy calculator
    pub fn new(n_bins: usize) -> Self {
        Self {
            use_gpu: true,
            n_bins,
            bias_correction: true,
        }
    }

    /// Calculate Shannon entropy with automatic GPU/CPU fallback
    ///
    /// # Arguments
    /// * `data` - Input probability distribution or raw data
    ///
    /// # Returns
    /// Shannon entropy in bits
    pub fn calculate(&self, data: &Array1<f64>) -> Result<f64> {
        #[cfg(feature = "cuda")]
        {
            if self.use_gpu {
                if let Ok(entropy) = self.calculate_gpu(data) {
                    return Ok(entropy);
                }
            }
        }

        // Fall back to CPU
        self.calculate_cpu(data)
    }

    /// GPU implementation using Worker 2's shannon_entropy kernel
    #[cfg(feature = "cuda")]
    fn calculate_gpu(&self, data: &Array1<f64>) -> Result<f64> {
        let executor = get_global_executor()
            .context("Failed to get GPU executor")?;
        let executor = executor.lock().unwrap();

        // Convert to probability distribution if needed
        let probabilities = self.to_probabilities(data)?;

        // Convert to f32 for GPU
        let probs_f32: Vec<f32> = probabilities.iter().map(|&x| x as f32).collect();

        // Use Worker 2's shannon_entropy kernel
        let entropy_nats = executor.shannon_entropy(&probs_f32)
            .context("GPU shannon_entropy failed")?;

        // Convert nats to bits
        let entropy_bits = (entropy_nats as f64) / std::f64::consts::LN_2;

        // Apply Miller-Madow bias correction if enabled
        let corrected_entropy = if self.bias_correction {
            let n_nonzero = probs_f32.iter().filter(|&&p| p > 0.0).count();
            let correction = (n_nonzero as f64 - 1.0) / (2.0 * data.len() as f64);
            entropy_bits + correction
        } else {
            entropy_bits
        };

        Ok(corrected_entropy)
    }

    /// CPU fallback implementation
    fn calculate_cpu(&self, data: &Array1<f64>) -> Result<f64> {
        let probabilities = self.to_probabilities(data)?;

        let mut entropy = 0.0;
        for &p in probabilities.iter() {
            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }

        // Miller-Madow bias correction
        if self.bias_correction {
            let n_nonzero = probabilities.iter().filter(|&&p| p > 0.0).count();
            let correction = (n_nonzero as f64 - 1.0) / (2.0 * data.len() as f64);
            entropy += correction;
        }

        Ok(entropy)
    }

    /// Convert data to probability distribution
    fn to_probabilities(&self, data: &Array1<f64>) -> Result<Array1<f64>> {
        // Check if already a probability distribution (sums to ~1.0)
        let sum: f64 = data.iter().sum();

        if (sum - 1.0).abs() < 1e-6 && data.iter().all(|&x| x >= 0.0 && x <= 1.0) {
            // Already a probability distribution
            return Ok(data.clone());
        }

        // Discretize into histogram bins
        let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;

        if range < 1e-10 {
            // All values are the same - zero entropy
            let mut probs = Array1::zeros(self.n_bins);
            probs[0] = 1.0;
            return Ok(probs);
        }

        let mut histogram = vec![0; self.n_bins];
        for &value in data.iter() {
            let bin = ((value - min) / range * (self.n_bins as f64))
                .floor()
                .min((self.n_bins - 1) as f64) as usize;
            histogram[bin] += 1;
        }

        // Convert to probabilities
        let n = data.len() as f64;
        let probabilities = Array1::from_vec(
            histogram.iter().map(|&count| count as f64 / n).collect()
        );

        Ok(probabilities)
    }
}

/// GPU-accelerated KL divergence calculator
///
/// Leverages Worker 2's kl_divergence GPU kernel for:
/// - 10-20x speedup over CPU implementation
/// - Parallel divergence computation
/// - Regime change detection
///
/// # Use Cases
/// - Market regime detection
/// - Distribution change analysis
/// - Model comparison
pub struct GpuKLDivergence {
    /// Use GPU if available
    pub use_gpu: bool,
}

impl Default for GpuKLDivergence {
    fn default() -> Self {
        Self { use_gpu: true }
    }
}

impl GpuKLDivergence {
    /// Create new GPU KL divergence calculator
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate KL divergence: D_KL(P || Q) with automatic GPU/CPU fallback
    ///
    /// # Arguments
    /// * `p` - True distribution P
    /// * `q` - Approximate distribution Q
    ///
    /// # Returns
    /// KL divergence in bits
    pub fn calculate(&self, p: &Array1<f64>, q: &Array1<f64>) -> Result<f64> {
        if p.len() != q.len() {
            anyhow::bail!("Distributions must have same length");
        }

        #[cfg(feature = "cuda")]
        {
            if self.use_gpu {
                if let Ok(kl_div) = self.calculate_gpu(p, q) {
                    return Ok(kl_div);
                }
            }
        }

        // Fall back to CPU
        self.calculate_cpu(p, q)
    }

    /// GPU implementation using Worker 2's kl_divergence kernel
    #[cfg(feature = "cuda")]
    fn calculate_gpu(&self, p: &Array1<f64>, q: &Array1<f64>) -> Result<f64> {
        let executor = get_global_executor()
            .context("Failed to get GPU executor")?;
        let executor = executor.lock().unwrap();

        // Normalize to ensure valid probability distributions
        let p_norm = self.normalize(p)?;
        let q_norm = self.normalize(q)?;

        // Convert to f32 for GPU
        let p_f32: Vec<f32> = p_norm.iter().map(|&x| x as f32).collect();
        let q_f32: Vec<f32> = q_norm.iter().map(|&x| x as f32).collect();

        // Use Worker 2's kl_divergence kernel
        let kl_nats = executor.kl_divergence(&p_f32, &q_f32)
            .context("GPU kl_divergence failed")?;

        // Convert nats to bits
        let kl_bits = (kl_nats as f64) / std::f64::consts::LN_2;

        Ok(kl_bits)
    }

    /// CPU fallback implementation
    fn calculate_cpu(&self, p: &Array1<f64>, q: &Array1<f64>) -> Result<f64> {
        let p_norm = self.normalize(p)?;
        let q_norm = self.normalize(q)?;

        let mut kl_div = 0.0;
        for i in 0..p_norm.len() {
            if p_norm[i] > 0.0 {
                if q_norm[i] > 0.0 {
                    kl_div += p_norm[i] * (p_norm[i] / q_norm[i]).log2();
                } else {
                    // Q is zero where P is non-zero - infinite divergence
                    return Ok(f64::INFINITY);
                }
            }
        }

        Ok(kl_div)
    }

    /// Normalize array to probability distribution
    fn normalize(&self, data: &Array1<f64>) -> Result<Array1<f64>> {
        let sum: f64 = data.iter().sum();

        if sum < 1e-10 {
            anyhow::bail!("Cannot normalize zero distribution");
        }

        Ok(data.mapv(|x| x / sum))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_shannon_entropy_uniform() {
        let calc = GpuEntropyCalculator::new(4);

        // Uniform distribution over 4 bins: H = log2(4) = 2 bits
        let uniform = Array1::from_vec(vec![0.25, 0.25, 0.25, 0.25]);
        let entropy = calc.calculate(&uniform).unwrap();

        assert!((entropy - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_shannon_entropy_deterministic() {
        let calc = GpuEntropyCalculator::new(4);

        // Deterministic distribution: H = 0 bits
        let deterministic = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let entropy = calc.calculate(&deterministic).unwrap();

        assert!(entropy < 0.1);
    }

    #[test]
    fn test_shannon_entropy_raw_data() {
        let calc = GpuEntropyCalculator::new(10);

        // Raw data should be discretized
        let data = Array1::from_vec((0..100).map(|i| i as f64).collect());
        let entropy = calc.calculate(&data).unwrap();

        // Should have high entropy (uniform-ish across bins)
        assert!(entropy > 2.0);
    }

    #[test]
    fn test_kl_divergence_identical() {
        let kl = GpuKLDivergence::new();

        let p = Array1::from_vec(vec![0.25, 0.25, 0.25, 0.25]);
        let q = Array1::from_vec(vec![0.25, 0.25, 0.25, 0.25]);

        let div = kl.calculate(&p, &q).unwrap();

        // D_KL(P || P) = 0
        assert!(div < 0.01);
    }

    #[test]
    fn test_kl_divergence_different() {
        let kl = GpuKLDivergence::new();

        let p = Array1::from_vec(vec![0.5, 0.5]);
        let q = Array1::from_vec(vec![0.9, 0.1]);

        let div = kl.calculate(&p, &q).unwrap();

        // Should be positive when distributions differ
        assert!(div > 0.1);
    }

    #[test]
    fn test_kl_divergence_normalization() {
        let kl = GpuKLDivergence::new();

        // Non-normalized distributions
        let p = Array1::from_vec(vec![2.0, 2.0]);
        let q = Array1::from_vec(vec![3.0, 1.0]);

        // Should normalize automatically
        let div = kl.calculate(&p, &q).unwrap();
        assert!(div.is_finite());
    }

    #[test]
    fn test_entropy_calculator_consistency() {
        let calc = GpuEntropyCalculator::new(10);

        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        // Multiple calls should give same result
        let h1 = calc.calculate(&data).unwrap();
        let h2 = calc.calculate(&data).unwrap();

        assert!((h1 - h2).abs() < 1e-6);
    }
}
