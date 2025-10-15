//! Enhanced Information Theory Metrics
//!
//! Provides mathematically rigorous IT computations with Miller-Madow bias correction.
//!
//! Worker 3 Task: Improve IT quality across all deliverables
//! - Bias-corrected Shannon entropy
//! - Mutual information estimators  
//! - Conditional entropy
//!
//! Constitutional Compliance:
//! - Article I: Worker 3 owns this file
//! - Article III: Comprehensive testing required


/// Enhanced information theory metrics with bias correction
pub struct EnhancedITMetrics {
    /// Whether to apply Miller-Madow bias correction
    bias_correction: bool,
    
    /// k-nearest neighbors for KSG estimator
    ksg_k: usize,
}

impl EnhancedITMetrics {
    /// Create new IT metrics computer with default configuration
    pub fn new() -> Self {
        Self {
            bias_correction: true,
            ksg_k: 3,
        }
    }
    
    /// Configure bias correction
    pub fn with_bias_correction(mut self, enabled: bool) -> Self {
        self.bias_correction = enabled;
        self
    }
    
    /// Configure KSG k parameter
    pub fn with_ksg_k(mut self, k: usize) -> Self {
        self.ksg_k = k;
        self
    }
    
    /// Compute bias-corrected Shannon entropy from histogram
    ///
    /// Applies Miller-Madow correction: H_corrected = H_naive + (K-1)/(2N)
    /// where K = non-zero bins, N = total samples
    pub fn shannon_entropy_from_histogram(&self, histogram: &[usize]) -> f64 {
        let n_samples: usize = histogram.iter().sum();
        if n_samples == 0 {
            return 0.0;
        }
        
        let mut entropy = 0.0;
        let mut k_nonzero = 0;
        
        for &count in histogram {
            if count > 0 {
                let p = count as f64 / n_samples as f64;
                entropy -= p * p.log2();
                k_nonzero += 1;
            }
        }
        
        // Miller-Madow bias correction
        if self.bias_correction && k_nonzero > 1 && n_samples > 0 {
            let correction = (k_nonzero - 1) as f64 / (2.0 * n_samples as f64);
            entropy += correction;
        }
        
        entropy
    }
}

impl Default for EnhancedITMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_shannon_entropy_bias_correction() {
        let metrics = EnhancedITMetrics::new();
        
        // Uniform distribution: maximum entropy
        let uniform_hist = vec![10, 10, 10, 10];
        let entropy = metrics.shannon_entropy_from_histogram(&uniform_hist);
        assert!((entropy - 2.0).abs() < 0.1);
        
        // Single value: minimum entropy
        let single_hist = vec![40, 0, 0, 0];
        let entropy = metrics.shannon_entropy_from_histogram(&single_hist);
        assert!(entropy < 0.1);
    }
    
    #[test]
    fn test_miller_madow_correction() {
        let metrics_with = EnhancedITMetrics::new().with_bias_correction(true);
        let metrics_without = EnhancedITMetrics::new().with_bias_correction(false);
        
        let hist = vec![5, 3, 2, 0];
        
        let h_with = metrics_with.shannon_entropy_from_histogram(&hist);
        let h_without = metrics_without.shannon_entropy_from_histogram(&hist);
        
        // Corrected entropy should be slightly higher
        assert!(h_with > h_without);
        assert!((h_with - h_without) < 0.5);
    }
}
