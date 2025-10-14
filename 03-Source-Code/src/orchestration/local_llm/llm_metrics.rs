//! Information-Theoretic Metrics for LLM Quality
//!
//! Implements rigorous mathematical metrics for measuring and monitoring
//! LLM inference quality, health, and performance.
//!
//! Key Metrics:
//! - Perplexity: Standard quality measurement (exp of cross-entropy)
//! - KL Divergence: Distribution drift detection
//! - Shannon Entropy: Uncertainty quantification
//! - Cross-Entropy: Information content measurement
//!
//! References:
//! - Perplexity: Lower is better, range [1, vocab_size]
//! - KL Divergence: D_KL(P || Q) = Σ P(x) log(P(x) / Q(x))
//! - Shannon Entropy: H(X) = -Σ P(x) log P(x)

use anyhow::Result;
use std::collections::HashMap;

/// LLM Quality Metrics Calculator
///
/// Provides information-theoretic measurements for model quality,
/// numerical stability, and distribution health.
pub struct LLMMetrics {
    /// Reference distributions for KL-divergence monitoring
    /// Maps layer index to reference probability distribution
    reference_distributions: HashMap<usize, Vec<f32>>,
}

impl LLMMetrics {
    /// Create new metrics calculator
    pub fn new() -> Self {
        Self {
            reference_distributions: HashMap::new(),
        }
    }

    /// Calculate perplexity from logits and target token
    ///
    /// Perplexity = exp(cross_entropy) = exp(-log P(target))
    ///
    /// Interpretation:
    /// - Lower perplexity = better model quality
    /// - Perplexity of 1 = perfect prediction
    /// - Perplexity of vocab_size = random guessing
    ///
    /// # Example
    /// ```
    /// let metrics = LLMMetrics::new();
    /// let perplexity = metrics.perplexity(&logits, target_token)?;
    /// println!("Model perplexity: {:.2}", perplexity);
    /// ```
    pub fn perplexity(&self, logits: &[f32], target_token: i32) -> Result<f32> {
        let log_probs = self.log_softmax(logits);
        let target_idx = target_token as usize;

        if target_idx >= log_probs.len() {
            return Err(anyhow::anyhow!("Target token {} out of bounds", target_token));
        }

        let log_prob = log_probs[target_idx];
        let perplexity = (-log_prob).exp();

        Ok(perplexity)
    }

    /// Calculate sequence perplexity (average over multiple tokens)
    ///
    /// Sequence Perplexity = exp(-(1/N) Σ log P(token_i))
    ///
    /// This is the standard metric for evaluating LLM quality on datasets.
    ///
    /// # Example
    /// ```
    /// let metrics = LLMMetrics::new();
    /// let perplexity = metrics.sequence_perplexity(&logits_sequence, &target_tokens)?;
    /// println!("Average perplexity: {:.2}", perplexity);
    /// ```
    pub fn sequence_perplexity(&self, logits_seq: &[Vec<f32>], targets: &[i32]) -> Result<f32> {
        if logits_seq.len() != targets.len() {
            return Err(anyhow::anyhow!(
                "Logits sequence length {} does not match targets length {}",
                logits_seq.len(),
                targets.len()
            ));
        }

        let mut total_log_prob = 0.0f32;

        for (logits, &target) in logits_seq.iter().zip(targets.iter()) {
            let log_probs = self.log_softmax(logits);
            let target_idx = target as usize;

            if target_idx >= log_probs.len() {
                return Err(anyhow::anyhow!("Target token {} out of bounds", target));
            }

            total_log_prob += log_probs[target_idx];
        }

        let avg_log_prob = total_log_prob / targets.len() as f32;
        let perplexity = (-avg_log_prob).exp();

        Ok(perplexity)
    }

    /// Calculate KL divergence: D_KL(P || Q)
    ///
    /// Measures how much distribution P diverges from distribution Q.
    /// Used for detecting model drift, quantization artifacts, and numerical issues.
    ///
    /// Properties:
    /// - D_KL(P || Q) >= 0 (non-negative)
    /// - D_KL(P || Q) = 0 iff P = Q
    /// - Not symmetric: D_KL(P || Q) != D_KL(Q || P)
    ///
    /// Interpretation:
    /// - KL < 0.1: Distributions are very similar
    /// - KL < 0.5: Distributions are reasonably similar
    /// - KL > 1.0: Significant divergence (warning)
    /// - KL > 2.0: Critical divergence (error)
    ///
    /// # Example
    /// ```
    /// let metrics = LLMMetrics::new();
    /// let kl_div = metrics.kl_divergence(&p_dist, &q_dist)?;
    /// if kl_div > 1.0 {
    ///     println!("Warning: High KL divergence: {:.3}", kl_div);
    /// }
    /// ```
    pub fn kl_divergence(&self, p: &[f32], q: &[f32]) -> Result<f32> {
        if p.len() != q.len() {
            return Err(anyhow::anyhow!(
                "Distribution lengths don't match: {} vs {}",
                p.len(),
                q.len()
            ));
        }

        let mut kl_div = 0.0f32;

        for (pi, qi) in p.iter().zip(q.iter()) {
            // Skip if either probability is too small to avoid numerical issues
            if *pi > 1e-10 && *qi > 1e-10 {
                kl_div += pi * (pi / qi).ln();
            }
        }

        Ok(kl_div)
    }

    /// Calculate Shannon entropy: H(X) = -Σ P(x) log P(x)
    ///
    /// Measures uncertainty/information content of a distribution.
    ///
    /// Properties:
    /// - H(X) >= 0 (non-negative)
    /// - H(X) = 0 for deterministic distribution (single token has prob 1)
    /// - H(X) = log(vocab_size) for uniform distribution (maximum entropy)
    ///
    /// Interpretation (bits):
    /// - Low entropy (< 1 bit): Model is confident
    /// - Medium entropy (2-5 bits): Moderate uncertainty
    /// - High entropy (> 8 bits): High uncertainty (nearly uniform)
    ///
    /// # Example
    /// ```
    /// let metrics = LLMMetrics::new();
    /// let entropy = metrics.entropy(&probs);
    /// println!("Distribution entropy: {:.2} bits", entropy);
    /// ```
    pub fn entropy(&self, probs: &[f32]) -> f32 {
        let mut entropy = 0.0f32;

        for &p in probs.iter() {
            if p > 1e-10 {
                entropy -= p * p.log2();
            }
        }

        entropy
    }

    /// Calculate cross-entropy: H(P, Q) = -Σ P(x) log Q(x)
    ///
    /// Measures the average number of bits needed to encode events from P
    /// using a coding scheme optimized for Q.
    ///
    /// Relationship to KL divergence:
    /// D_KL(P || Q) = H(P, Q) - H(P)
    ///
    /// # Example
    /// ```
    /// let metrics = LLMMetrics::new();
    /// let cross_entropy = metrics.cross_entropy(&true_dist, &model_dist)?;
    /// ```
    pub fn cross_entropy(&self, p: &[f32], q: &[f32]) -> Result<f32> {
        if p.len() != q.len() {
            return Err(anyhow::anyhow!(
                "Distribution lengths don't match: {} vs {}",
                p.len(),
                q.len()
            ));
        }

        let mut cross_ent = 0.0f32;

        for (pi, qi) in p.iter().zip(q.iter()) {
            if *pi > 1e-10 && *qi > 1e-10 {
                cross_ent -= pi * qi.ln();
            }
        }

        Ok(cross_ent)
    }

    /// Store reference distribution for a layer
    ///
    /// Used for KL-divergence monitoring over time.
    /// Call this with a "known good" distribution (e.g., from F16 model).
    pub fn set_reference_distribution(&mut self, layer: usize, distribution: Vec<f32>) {
        self.reference_distributions.insert(layer, distribution);
    }

    /// Check distribution health against reference
    ///
    /// Returns health status based on KL divergence from reference distribution.
    ///
    /// # Example
    /// ```
    /// let mut metrics = LLMMetrics::new();
    /// metrics.set_reference_distribution(0, reference_dist);
    ///
    /// let health = metrics.check_distribution_health(0, &current_logits)?;
    /// match health {
    ///     DistributionHealth::Healthy => println!("✅ Distribution normal"),
    ///     DistributionHealth::Warning(msg) => println!("⚠️  {}", msg),
    ///     DistributionHealth::Critical(msg) => println!("❌ {}", msg),
    /// }
    /// ```
    pub fn check_distribution_health(
        &mut self,
        layer: usize,
        logits: &[f32],
    ) -> Result<DistributionHealth> {
        let current_probs = self.softmax(logits);

        if let Some(ref_probs) = self.reference_distributions.get(&layer) {
            let kl_div = self.kl_divergence(&current_probs, ref_probs)?;

            if kl_div > 2.0 {
                Ok(DistributionHealth::Critical(format!(
                    "High KL divergence: {:.3} (threshold: 2.0)",
                    kl_div
                )))
            } else if kl_div > 0.5 {
                Ok(DistributionHealth::Warning(format!(
                    "Moderate KL divergence: {:.3} (threshold: 0.5)",
                    kl_div
                )))
            } else {
                Ok(DistributionHealth::Healthy)
            }
        } else {
            // First time seeing this layer, store as reference
            self.reference_distributions
                .insert(layer, current_probs.clone());
            Ok(DistributionHealth::Healthy)
        }
    }

    /// Clear all reference distributions
    pub fn clear_references(&mut self) {
        self.reference_distributions.clear();
    }

    /// Softmax (for converting logits to probabilities)
    fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let exps: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();

        let sum: f32 = exps.iter().sum();
        exps.iter().map(|&x| x / sum).collect()
    }

    /// Log-softmax (numerically stable)
    fn log_softmax(&self, logits: &[f32]) -> Vec<f32> {
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let log_sum_exp = logits.iter()
            .map(|&x| (x - max_logit).exp())
            .sum::<f32>()
            .ln();

        logits.iter()
            .map(|&x| x - max_logit - log_sum_exp)
            .collect()
    }
}

impl Default for LLMMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Distribution health status
#[derive(Debug, Clone, PartialEq)]
pub enum DistributionHealth {
    /// Distribution is healthy (KL divergence < 0.5)
    Healthy,
    /// Distribution has moderate divergence (0.5 <= KL < 2.0)
    Warning(String),
    /// Distribution has critical divergence (KL >= 2.0)
    Critical(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perplexity_perfect_prediction() -> Result<()> {
        let metrics = LLMMetrics::new();

        // Logits heavily favoring token 2
        let logits = vec![-10.0, -10.0, 10.0, -10.0];
        let perplexity = metrics.perplexity(&logits, 2)?;

        // Should be close to 1 (perfect prediction)
        assert!(perplexity < 1.1);

        Ok(())
    }

    #[test]
    fn test_perplexity_uniform_distribution() -> Result<()> {
        let metrics = LLMMetrics::new();

        // Uniform logits
        let logits = vec![0.0, 0.0, 0.0, 0.0];
        let perplexity = metrics.perplexity(&logits, 2)?;

        // Should be close to vocab_size (4)
        assert!((perplexity - 4.0).abs() < 0.1);

        Ok(())
    }

    #[test]
    fn test_sequence_perplexity() -> Result<()> {
        let metrics = LLMMetrics::new();

        let logits_seq = vec![
            vec![10.0, -10.0, -10.0],
            vec![-10.0, 10.0, -10.0],
            vec![-10.0, -10.0, 10.0],
        ];
        let targets = vec![0, 1, 2];

        let perplexity = metrics.sequence_perplexity(&logits_seq, &targets)?;

        // All predictions are nearly perfect
        assert!(perplexity < 1.1);

        Ok(())
    }

    #[test]
    fn test_kl_divergence_identical_distributions() -> Result<()> {
        let metrics = LLMMetrics::new();

        let p = vec![0.25, 0.25, 0.25, 0.25];
        let q = vec![0.25, 0.25, 0.25, 0.25];

        let kl_div = metrics.kl_divergence(&p, &q)?;

        // KL divergence should be 0 for identical distributions
        assert!(kl_div.abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_kl_divergence_different_distributions() -> Result<()> {
        let metrics = LLMMetrics::new();

        let p = vec![0.9, 0.1];
        let q = vec![0.1, 0.9];

        let kl_div = metrics.kl_divergence(&p, &q)?;

        // KL divergence should be positive and significant
        assert!(kl_div > 1.0);

        Ok(())
    }

    #[test]
    fn test_entropy_deterministic() {
        let metrics = LLMMetrics::new();

        // Deterministic distribution (all probability on one token)
        let probs = vec![1.0, 0.0, 0.0, 0.0];
        let entropy = metrics.entropy(&probs);

        // Entropy should be 0
        assert!(entropy.abs() < 1e-6);
    }

    #[test]
    fn test_entropy_uniform() {
        let metrics = LLMMetrics::new();

        // Uniform distribution
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let entropy = metrics.entropy(&probs);

        // Entropy should be log2(4) = 2 bits
        assert!((entropy - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_distribution_health_monitoring() -> Result<()> {
        let mut metrics = LLMMetrics::new();

        // Set reference (uniform distribution)
        let reference = vec![0.25, 0.25, 0.25, 0.25];
        metrics.set_reference_distribution(0, reference);

        // Check healthy distribution (similar to reference)
        let healthy_logits = vec![0.0, 0.0, 0.0, 0.0];
        let health = metrics.check_distribution_health(0, &healthy_logits)?;
        assert!(matches!(health, DistributionHealth::Healthy));

        // Check diverged distribution (very different)
        let diverged_logits = vec![10.0, -10.0, -10.0, -10.0];
        let health = metrics.check_distribution_health(0, &diverged_logits)?;
        assert!(matches!(health, DistributionHealth::Critical(_)));

        Ok(())
    }

    #[test]
    fn test_cross_entropy() -> Result<()> {
        let metrics = LLMMetrics::new();

        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];

        let cross_ent = metrics.cross_entropy(&p, &q)?;

        // Cross-entropy of identical distributions equals entropy
        let entropy = metrics.entropy(&p);
        assert!((cross_ent - entropy).abs() < 0.01);

        Ok(())
    }
}
