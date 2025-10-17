//! Sampling Strategies for LLM Text Generation
//!
//! Implements various sampling methods for next-token prediction:
//! - Greedy sampling (deterministic)
//! - Temperature sampling
//! - Top-k sampling
//! - Top-p (nucleus) sampling
//! - Min-p sampling (2025 state-of-the-art)
//!
//! References:
//! - Temperature: Controls randomness in token selection
//! - Top-k: Samples from k most likely tokens
//! - Top-p (Nucleus): Samples from smallest set with cumulative probability >= p
//! - Min-p: Dynamic threshold based on top token probability

use anyhow::Result;
use rand::distributions::{Distribution, WeightedIndex};
use rand::thread_rng;

/// Sampling strategy configuration
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Temperature for scaling logits (0.0 = deterministic, higher = more random)
    /// Typical range: 0.0 - 2.0
    /// Default: 1.0 (no scaling)
    pub temperature: f32,

    /// Top-k: Only sample from k most likely tokens
    /// Set to 0 to disable
    /// Typical range: 1 - 100
    pub top_k: usize,

    /// Top-p (nucleus sampling): Sample from smallest set with cumulative probability >= p
    /// Set to 1.0 to disable
    /// Typical range: 0.9 - 0.95
    pub top_p: f32,

    /// Min-p: Dynamic threshold, filters tokens with prob < (top_token_prob * min_p)
    /// Set to 0.0 to disable
    /// Recommended: 0.05 (as of 2025)
    pub min_p: f32,

    /// Repetition penalty (> 1.0 discourages repetition)
    /// Set to 1.0 to disable
    /// Typical range: 1.0 - 1.5
    pub repetition_penalty: f32,

    /// Entropy-guided sampling weight (NEW - Phase 2)
    /// Balances probability with entropy for information-theoretic sampling
    /// 0.0 = pure probability (standard sampling)
    /// 1.0 = pure entropy (maximum information)
    /// Typical range: 0.0 - 0.3
    pub entropy_weight: f32,

    /// Minimum entropy threshold for entropy-guided sampling
    /// Only apply entropy guidance if distribution entropy > threshold
    /// Typical range: 1.0 - 3.0 bits
    pub min_entropy: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,          // Disabled by default
            top_p: 1.0,        // Disabled by default
            min_p: 0.0,        // Disabled by default
            repetition_penalty: 1.0,
            entropy_weight: 0.0,   // Disabled by default
            min_entropy: 2.0,      // 2 bits threshold
        }
    }
}

impl SamplingConfig {
    /// Create greedy sampling config (always picks most likely token)
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            entropy_weight: 0.0,
            min_entropy: 2.0,
        }
    }

    /// Create standard sampling config (balanced)
    pub fn standard() -> Self {
        Self {
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            min_p: 0.0,
            repetition_penalty: 1.1,
            entropy_weight: 0.0,
            min_entropy: 2.0,
        }
    }

    /// Create creative sampling config (more diverse)
    pub fn creative() -> Self {
        Self {
            temperature: 0.9,
            top_k: 100,
            top_p: 0.95,
            min_p: 0.0,
            repetition_penalty: 1.2,
            entropy_weight: 0.0,
            min_entropy: 2.0,
        }
    }

    /// Create precise sampling config (more conservative)
    pub fn precise() -> Self {
        Self {
            temperature: 0.3,
            top_k: 10,
            top_p: 0.85,
            min_p: 0.0,
            repetition_penalty: 1.0,
            entropy_weight: 0.0,
            min_entropy: 2.0,
        }
    }

    /// Create min-p sampling config (2025 recommended)
    pub fn min_p_recommended() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.05,  // As recommended by major providers
            repetition_penalty: 1.1,
            entropy_weight: 0.0,
            min_entropy: 2.0,
        }
    }

    /// Create entropy-guided sampling config (NEW - Information-theoretic)
    ///
    /// Uses Shannon entropy to guide token selection for maximum information
    /// Balances probability with entropy for optimal information per token
    ///
    /// Benefits:
    /// - Reduces repetition naturally (high entropy = diverse)
    /// - Theoretically optimal token selection
    /// - Novel 2025 sampling strategy
    pub fn entropy_guided() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.95,
            min_p: 0.0,
            repetition_penalty: 1.0,
            entropy_weight: 0.2,  // 20% entropy, 80% probability
            min_entropy: 2.0,     // Only apply if distribution has >2 bits entropy
        }
    }
}

/// Token sampler for LLM generation
pub struct TokenSampler {
    config: SamplingConfig,
}

impl TokenSampler {
    /// Create new token sampler with config
    pub fn new(config: SamplingConfig) -> Self {
        Self { config }
    }

    /// Sample next token from logits
    ///
    /// # Arguments
    /// * `logits` - Raw logit scores for each token in vocabulary
    /// * `previous_tokens` - Previously generated tokens (for repetition penalty)
    ///
    /// # Returns
    /// Sampled token ID
    pub fn sample(&self, logits: &[f32], previous_tokens: &[i32]) -> Result<i32> {
        let mut logits = logits.to_vec();

        // Apply repetition penalty
        if self.config.repetition_penalty != 1.0 {
            self.apply_repetition_penalty(&mut logits, previous_tokens);
        }

        // Apply temperature
        if self.config.temperature != 1.0 && self.config.temperature > 0.0 {
            self.apply_temperature(&mut logits);
        }

        // Convert logits to probabilities (softmax)
        let mut probs = self.softmax(&logits);

        // Apply entropy-guided sampling (NEW - Phase 2)
        if self.config.entropy_weight > 0.0 {
            let entropy = self.calculate_entropy(&probs);
            if entropy >= self.config.min_entropy {
                return self.sample_entropy_guided(&logits, &probs);
            }
        }

        // Apply min-p filtering
        if self.config.min_p > 0.0 {
            self.apply_min_p(&mut probs)?;
        }

        // Apply top-k filtering
        if self.config.top_k > 0 {
            self.apply_top_k(&mut probs);
        }

        // Apply top-p (nucleus) filtering
        if self.config.top_p < 1.0 {
            self.apply_top_p(&mut probs);
        }

        // Handle greedy sampling (temperature = 0 or top_k = 1)
        if self.config.temperature == 0.0 || self.config.top_k == 1 {
            return self.sample_greedy(&probs);
        }

        // Sample from filtered probability distribution
        self.sample_multinomial(&probs)
    }

    /// Apply temperature scaling to logits
    fn apply_temperature(&self, logits: &mut [f32]) {
        let temp = self.config.temperature;
        for logit in logits.iter_mut() {
            *logit /= temp;
        }
    }

    /// Apply repetition penalty
    fn apply_repetition_penalty(&self, logits: &mut [f32], previous_tokens: &[i32]) {
        let penalty = self.config.repetition_penalty;
        for &token_id in previous_tokens {
            if token_id >= 0 && (token_id as usize) < logits.len() {
                let idx = token_id as usize;
                if logits[idx] > 0.0 {
                    logits[idx] /= penalty;
                } else {
                    logits[idx] *= penalty;
                }
            }
        }
    }

    /// Apply min-p filtering (dynamic threshold based on top token)
    fn apply_min_p(&self, probs: &mut [f32]) -> Result<()> {
        // Find maximum probability
        let max_prob = probs.iter().cloned().fold(0.0f32, f32::max);

        // Calculate threshold
        let threshold = max_prob * self.config.min_p;

        // Zero out probabilities below threshold
        for prob in probs.iter_mut() {
            if *prob < threshold {
                *prob = 0.0;
            }
        }

        // Renormalize
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for prob in probs.iter_mut() {
                *prob /= sum;
            }
        } else {
            // If all filtered out, keep only the top token
            let max_idx = probs.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            probs[max_idx] = 1.0;
        }

        Ok(())
    }

    /// Apply top-k filtering
    fn apply_top_k(&self, probs: &mut [f32]) {
        let k = self.config.top_k;

        // Find k-th largest value
        let mut sorted: Vec<f32> = probs.iter().cloned().collect();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());

        let threshold = if k < sorted.len() {
            sorted[k]
        } else {
            0.0
        };

        // Zero out probabilities below k-th largest
        for prob in probs.iter_mut() {
            if *prob < threshold {
                *prob = 0.0;
            }
        }

        // Renormalize
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for prob in probs.iter_mut() {
                *prob /= sum;
            }
        }
    }

    /// Apply top-p (nucleus) filtering
    fn apply_top_p(&self, probs: &mut [f32]) {
        let p = self.config.top_p;

        // Create sorted indices by probability
        let mut indices: Vec<usize> = (0..probs.len()).collect();
        indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

        // Calculate cumulative probabilities
        let mut cumsum = 0.0;
        let mut cutoff_idx = probs.len();

        for (i, &idx) in indices.iter().enumerate() {
            cumsum += probs[idx];
            if cumsum >= p {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Zero out probabilities outside nucleus
        for (i, &idx) in indices.iter().enumerate() {
            if i >= cutoff_idx {
                probs[idx] = 0.0;
            }
        }

        // Renormalize
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for prob in probs.iter_mut() {
                *prob /= sum;
            }
        }
    }

    /// Greedy sampling (argmax)
    fn sample_greedy(&self, probs: &[f32]) -> Result<i32> {
        let token = probs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as i32)
            .unwrap_or(0);
        Ok(token)
    }

    /// Multinomial sampling
    fn sample_multinomial(&self, probs: &[f32]) -> Result<i32> {
        // Filter out zero probabilities and create distribution
        let non_zero: Vec<(usize, f32)> = probs.iter()
            .enumerate()
            .filter(|(_, &p)| p > 0.0)
            .map(|(i, &p)| (i, p))
            .collect();

        if non_zero.is_empty() {
            // Fallback to greedy if no valid probabilities
            return self.sample_greedy(probs);
        }

        let indices: Vec<usize> = non_zero.iter().map(|(i, _)| *i).collect();
        let weights: Vec<f32> = non_zero.iter().map(|(_, p)| *p).collect();

        let dist = WeightedIndex::new(&weights)
            .map_err(|e| anyhow::anyhow!("Failed to create weighted distribution: {}", e))?;

        let sampled_idx = dist.sample(&mut thread_rng());
        Ok(indices[sampled_idx] as i32)
    }

    /// Softmax function (numerically stable)
    fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        // Find max for numerical stability
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Compute exp(logit - max)
        let exps: Vec<f32> = logits.iter()
            .map(|&x| (x - max_logit).exp())
            .collect();

        // Normalize
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|&x| x / sum).collect()
    }

    /// Log-softmax: numerically stable version for log probabilities
    ///
    /// Computes log(softmax(x_i)) = x_i - max(x) - log(Σ exp(x_j - max(x)))
    /// More stable than log(softmax(x)) for extreme logit values
    ///
    /// Used for perplexity calculation and information-theoretic metrics
    pub fn log_softmax(&self, logits: &[f32]) -> Vec<f32> {
        // Subtract max for numerical stability
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Compute log(sum(exp(x - max)))
        let log_sum_exp = logits.iter()
            .map(|&x| (x - max_logit).exp())
            .sum::<f32>()
            .ln();

        // log_softmax(x_i) = x_i - max - log_sum_exp
        logits.iter()
            .map(|&x| x - max_logit - log_sum_exp)
            .collect()
    }

    /// Sample from log probabilities (more numerically stable)
    ///
    /// Used when working in log-space to avoid underflow
    pub fn sample_from_log_probs(&self, log_probs: &[f32]) -> Result<i32> {
        // Convert log probs to probabilities for sampling
        // This is safe because log_softmax already handles numerical stability
        let probs: Vec<f32> = log_probs.iter()
            .map(|&lp| lp.exp())
            .collect();

        self.sample_multinomial(&probs)
    }

    /// Calculate Shannon entropy of probability distribution
    ///
    /// H(X) = -Σ P(x) log₂ P(x)
    ///
    /// Measures uncertainty/information content in bits.
    fn calculate_entropy(&self, probs: &[f32]) -> f32 {
        let mut entropy = 0.0f32;

        for &p in probs.iter() {
            if p > 1e-10 {
                entropy -= p * p.log2();
            }
        }

        entropy
    }

    /// Entropy-guided sampling (NEW - Phase 2)
    ///
    /// Balances probability with entropy for information-theoretic token selection.
    /// Score = (1 - w) * log_prob + w * contribution_to_entropy
    ///
    /// This encourages selecting tokens that:
    /// 1. Have reasonable probability (standard criterion)
    /// 2. Maintain high distribution entropy (diverse future options)
    ///
    /// Benefits:
    /// - Reduces repetition naturally (high entropy = diverse)
    /// - Theoretically optimal information per token
    /// - Novel sampling strategy based on information theory
    fn sample_entropy_guided(&self, logits: &[f32], probs: &[f32]) -> Result<i32> {
        let weight = self.config.entropy_weight;
        let log_probs = self.log_softmax(logits);

        // Calculate global entropy of the distribution
        let global_entropy = self.calculate_entropy(probs);

        // Score each token by its information contribution
        let scores: Vec<f32> = probs.iter().enumerate()
            .map(|(i, &p)| {
                if p < 1e-10 {
                    return f32::NEG_INFINITY;
                }

                // Probability score (negative log prob = surprise)
                let prob_score = log_probs[i];

                // Information contribution: selecting this token maintains diversity
                // Higher probability tokens reduce future entropy more
                // We want tokens that keep options open
                let info_score = global_entropy * (1.0 - p).ln();

                // Combine: (1-w)*probability + w*information
                (1.0 - weight) * prob_score + weight * info_score
            })
            .collect();

        // Find max score for numerical stability
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Convert scores to probabilities
        let score_probs: Vec<f32> = scores.iter()
            .map(|&s| {
                if s == f32::NEG_INFINITY {
                    0.0
                } else {
                    (s - max_score).exp()
                }
            })
            .collect();

        // Normalize
        let sum: f32 = score_probs.iter().sum();
        let normalized_probs: Vec<f32> = score_probs.iter()
            .map(|&p| if sum > 0.0 { p / sum } else { 0.0 })
            .collect();

        // Sample from entropy-guided distribution
        self.sample_multinomial(&normalized_probs)
    }

    /// Get current config
    pub fn config(&self) -> &SamplingConfig {
        &self.config
    }

    /// Update config
    pub fn set_config(&mut self, config: SamplingConfig) {
        self.config = config;
    }
}

impl Default for TokenSampler {
    fn default() -> Self {
        Self::new(SamplingConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let sampler = TokenSampler::default();
        let logits = vec![1.0, 2.0, 3.0];
        let probs = sampler.softmax(&logits);

        // Check probabilities sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check highest logit has highest probability
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_greedy_sampling() -> Result<()> {
        let config = SamplingConfig::greedy();
        let sampler = TokenSampler::new(config);

        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let token = sampler.sample(&logits, &[])?;

        // Should always pick token with highest logit
        assert_eq!(token, 3);

        Ok(())
    }

    #[test]
    fn test_temperature_effect() {
        let sampler = TokenSampler::default();

        let mut logits = vec![1.0, 2.0, 3.0];
        let temp_sampler = TokenSampler::new(SamplingConfig {
            temperature: 2.0,
            ..Default::default()
        });

        let original_probs = sampler.softmax(&logits);
        temp_sampler.apply_temperature(&mut logits);
        let temp_probs = sampler.softmax(&logits);

        // Higher temperature should make distribution more uniform
        let original_entropy = -original_probs.iter()
            .map(|&p| if p > 0.0 { p * p.log2() } else { 0.0 })
            .sum::<f32>();
        let temp_entropy = -temp_probs.iter()
            .map(|&p| if p > 0.0 { p * p.log2() } else { 0.0 })
            .sum::<f32>();

        assert!(temp_entropy > original_entropy);
    }

    #[test]
    fn test_config_presets() {
        let greedy = SamplingConfig::greedy();
        assert_eq!(greedy.temperature, 0.0);
        assert_eq!(greedy.top_k, 1);

        let standard = SamplingConfig::standard();
        assert_eq!(standard.temperature, 0.7);
        assert_eq!(standard.top_k, 50);

        let creative = SamplingConfig::creative();
        assert_eq!(creative.temperature, 0.9);
        assert_eq!(creative.top_p, 0.95);

        let precise = SamplingConfig::precise();
        assert_eq!(precise.temperature, 0.3);
        assert_eq!(precise.top_k, 10);

        let min_p = SamplingConfig::min_p_recommended();
        assert_eq!(min_p.min_p, 0.05);

        let entropy_guided = SamplingConfig::entropy_guided();
        assert_eq!(entropy_guided.entropy_weight, 0.2);
        assert_eq!(entropy_guided.min_entropy, 2.0);
    }

    #[test]
    fn test_entropy_calculation() {
        let sampler = TokenSampler::default();

        // Deterministic distribution (entropy = 0)
        let det_probs = vec![1.0, 0.0, 0.0, 0.0];
        let det_entropy = sampler.calculate_entropy(&det_probs);
        assert!(det_entropy.abs() < 0.01);

        // Uniform distribution (entropy = log2(4) = 2)
        let unif_probs = vec![0.25, 0.25, 0.25, 0.25];
        let unif_entropy = sampler.calculate_entropy(&unif_probs);
        assert!((unif_entropy - 2.0).abs() < 0.01);

        // Mixed distribution (entropy between 0 and 2)
        let mixed_probs = vec![0.5, 0.3, 0.15, 0.05];
        let mixed_entropy = sampler.calculate_entropy(&mixed_probs);
        assert!(mixed_entropy > 0.0 && mixed_entropy < 2.0);
    }

    #[test]
    fn test_entropy_guided_sampling() -> Result<()> {
        let config = SamplingConfig::entropy_guided();
        let sampler = TokenSampler::new(config);

        // High entropy distribution (should trigger entropy guidance)
        let logits = vec![1.0, 1.1, 0.9, 1.05, 0.95];  // Nearly uniform
        let token = sampler.sample(&logits, &[])?;

        // Should sample from valid token range
        assert!(token >= 0 && token < 5);

        Ok(())
    }
}
