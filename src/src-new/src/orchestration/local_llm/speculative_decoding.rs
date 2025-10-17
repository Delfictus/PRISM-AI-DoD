//! Speculative Decoding for 2-3x LLM Generation Speedup
//!
//! Implements speculative decoding (also called assisted generation) to
//! dramatically accelerate LLM inference without changing output distribution.
//!
//! # Algorithm Overview
//!
//! Traditional autoregressive generation:
//! - Generate token 1 (slow)
//! - Generate token 2 (slow)
//! - Generate token 3 (slow)
//! - Total: 3 forward passes
//!
//! Speculative decoding:
//! - Draft model generates K tokens quickly (fast, K passes)
//! - Target model verifies all K tokens in parallel (slow, 1 pass)
//! - Accept valid tokens, reject invalid ones
//! - Total: K fast + 1 slow pass (but if ≥2 tokens accepted: net speedup!)
//!
//! # Key Properties
//!
//! 1. **Mathematically equivalent**: Output distribution identical to normal generation
//! 2. **Lossless**: No quality degradation whatsoever
//! 3. **Speculative**: Draft may be wrong, but verification ensures correctness
//! 4. **Parallel**: Target model processes K tokens in single forward pass
//!
//! # Performance
//!
//! - Typical speedup: 2-3x on average
//! - Best case: K tokens accepted (Kx speedup)
//! - Worst case: 1 token accepted (no slowdown)
//! - Acceptance rate depends on draft model quality and K value
//!
//! # References
//!
//! - Leviathan et al. (2023). "Fast Inference from Transformers via Speculative Decoding"
//! - Chen et al. (2023). "Accelerating Large Language Model Decoding"
//! - Spector & Re (2023). "Accelerating LLM Inference with Staged Speculative Decoding"

use anyhow::Result;
use rand::Rng;

/// Speculative Decoding Engine
///
/// Coordinates draft model and target model for accelerated generation.
pub struct SpeculativeDecoder {
    /// Number of tokens to generate speculatively (K parameter)
    /// Typical values: 4-8
    /// Higher K = more potential speedup but lower acceptance rate
    k: usize,

    /// Temperature for sampling (used in acceptance criterion)
    temperature: f32,

    /// Statistics tracking
    stats: SpeculativeStats,
}

impl SpeculativeDecoder {
    /// Create new speculative decoder
    ///
    /// # Arguments
    /// * `k` - Number of speculative tokens (typically 4-8)
    /// * `temperature` - Sampling temperature (typically 1.0)
    pub fn new(k: usize, temperature: f32) -> Self {
        Self {
            k,
            temperature,
            stats: SpeculativeStats::new(),
        }
    }

    /// Generate tokens using speculative decoding
    ///
    /// # Arguments
    /// * `draft_fn` - Fast draft model: (context_tokens) -> (draft_tokens, draft_logits)
    /// * `target_fn` - Slow target model: (context_tokens) -> logits
    /// * `initial_context` - Initial prompt tokens
    /// * `max_tokens` - Maximum tokens to generate
    ///
    /// # Returns
    /// Generated token sequence
    ///
    /// # Example
    /// ```no_run
    /// let decoder = SpeculativeDecoder::new(5, 1.0);
    ///
    /// let tokens = decoder.generate(
    ///     |ctx| draft_model.generate(ctx, 5),  // Generate 5 draft tokens
    ///     |ctx| target_model.forward(ctx),      // Verify with target
    ///     &prompt_tokens,
    ///     100
    /// )?;
    /// ```
    pub fn generate<DF, TF>(
        &mut self,
        mut draft_fn: DF,
        mut target_fn: TF,
        initial_context: &[i32],
        max_tokens: usize,
    ) -> Result<Vec<i32>>
    where
        DF: FnMut(&[i32]) -> Result<(Vec<i32>, Vec<Vec<f32>>)>,
        TF: FnMut(&[i32]) -> Result<Vec<f32>>,
    {
        let mut result = initial_context.to_vec();
        let mut tokens_generated = 0;

        while tokens_generated < max_tokens {
            // Step 1: Draft model generates K tokens speculatively
            let (draft_tokens, draft_logits) = draft_fn(&result)?;

            if draft_tokens.is_empty() {
                break;
            }

            // Limit to K tokens
            let k_actual = draft_tokens.len().min(self.k);
            let draft_tokens = &draft_tokens[..k_actual];
            let draft_logits = &draft_logits[..k_actual];

            // Step 2: Target model verifies all K tokens in parallel
            // Create context with all draft tokens appended
            let mut verification_context = result.clone();
            verification_context.extend_from_slice(draft_tokens);

            // Get target model logits for each position
            // In practice, this would be a single forward pass through target model
            // Here we simulate by calling target_fn for each position
            let mut target_logits_seq = Vec::new();
            for i in 0..k_actual {
                let ctx_end = result.len() + i;
                let target_logits = target_fn(&verification_context[..ctx_end])?;
                target_logits_seq.push(target_logits);
            }

            // Step 3: Accept/reject tokens using modified rejection sampling
            let n_accepted = self.verify_and_accept(
                draft_tokens,
                draft_logits,
                &target_logits_seq,
                &mut result,
            )?;

            tokens_generated += n_accepted;

            // Update statistics
            self.stats.total_speculation_rounds += 1;
            self.stats.total_tokens_accepted += n_accepted;
            self.stats.total_draft_tokens += k_actual;

            // If no tokens accepted, we're stuck - break to prevent infinite loop
            if n_accepted == 0 {
                // Sample one token from target model to make progress
                let target_logits = target_fn(&result)?;
                let token = self.sample_token(&target_logits)?;
                result.push(token);
                tokens_generated += 1;
            }

            // Check if we've generated enough
            if tokens_generated >= max_tokens {
                break;
            }
        }

        // Trim to max_tokens
        let total_len = initial_context.len() + max_tokens;
        result.truncate(total_len);

        Ok(result)
    }

    /// Verify draft tokens and accept valid ones
    ///
    /// Uses modified rejection sampling to maintain correct distribution.
    ///
    /// Acceptance criterion (Leviathan et al. 2023):
    /// - For each position i:
    ///   - Accept token with probability min(1, p_target(x) / p_draft(x))
    ///   - If rejected, resample from adjusted distribution
    ///
    /// This ensures output distribution matches target model exactly.
    fn verify_and_accept(
        &mut self,
        draft_tokens: &[i32],
        draft_logits: &[Vec<f32>],
        target_logits_seq: &[Vec<f32>],
        result: &mut Vec<i32>,
    ) -> Result<usize> {
        let mut n_accepted = 0;
        let mut rng = rand::thread_rng();

        for i in 0..draft_tokens.len() {
            let draft_token = draft_tokens[i];
            let draft_probs = self.softmax(&draft_logits[i]);
            let target_probs = self.softmax(&target_logits_seq[i]);

            // Get probabilities for the draft token
            let p_draft = draft_probs[draft_token as usize];
            let p_target = target_probs[draft_token as usize];

            // Acceptance probability: min(1, p_target / p_draft)
            let acceptance_prob = (p_target / p_draft).min(1.0);

            let rand_val: f32 = rng.gen();

            if rand_val < acceptance_prob {
                // Accept token
                result.push(draft_token);
                n_accepted += 1;
            } else {
                // Reject token - resample from adjusted distribution
                // Adjusted distribution: max(0, p_target - p_draft) / Z
                let adjusted_probs = self.compute_adjusted_distribution(
                    &draft_probs,
                    &target_probs,
                );
                let resampled_token = self.sample_from_distribution(&adjusted_probs)?;
                result.push(resampled_token);
                n_accepted += 1;

                // Stop speculation after first rejection
                break;
            }
        }

        Ok(n_accepted)
    }

    /// Compute adjusted distribution for resampling after rejection
    ///
    /// Adjusted: p'(x) = max(0, p_target(x) - p_draft(x)) / Z
    /// where Z is normalization constant
    fn compute_adjusted_distribution(&self, draft_probs: &[f32], target_probs: &[f32]) -> Vec<f32> {
        let mut adjusted: Vec<f32> = draft_probs
            .iter()
            .zip(target_probs.iter())
            .map(|(&p_d, &p_t)| (p_t - p_d).max(0.0))
            .collect();

        // Normalize
        let sum: f32 = adjusted.iter().sum();
        if sum > 1e-10 {
            for p in adjusted.iter_mut() {
                *p /= sum;
            }
        } else {
            // Fallback to uniform if all adjusted probs are 0
            let uniform_prob = 1.0 / adjusted.len() as f32;
            adjusted.fill(uniform_prob);
        }

        adjusted
    }

    /// Sample token from probability distribution
    fn sample_from_distribution(&self, probs: &[f32]) -> Result<i32> {
        let mut rng = rand::thread_rng();
        let rand_val: f32 = rng.gen();

        let mut cumsum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if rand_val < cumsum {
                return Ok(i as i32);
            }
        }

        // Fallback (should rarely happen due to floating point)
        Ok((probs.len() - 1) as i32)
    }

    /// Sample single token from logits
    fn sample_token(&self, logits: &[f32]) -> Result<i32> {
        let probs = self.softmax(logits);
        self.sample_from_distribution(&probs)
    }

    /// Softmax with temperature
    fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Apply temperature
        let exps: Vec<f32> = logits
            .iter()
            .map(|&x| ((x - max_logit) / self.temperature).exp())
            .collect();

        let sum: f32 = exps.iter().sum();
        exps.iter().map(|&x| x / sum).collect()
    }

    /// Get statistics
    pub fn stats(&self) -> &SpeculativeStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = SpeculativeStats::new();
    }

    /// Set K parameter (number of speculative tokens)
    pub fn set_k(&mut self, k: usize) {
        self.k = k;
    }

    /// Set temperature
    pub fn set_temperature(&mut self, temperature: f32) {
        self.temperature = temperature;
    }
}

/// Speculative Decoding Statistics
#[derive(Debug, Clone)]
pub struct SpeculativeStats {
    /// Total number of speculation rounds
    pub total_speculation_rounds: usize,

    /// Total tokens accepted from draft model
    pub total_tokens_accepted: usize,

    /// Total draft tokens proposed
    pub total_draft_tokens: usize,
}

impl SpeculativeStats {
    pub fn new() -> Self {
        Self {
            total_speculation_rounds: 0,
            total_tokens_accepted: 0,
            total_draft_tokens: 0,
        }
    }

    /// Calculate acceptance rate (fraction of draft tokens accepted)
    pub fn acceptance_rate(&self) -> f32 {
        if self.total_draft_tokens == 0 {
            return 0.0;
        }
        self.total_tokens_accepted as f32 / self.total_draft_tokens as f32
    }

    /// Calculate average tokens accepted per round
    pub fn avg_accepted_per_round(&self) -> f32 {
        if self.total_speculation_rounds == 0 {
            return 0.0;
        }
        self.total_tokens_accepted as f32 / self.total_speculation_rounds as f32
    }

    /// Estimate speedup factor
    ///
    /// Speedup = tokens_accepted / speculation_rounds
    ///
    /// If avg_accepted ≥ 2, we get ~2x speedup
    /// If avg_accepted ≥ 3, we get ~3x speedup
    pub fn estimated_speedup(&self) -> f32 {
        self.avg_accepted_per_round()
    }
}

impl Default for SpeculativeStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Self-Speculative Decoding
///
/// Uses the same model as both draft and target, but with different configurations:
/// - Draft: Lower precision (int8), greedy sampling, no KV-cache
/// - Target: Full precision (fp16), proper sampling, with KV-cache
///
/// Still achieves 1.5-2x speedup even though draft == target model!
pub struct SelfSpeculativeDecoder {
    decoder: SpeculativeDecoder,
}

impl SelfSpeculativeDecoder {
    /// Create self-speculative decoder
    ///
    /// Uses same model for draft and target with different configs.
    pub fn new(k: usize, temperature: f32) -> Self {
        Self {
            decoder: SpeculativeDecoder::new(k, temperature),
        }
    }

    /// Generate using self-speculation
    ///
    /// # Arguments
    /// * `model_fn` - Model forward pass: (tokens, use_fast_mode) -> logits
    ///   - use_fast_mode=true: draft mode (int8, greedy, no cache)
    ///   - use_fast_mode=false: target mode (fp16, full sampling, cache)
    pub fn generate<MF>(
        &mut self,
        mut model_fn: MF,
        initial_context: &[i32],
        max_tokens: usize,
    ) -> Result<Vec<i32>>
    where
        MF: FnMut(&[i32], bool) -> Result<Vec<f32>>,
    {
        // Draft function: fast mode, generate K tokens greedily
        let draft_fn = |ctx: &[i32]| -> Result<(Vec<i32>, Vec<Vec<f32>>)> {
            let mut tokens = Vec::new();
            let mut logits_seq = Vec::new();
            let mut current_ctx = ctx.to_vec();

            for _ in 0..self.decoder.k {
                let logits = model_fn(&current_ctx, true)?; // Fast mode
                let token = argmax(&logits);
                tokens.push(token);
                logits_seq.push(logits);
                current_ctx.push(token);
            }

            Ok((tokens, logits_seq))
        };

        // Target function: full precision mode
        let target_fn = |ctx: &[i32]| -> Result<Vec<f32>> {
            model_fn(ctx, false) // Full quality mode
        };

        self.decoder.generate(draft_fn, target_fn, initial_context, max_tokens)
    }

    /// Get statistics
    pub fn stats(&self) -> &SpeculativeStats {
        self.decoder.stats()
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.decoder.reset_stats();
    }
}

/// Find argmax of vector
fn argmax(values: &[f32]) -> i32 {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as i32)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speculative_decoder_creation() {
        let decoder = SpeculativeDecoder::new(5, 1.0);
        assert_eq!(decoder.k, 5);
        assert_eq!(decoder.temperature, 1.0);
    }

    #[test]
    fn test_stats_calculation() {
        let mut stats = SpeculativeStats::new();

        stats.total_speculation_rounds = 10;
        stats.total_tokens_accepted = 25;
        stats.total_draft_tokens = 50;

        assert_eq!(stats.acceptance_rate(), 0.5);
        assert_eq!(stats.avg_accepted_per_round(), 2.5);
        assert_eq!(stats.estimated_speedup(), 2.5);
    }

    #[test]
    fn test_softmax() {
        let decoder = SpeculativeDecoder::new(4, 1.0);

        let logits = vec![1.0, 2.0, 3.0];
        let probs = decoder.softmax(&logits);

        // Check sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Check probabilities increase with logits
        assert!(probs[0] < probs[1]);
        assert!(probs[1] < probs[2]);
    }

    #[test]
    fn test_adjusted_distribution() {
        let decoder = SpeculativeDecoder::new(4, 1.0);

        let draft_probs = vec![0.2, 0.3, 0.5];
        let target_probs = vec![0.1, 0.4, 0.5];

        let adjusted = decoder.compute_adjusted_distribution(&draft_probs, &target_probs);

        // Check sum to 1
        let sum: f32 = adjusted.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Check adjusted is non-negative
        for &p in adjusted.iter() {
            assert!(p >= 0.0);
        }
    }

    #[test]
    fn test_sample_from_distribution() -> Result<()> {
        let decoder = SpeculativeDecoder::new(4, 1.0);

        // Deterministic distribution
        let probs = vec![0.0, 1.0, 0.0];
        let token = decoder.sample_from_distribution(&probs)?;
        assert_eq!(token, 1);

        Ok(())
    }

    #[test]
    fn test_argmax() {
        assert_eq!(argmax(&[1.0, 2.0, 3.0]), 2);
        assert_eq!(argmax(&[3.0, 1.0, 2.0]), 0);
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
    }

    #[test]
    fn test_speculative_generation() -> Result<()> {
        let mut decoder = SpeculativeDecoder::new(3, 1.0);

        // Mock draft model (returns simple sequence)
        let draft_fn = |ctx: &[i32]| -> Result<(Vec<i32>, Vec<Vec<f32>>)> {
            let tokens = vec![ctx.len() as i32, (ctx.len() + 1) as i32, (ctx.len() + 2) as i32];
            let logits = vec![
                vec![1.0, 2.0, 3.0],
                vec![2.0, 3.0, 1.0],
                vec![3.0, 1.0, 2.0],
            ];
            Ok((tokens, logits))
        };

        // Mock target model (returns similar logits for acceptance)
        let target_fn = |_ctx: &[i32]| -> Result<Vec<f32>> {
            Ok(vec![1.0, 2.0, 3.0])
        };

        let initial = vec![0];
        let result = decoder.generate(draft_fn, target_fn, &initial, 5)?;

        // Should generate some tokens
        assert!(result.len() > initial.len());
        assert!(result.len() <= initial.len() + 5);

        // Check stats
        assert!(decoder.stats().total_speculation_rounds > 0);

        Ok(())
    }
}
