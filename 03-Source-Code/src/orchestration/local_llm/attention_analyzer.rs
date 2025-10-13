//! Attention Entropy Analysis for LLM Interpretability
//!
//! Analyzes attention patterns using information theory to understand
//! what the model is "looking at" during generation.
//!
//! Key Metrics:
//! - Attention Entropy: Measures focus vs diffusion
//! - Attention Collapse Detection: All heads focus on same tokens
//! - Token Importance: Which tokens influence generation most
//!
//! References:
//! - Low entropy = focused attention (specific tokens)
//! - High entropy = diffuse attention (many tokens)
//! - Attention collapse = model degradation

use anyhow::Result;

/// Attention pattern analyzer
///
/// Provides information-theoretic analysis of attention weights
/// for interpretability and debugging.
pub struct AttentionAnalyzer {
    /// Track attention entropy over time
    entropy_history: Vec<Vec<f32>>,  // [layer][step]
}

impl AttentionAnalyzer {
    /// Create new attention analyzer
    pub fn new() -> Self {
        Self {
            entropy_history: Vec::new(),
        }
    }

    /// Calculate attention entropy per head
    ///
    /// For each attention head, computes Shannon entropy of attention weights.
    ///
    /// H(attention) = -Σ w_i log₂(w_i)
    ///
    /// Interpretation:
    /// - Low entropy (< 1 bit): Head is highly focused on 1-2 tokens
    /// - Medium entropy (1-3 bits): Head attends to small set of tokens
    /// - High entropy (> 5 bits): Head attends broadly to many tokens
    ///
    /// # Arguments
    /// * `attn_weights` - Attention weights [seq_len, seq_len]
    ///                    Each row is a probability distribution
    ///
    /// # Returns
    /// Vector of entropy values, one per query position
    ///
    /// # Example
    /// ```
    /// let analyzer = AttentionAnalyzer::new();
    /// let entropy_per_head = analyzer.attention_entropy(&attn_weights);
    ///
    /// for (i, entropy) in entropy_per_head.iter().enumerate() {
    ///     if *entropy < 1.0 {
    ///         println!("Head {} is highly focused (entropy: {:.2} bits)", i, entropy);
    ///     }
    /// }
    /// ```
    pub fn attention_entropy(&self, attn_weights: &[Vec<f32>]) -> Vec<f32> {
        attn_weights.iter()
            .map(|row| {
                let mut entropy = 0.0f32;
                for &w in row.iter() {
                    if w > 1e-10 {
                        entropy -= w * w.log2();
                    }
                }
                entropy
            })
            .collect()
    }

    /// Detect attention collapse
    ///
    /// Attention collapse occurs when all heads focus on the same tokens,
    /// losing the diversity that makes multi-head attention effective.
    ///
    /// Detection criteria:
    /// - Average entropy across all heads < 1.0 bit
    /// - OR: >80% of heads have entropy < 0.5 bits
    ///
    /// This indicates potential model issues:
    /// - Numerical instability
    /// - Training problems
    /// - Quantization artifacts
    ///
    /// # Example
    /// ```
    /// let analyzer = AttentionAnalyzer::new();
    /// if analyzer.detect_attention_collapse(&multi_head_weights) {
    ///     println!("⚠️  Attention collapse detected!");
    /// }
    /// ```
    pub fn detect_attention_collapse(&self, multi_head_attn: &[Vec<Vec<f32>>]) -> bool {
        if multi_head_attn.is_empty() {
            return false;
        }

        // Calculate entropy for each head
        let mut all_entropies = Vec::new();
        for head_weights in multi_head_attn.iter() {
            let head_entropy = self.attention_entropy(head_weights);
            all_entropies.extend(head_entropy);
        }

        if all_entropies.is_empty() {
            return false;
        }

        // Average entropy across all heads
        let avg_entropy: f32 = all_entropies.iter().sum::<f32>() / all_entropies.len() as f32;

        // Count low-entropy heads
        let low_entropy_count = all_entropies.iter()
            .filter(|&&e| e < 0.5)
            .count();
        let low_entropy_ratio = low_entropy_count as f32 / all_entropies.len() as f32;

        // Collapse if average entropy very low OR most heads have low entropy
        avg_entropy < 1.0 || low_entropy_ratio > 0.8
    }

    /// Analyze token importance based on attention
    ///
    /// Computes how much each token is attended to across all query positions.
    /// High importance = token is frequently attended to = influential token.
    ///
    /// # Arguments
    /// * `attn_weights` - Attention weights [query_len, key_len]
    ///
    /// # Returns
    /// Vector of importance scores, one per key position
    ///
    /// # Example
    /// ```
    /// let analyzer = AttentionAnalyzer::new();
    /// let importance = analyzer.token_importance(&attn_weights);
    ///
    /// // Find most important token
    /// let (max_idx, max_score) = importance.iter().enumerate()
    ///     .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
    ///     .unwrap();
    /// println!("Most important token: {} (score: {:.3})", max_idx, max_score);
    /// ```
    pub fn token_importance(&self, attn_weights: &[Vec<f32>]) -> Vec<f32> {
        if attn_weights.is_empty() {
            return Vec::new();
        }

        let key_len = attn_weights[0].len();
        let mut importance = vec![0.0f32; key_len];

        // Sum attention weights received by each token
        for row in attn_weights.iter() {
            for (i, &w) in row.iter().enumerate() {
                importance[i] += w;
            }
        }

        // Normalize by number of query positions
        let query_len = attn_weights.len() as f32;
        for imp in importance.iter_mut() {
            *imp /= query_len;
        }

        importance
    }

    /// Track attention entropy over generation
    ///
    /// Useful for monitoring attention patterns during long generations.
    ///
    /// # Example
    /// ```
    /// let mut analyzer = AttentionAnalyzer::new();
    ///
    /// // During generation loop
    /// for step in 0..num_tokens {
    ///     let attn = model.get_attention_weights(layer);
    ///     let entropy = analyzer.attention_entropy(&attn);
    ///     analyzer.record_entropy(layer, entropy);
    /// }
    ///
    /// // Analyze trends
    /// let stats = analyzer.entropy_statistics(layer);
    /// ```
    pub fn record_entropy(&mut self, layer: usize, entropy: Vec<f32>) {
        // Ensure we have enough space
        while self.entropy_history.len() <= layer {
            self.entropy_history.push(Vec::new());
        }

        // Average entropy for this step
        let avg_entropy = entropy.iter().sum::<f32>() / entropy.len() as f32;
        self.entropy_history[layer].push(avg_entropy);
    }

    /// Get entropy statistics for a layer
    ///
    /// Returns (mean, std_dev, min, max) of attention entropy over time.
    pub fn entropy_statistics(&self, layer: usize) -> Option<AttentionStats> {
        if layer >= self.entropy_history.len() {
            return None;
        }

        let entropies = &self.entropy_history[layer];
        if entropies.is_empty() {
            return None;
        }

        let mean = entropies.iter().sum::<f32>() / entropies.len() as f32;

        let variance = entropies.iter()
            .map(|&e| (e - mean).powi(2))
            .sum::<f32>() / entropies.len() as f32;
        let std_dev = variance.sqrt();

        let min = entropies.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = entropies.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        Some(AttentionStats {
            mean,
            std_dev,
            min,
            max,
            num_samples: entropies.len(),
        })
    }

    /// Clear entropy history
    pub fn clear_history(&mut self) {
        self.entropy_history.clear();
    }

    /// Analyze attention pattern health
    ///
    /// Provides diagnostic information about attention quality.
    pub fn attention_health(&self, attn_weights: &[Vec<f32>]) -> AttentionHealth {
        let entropy_values = self.attention_entropy(attn_weights);

        if entropy_values.is_empty() {
            return AttentionHealth::Unknown;
        }

        let avg_entropy = entropy_values.iter().sum::<f32>() / entropy_values.len() as f32;

        // Count focused heads (low entropy)
        let focused_count = entropy_values.iter().filter(|&&e| e < 1.0).count();
        let focused_ratio = focused_count as f32 / entropy_values.len() as f32;

        // Count diffuse heads (high entropy)
        let diffuse_count = entropy_values.iter().filter(|&&e| e > 5.0).count();
        let diffuse_ratio = diffuse_count as f32 / entropy_values.len() as f32;

        if avg_entropy < 0.5 {
            AttentionHealth::Collapsed(format!(
                "Average entropy: {:.2} bits (threshold: 0.5)",
                avg_entropy
            ))
        } else if focused_ratio > 0.9 {
            AttentionHealth::TooFocused(format!(
                "{:.0}% of heads are highly focused",
                focused_ratio * 100.0
            ))
        } else if diffuse_ratio > 0.9 {
            AttentionHealth::TooDiffuse(format!(
                "{:.0}% of heads are too diffuse",
                diffuse_ratio * 100.0
            ))
        } else {
            AttentionHealth::Healthy {
                avg_entropy,
                focused_ratio,
                diffuse_ratio,
            }
        }
    }
}

impl Default for AttentionAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Attention entropy statistics
#[derive(Debug, Clone)]
pub struct AttentionStats {
    pub mean: f32,
    pub std_dev: f32,
    pub min: f32,
    pub max: f32,
    pub num_samples: usize,
}

/// Attention health status
#[derive(Debug, Clone)]
pub enum AttentionHealth {
    /// Attention is healthy (balanced entropy)
    Healthy {
        avg_entropy: f32,
        focused_ratio: f32,
        diffuse_ratio: f32,
    },
    /// Attention has collapsed (all heads focus on same tokens)
    Collapsed(String),
    /// Too many heads are overly focused
    TooFocused(String),
    /// Too many heads are overly diffuse
    TooDiffuse(String),
    /// Unknown (no attention data)
    Unknown,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_entropy_focused() {
        let analyzer = AttentionAnalyzer::new();

        // Highly focused attention (one hot)
        let focused = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
        ];

        let entropy = analyzer.attention_entropy(&focused);

        // Entropy should be near 0 (deterministic)
        assert!(entropy[0] < 0.01);
        assert!(entropy[1] < 0.01);
    }

    #[test]
    fn test_attention_entropy_diffuse() {
        let analyzer = AttentionAnalyzer::new();

        // Uniform attention (diffuse)
        let diffuse = vec![
            vec![0.25, 0.25, 0.25, 0.25],
            vec![0.25, 0.25, 0.25, 0.25],
        ];

        let entropy = analyzer.attention_entropy(&diffuse);

        // Entropy should be log2(4) = 2 bits
        assert!((entropy[0] - 2.0).abs() < 0.01);
        assert!((entropy[1] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_attention_collapse_detection() {
        let analyzer = AttentionAnalyzer::new();

        // All heads focus on same token (collapsed)
        let collapsed = vec![
            vec![vec![1.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]],
            vec![vec![1.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]],
        ];

        assert!(analyzer.detect_attention_collapse(&collapsed));

        // Diverse attention (healthy)
        let healthy = vec![
            vec![vec![0.5, 0.3, 0.2], vec![0.2, 0.5, 0.3]],
            vec![vec![0.3, 0.2, 0.5], vec![0.4, 0.4, 0.2]],
        ];

        assert!(!analyzer.detect_attention_collapse(&healthy));
    }

    #[test]
    fn test_token_importance() {
        let analyzer = AttentionAnalyzer::new();

        // Token 0 is most attended to
        let attn = vec![
            vec![0.8, 0.1, 0.1],
            vec![0.7, 0.2, 0.1],
            vec![0.6, 0.3, 0.1],
        ];

        let importance = analyzer.token_importance(&attn);

        // Token 0 should have highest importance
        assert!(importance[0] > importance[1]);
        assert!(importance[1] > importance[2]);

        // Sum should equal 1.0 (normalized)
        let sum: f32 = importance.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_entropy_tracking() {
        let mut analyzer = AttentionAnalyzer::new();

        // Record entropy over multiple steps
        analyzer.record_entropy(0, vec![1.5, 1.6, 1.4]);
        analyzer.record_entropy(0, vec![1.6, 1.5, 1.5]);
        analyzer.record_entropy(0, vec![1.4, 1.5, 1.6]);

        let stats = analyzer.entropy_statistics(0).unwrap();

        // Mean should be around 1.5
        assert!((stats.mean - 1.5).abs() < 0.1);
        assert!(stats.num_samples == 3);
    }

    #[test]
    fn test_attention_health() {
        let analyzer = AttentionAnalyzer::new();

        // Healthy attention (mixed entropy)
        let healthy_attn = vec![
            vec![0.5, 0.3, 0.2],
            vec![0.4, 0.4, 0.2],
            vec![0.3, 0.5, 0.2],
        ];

        match analyzer.attention_health(&healthy_attn) {
            AttentionHealth::Healthy { .. } => {}
            _ => panic!("Expected healthy attention"),
        }

        // Collapsed attention
        let collapsed_attn = vec![
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
        ];

        match analyzer.attention_health(&collapsed_attn) {
            AttentionHealth::Collapsed(_) => {}
            _ => panic!("Expected collapsed attention"),
        }
    }
}
