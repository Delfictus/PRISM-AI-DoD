//! LLM Analysis Integration
//!
//! Provides unified interface for information-theoretic analysis tools.
//! Integrates metrics tracking, attention analysis, and transfer entropy
//! into a single convenient API for monitoring and debugging LLM inference.

use anyhow::Result;
use crate::orchestration::local_llm::{
    LLMMetrics, AttentionAnalyzer, TransferEntropyLLM,
    AttentionHealth, DistributionHealth, TransferEntropyStats,
};

/// Comprehensive LLM analysis suite
///
/// Combines all Phase 1-3 information-theoretic enhancements into a single,
/// easy-to-use interface for monitoring LLM quality, attention patterns,
/// and causal information flow.
pub struct LLMAnalysis {
    metrics: LLMMetrics,
    attention_analyzer: AttentionAnalyzer,
    transfer_entropy: TransferEntropyLLM,
    enabled: bool,
}

impl LLMAnalysis {
    /// Create new analysis suite with all tools enabled
    ///
    /// # Arguments
    /// * `n_bins` - Number of bins for transfer entropy discretization (typically 10-20)
    pub fn new(n_bins: usize) -> Self {
        Self {
            metrics: LLMMetrics::new(),
            attention_analyzer: AttentionAnalyzer::new(),
            transfer_entropy: TransferEntropyLLM::new(n_bins),
            enabled: true,
        }
    }

    /// Enable all analysis tools
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable all analysis tools (stops collecting data)
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Check if analysis is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Record a generation step for transfer entropy tracking
    ///
    /// Call this after each token is generated to build up causality history.
    ///
    /// # Arguments
    /// * `logits` - Model output logits for this step
    /// * `token` - Generated token ID
    pub fn record_step(&mut self, logits: Vec<f32>, token: i32) {
        if !self.enabled {
            return;
        }
        self.transfer_entropy.record_step(logits, token);
    }

    /// Calculate perplexity for current prediction
    ///
    /// Measures model quality - lower is better.
    /// Range: [1, vocab_size]
    ///
    /// # Returns
    /// Perplexity value, or None if analysis is disabled
    pub fn perplexity(&self, logits: &[f32], target_token: i32) -> Option<f32> {
        if !self.enabled {
            return None;
        }
        self.metrics.perplexity(logits, target_token).ok()
    }

    /// Check distribution health against reference
    ///
    /// Monitors for distribution drift, quantization artifacts, or numerical issues.
    ///
    /// # Returns
    /// DistributionHealth status, or None if analysis is disabled
    pub fn check_distribution_health(
        &mut self,
        layer: usize,
        logits: &[f32],
    ) -> Option<DistributionHealth> {
        if !self.enabled {
            return None;
        }
        self.metrics.check_distribution_health(layer, logits).ok()
    }

    /// Analyze attention pattern health
    ///
    /// Detects attention collapse and other attention issues.
    ///
    /// # Arguments
    /// * `attn_weights` - Attention weights matrix [query_len, key_len]
    ///
    /// # Returns
    /// AttentionHealth status, or None if analysis is disabled
    pub fn attention_health(&self, attn_weights: &[Vec<f32>]) -> Option<AttentionHealth> {
        if !self.enabled {
            return None;
        }
        Some(self.attention_analyzer.attention_health(attn_weights))
    }

    /// Detect attention collapse
    ///
    /// Returns true if all attention heads focus on same tokens.
    /// Indicates potential model degradation.
    pub fn detect_attention_collapse(&self, multi_head_attn: &[Vec<Vec<f32>>]) -> bool {
        if !self.enabled {
            return false;
        }
        self.attention_analyzer.detect_attention_collapse(multi_head_attn)
    }

    /// Calculate token importance from attention weights
    ///
    /// Returns normalized importance scores for each token.
    /// Higher scores = more influential tokens.
    pub fn token_importance(&self, attn_weights: &[Vec<f32>]) -> Option<Vec<f32>> {
        if !self.enabled {
            return None;
        }
        Some(self.attention_analyzer.token_importance(attn_weights))
    }

    /// Calculate transfer entropy between two token positions
    ///
    /// Measures causal information flow: how much knowing source token
    /// helps predict target token.
    ///
    /// # Arguments
    /// * `source_pos` - Position of source token (earlier)
    /// * `target_pos` - Position of target token (later)
    /// * `history_length` - How many past steps to consider (k parameter, typically 1-3)
    ///
    /// # Returns
    /// Transfer entropy in bits, or None if analysis is disabled
    pub fn calculate_transfer_entropy(
        &self,
        source_pos: usize,
        target_pos: usize,
        history_length: usize,
    ) -> Option<f32> {
        if !self.enabled {
            return None;
        }
        self.transfer_entropy
            .calculate_transfer_entropy(source_pos, target_pos, history_length)
            .ok()
    }

    /// Find most influential tokens based on transfer entropy
    ///
    /// Returns (position, total_influence) pairs sorted by influence.
    ///
    /// # Arguments
    /// * `history_length` - History parameter for TE calculation
    /// * `top_k` - Number of top tokens to return
    pub fn find_influential_tokens(
        &self,
        history_length: usize,
        top_k: usize,
    ) -> Option<Vec<(usize, f32)>> {
        if !self.enabled {
            return None;
        }
        self.transfer_entropy
            .find_influential_tokens(history_length, top_k)
            .ok()
    }

    /// Get transfer entropy statistics
    ///
    /// Returns summary statistics for all token pairs.
    pub fn transfer_entropy_stats(
        &self,
        history_length: usize,
        significance_threshold: f32,
    ) -> Option<TransferEntropyStats> {
        if !self.enabled {
            return None;
        }
        self.transfer_entropy
            .calculate_statistics(history_length, significance_threshold)
            .ok()
    }

    /// Clear all analysis history
    ///
    /// Call this before starting a new generation to reset state.
    pub fn clear_history(&mut self) {
        self.attention_analyzer.clear_history();
        self.transfer_entropy.clear_history();
        self.metrics.clear_references();
    }

    /// Get comprehensive analysis report
    ///
    /// Returns a human-readable summary of all analysis metrics.
    ///
    /// # Arguments
    /// * `logits` - Current logits for quality analysis
    /// * `attn_weights` - Attention weights for attention analysis
    pub fn generate_report(
        &mut self,
        logits: &[f32],
        attn_weights: Option<&[Vec<f32>]>,
    ) -> String {
        if !self.enabled {
            return "Analysis disabled".to_string();
        }

        let mut report = String::new();
        report.push_str("╔══════════════════════════════════════════╗\n");
        report.push_str("║  LLM ANALYSIS REPORT                     ║\n");
        report.push_str("╚══════════════════════════════════════════╝\n\n");

        // Distribution metrics
        report.push_str("📊 Distribution Metrics:\n");
        let entropy = self.metrics.entropy(&self.metrics.softmax(logits));
        report.push_str(&format!("   Entropy: {:.2} bits\n", entropy));

        if entropy < 1.0 {
            report.push_str("   → Model is confident (low entropy)\n");
        } else if entropy > 5.0 {
            report.push_str("   → Model is uncertain (high entropy)\n");
        } else {
            report.push_str("   → Model has moderate confidence\n");
        }

        // Attention analysis
        if let Some(attn) = attn_weights {
            report.push_str("\n🔍 Attention Analysis:\n");
            match self.attention_analyzer.attention_health(attn) {
                AttentionHealth::Healthy {
                    avg_entropy,
                    focused_ratio,
                    diffuse_ratio,
                } => {
                    report.push_str(&format!("   Status: ✅ Healthy\n"));
                    report.push_str(&format!("   Avg entropy: {:.2} bits\n", avg_entropy));
                    report.push_str(&format!("   Focused heads: {:.0}%\n", focused_ratio * 100.0));
                    report.push_str(&format!("   Diffuse heads: {:.0}%\n", diffuse_ratio * 100.0));
                }
                AttentionHealth::Collapsed(msg) => {
                    report.push_str(&format!("   Status: ❌ Collapsed\n"));
                    report.push_str(&format!("   {}\n", msg));
                }
                AttentionHealth::TooFocused(msg) => {
                    report.push_str(&format!("   Status: ⚠️  Too Focused\n"));
                    report.push_str(&format!("   {}\n", msg));
                }
                AttentionHealth::TooDiffuse(msg) => {
                    report.push_str(&format!("   Status: ⚠️  Too Diffuse\n"));
                    report.push_str(&format!("   {}\n", msg));
                }
                AttentionHealth::Unknown => {
                    report.push_str("   Status: ❓ Unknown\n");
                }
            }
        }

        // Transfer entropy analysis (if we have history)
        if self.transfer_entropy.history_length() > 2 {
            report.push_str("\n🔗 Causality Analysis:\n");
            if let Ok(stats) = self.transfer_entropy.calculate_statistics(1, 0.1) {
                report.push_str(&format!("   Mean TE: {:.3} bits\n", stats.mean_te));
                report.push_str(&format!("   Max TE: {:.3} bits\n", stats.max_te));
                report.push_str(&format!(
                    "   Strongest link: token {} → token {}\n",
                    stats.max_source, stats.max_target
                ));
                report.push_str(&format!("   Significant links: {}\n", stats.significant_links));
            }
        }

        report.push_str("\n");
        report
    }
}

impl Default for LLMAnalysis {
    fn default() -> Self {
        Self::new(10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llm_analysis_creation() {
        let analysis = LLMAnalysis::new(10);
        assert!(analysis.is_enabled());
    }

    #[test]
    fn test_enable_disable() {
        let mut analysis = LLMAnalysis::new(10);

        analysis.disable();
        assert!(!analysis.is_enabled());

        analysis.enable();
        assert!(analysis.is_enabled());
    }

    #[test]
    fn test_perplexity_disabled() {
        let mut analysis = LLMAnalysis::new(10);
        analysis.disable();

        let logits = vec![1.0, 2.0, 3.0];
        let ppl = analysis.perplexity(&logits, 2);

        assert!(ppl.is_none());
    }

    #[test]
    fn test_perplexity_enabled() {
        let analysis = LLMAnalysis::new(10);

        // Logits favoring token 2
        let logits = vec![-10.0, -10.0, 10.0, -10.0];
        let ppl = analysis.perplexity(&logits, 2);

        assert!(ppl.is_some());
        assert!(ppl.unwrap() < 1.5); // Should be near 1 for correct prediction
    }

    #[test]
    fn test_attention_health() {
        let analysis = LLMAnalysis::new(10);

        // Healthy attention (mixed entropy)
        let healthy_attn = vec![
            vec![0.5, 0.3, 0.2],
            vec![0.4, 0.4, 0.2],
            vec![0.3, 0.5, 0.2],
        ];

        let health = analysis.attention_health(&healthy_attn);
        assert!(health.is_some());

        match health.unwrap() {
            AttentionHealth::Healthy { .. } => {}
            _ => panic!("Expected healthy attention"),
        }
    }

    #[test]
    fn test_record_step() {
        let mut analysis = LLMAnalysis::new(10);

        // Record a few steps
        analysis.record_step(vec![1.0, 2.0, 3.0], 2);
        analysis.record_step(vec![2.0, 3.0, 1.0], 1);
        analysis.record_step(vec![3.0, 1.0, 2.0], 0);

        // Should have 3 steps recorded
        // (Can't directly test without exposing internal state,
        //  but this shouldn't panic)
    }

    #[test]
    fn test_clear_history() {
        let mut analysis = LLMAnalysis::new(10);

        // Record some data
        analysis.record_step(vec![1.0, 2.0, 3.0], 2);
        analysis.record_step(vec![2.0, 3.0, 1.0], 1);

        // Clear should not panic
        analysis.clear_history();
    }

    #[test]
    fn test_generate_report() {
        let mut analysis = LLMAnalysis::new(10);

        let logits = vec![1.0, 2.0, 3.0, 0.5];
        let attn = vec![
            vec![0.5, 0.3, 0.2],
            vec![0.4, 0.4, 0.2],
        ];

        let report = analysis.generate_report(&logits, Some(&attn));

        // Report should contain key sections
        assert!(report.contains("Distribution Metrics"));
        assert!(report.contains("Attention Analysis"));
    }

    #[test]
    fn test_generate_report_disabled() {
        let mut analysis = LLMAnalysis::new(10);
        analysis.disable();

        let logits = vec![1.0, 2.0, 3.0];
        let report = analysis.generate_report(&logits, None);

        assert_eq!(report, "Analysis disabled");
    }
}
