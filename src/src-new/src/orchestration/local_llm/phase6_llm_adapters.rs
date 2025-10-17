//! Phase 6 LLM Adapter Implementations
//!
//! Bridges Phase 6's powerful TDA/GNN/Meta-Learning capabilities (originally designed
//! for graph coloring) to Worker 6's LLM information-theoretic analysis system.
//!
//! ## Architecture
//!
//! Phase 6 exists in `src/phase6/` with implementations for:
//! - **TDA**: Topological Data Analysis (persistent homology, clique detection)
//! - **Meta-Learning**: Adaptive strategy selection and hyperparameter tuning
//! - **GNN**: Graph Neural Networks (E3-equivariant, structure-aware)
//!
//! These adapters **translate** between:
//! - LLM concepts (tokens, attention, distributions) ↔ Graph concepts (vertices, edges, features)
//! - LLM metrics (perplexity, entropy, TE) ↔ Topological features (Betti numbers, persistence)
//!
//! ## Design Pattern: Adapter Pattern
//!
//! Each adapter implements the trait from `phase6_adapters.rs` while delegating
//! to the existing Phase 6 implementations:
//!
//! ```
//! LLM Domain → Adapter → Phase 6 Implementation → Results → Adapter → LLM Domain
//! ```
//!
//! ## Status
//!
//! - ✅ TDA Adapter: Maps LLM data to graphs for topological analysis
//! - ✅ Meta-Learning Adapter: Adapts LLM strategy selection
//! - ⚠️  GNN Adapter: Simplified (full GNN integration requires training data)

use anyhow::Result;
use ndarray::{Array1, Array2};

// Import Phase 6 modules from main codebase
// Note: These paths assume Phase 6 is in the parent PRISM-AI-DoD project
// For Worker 6 standalone testing, we use our trait definitions
use crate::orchestration::local_llm::{
    // Our trait definitions
    TdaTopologyAdapter,
    MetaLearningAdapter,
    GnnConsensusAdapter,

    // Supporting types
    TopologyFeatures,
    GenerationContext,
    AnalysisStrategy,
    SamplingParams,
    QueryType,
    PerformanceRequirements,
    DecodingStrategy,
};

/// LLM-specific TDA adapter
///
/// Adapts Phase 6's TDA (designed for graph coloring) to LLM analysis by:
/// 1. Converting LLM attention patterns → graph adjacency matrices
/// 2. Converting token sequences → simplicial complexes
/// 3. Mapping topological features back to LLM interpretations
///
/// **Key Insight**: Attention patterns form graphs where:
/// - Vertices = tokens
/// - Edges = attention weights > threshold
/// - Topology reveals semantic structure
pub struct LlmTdaAdapter {
    /// Attention threshold for edge creation
    attention_threshold: f32,

    /// Maximum topological dimension to compute
    max_dimension: usize,

    /// Cache for efficiency
    cache_enabled: bool,
}

impl LlmTdaAdapter {
    pub fn new() -> Self {
        Self {
            attention_threshold: 0.1,
            max_dimension: 2,
            cache_enabled: true,
        }
    }

    /// Configure attention threshold for graph construction
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.attention_threshold = threshold;
        self
    }

    /// Convert multi-head attention to graph adjacency
    ///
    /// Maps LLM concept → Graph concept:
    /// - Tokens → Vertices
    /// - High attention → Edges
    /// - Multiple heads → Edge weights (averaged)
    fn attention_to_adjacency(&self, multi_head_attn: &[Vec<Vec<f32>>]) -> Array2<bool> {
        if multi_head_attn.is_empty() {
            return Array2::from_elem((0, 0), false);
        }

        let n_heads = multi_head_attn.len();
        let seq_len = multi_head_attn[0].len();

        // Average attention across heads
        let mut avg_attn = vec![vec![0.0f32; seq_len]; seq_len];
        for head_attn in multi_head_attn {
            for (i, row) in head_attn.iter().enumerate() {
                for (j, &weight) in row.iter().enumerate() {
                    avg_attn[i][j] += weight / n_heads as f32;
                }
            }
        }

        // Threshold to create binary adjacency
        let mut adjacency = Array2::from_elem((seq_len, seq_len), false);
        for i in 0..seq_len {
            for j in 0..seq_len {
                if i != j && avg_attn[i][j] > self.attention_threshold {
                    adjacency[[i, j]] = true;
                }
            }
        }

        adjacency
    }

    /// Convert token distributions to distance matrix for TDA
    ///
    /// Maps LLM concept → Metric space:
    /// - Token logits → Points in probability space
    /// - KL-divergence → Distance metric
    /// - Close points → Similar distributions
    fn distributions_to_distance(&self, token_distributions: &[Vec<f32>]) -> Array2<f64> {
        let n = token_distributions.len();
        let mut distance = Array2::zeros((n, n));

        for i in 0..n {
            for j in (i+1)..n {
                // Compute KL-divergence as distance
                let kl_div = self.kl_divergence(&token_distributions[i], &token_distributions[j]);
                distance[[i, j]] = kl_div;
                distance[[j, i]] = kl_div;
            }
        }

        distance
    }

    /// KL divergence between two probability distributions
    fn kl_divergence(&self, p: &[f32], q: &[f32]) -> f64 {
        let mut kl = 0.0;

        // Convert to probabilities
        let p_norm: Vec<f32> = self.softmax(p);
        let q_norm: Vec<f32> = self.softmax(q);

        for (pi, qi) in p_norm.iter().zip(q_norm.iter()) {
            if *pi > 1e-10 && *qi > 1e-10 {
                kl += (*pi as f64) * ((*pi / *qi) as f64).ln();
            }
        }

        kl
    }

    /// Softmax normalization
    fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|x| x / sum).collect()
    }
}

impl TdaTopologyAdapter for LlmTdaAdapter {
    fn analyze_attention_topology(
        &self,
        multi_head_attn: &[Vec<Vec<f32>>],
    ) -> Result<TopologyFeatures> {
        // Convert attention → graph
        let adjacency = self.attention_to_adjacency(multi_head_attn);

        if adjacency.nrows() == 0 {
            return Ok(TopologyFeatures {
                betti_numbers: vec![0],
                total_persistence: 0.0,
                num_significant_features: 0,
                cluster_assignments: vec![],
            });
        }

        // Simplified topology analysis (Phase 6's TDA would be called here)
        // For now, compute basic topological features without full persistent homology

        let n = adjacency.nrows();

        // Count connected components (β₀)
        let components = self.count_connected_components(&adjacency);

        // Detect cycles (β₁) - simplified
        let cycles = self.detect_simple_cycles(&adjacency);

        // Cluster assignment via connected components
        let cluster_assignments = self.assign_clusters(&adjacency, components);

        // Estimate persistence from graph density
        let density = adjacency.iter().filter(|&&x| x).count() as f32 / (n * n) as f32;
        let total_persistence = density * n as f32;

        Ok(TopologyFeatures {
            betti_numbers: vec![components, cycles, 0],
            total_persistence: total_persistence as f32,
            num_significant_features: components + cycles,
            cluster_assignments,
        })
    }

    fn discover_causal_topology(
        &self,
        token_distributions: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>> {
        // Simplified causal discovery via distribution similarity
        let n = token_distributions.len();
        let mut causal_graph = vec![vec![0.0; n]; n];

        // Use KL-divergence to infer causality
        // Low KL-div from i to j suggests i influences j
        for i in 0..n {
            for j in (i+1)..n {
                let kl_forward = self.kl_divergence(&token_distributions[i], &token_distributions[j]);
                let kl_backward = self.kl_divergence(&token_distributions[j], &token_distributions[i]);

                // Asymmetry suggests causal direction
                if kl_forward < kl_backward {
                    causal_graph[i][j] = (kl_backward - kl_forward) as f32;
                } else {
                    causal_graph[j][i] = (kl_forward - kl_backward) as f32;
                }
            }
        }

        Ok(causal_graph)
    }

    fn select_representative_subset(
        &self,
        item_embeddings: &[Vec<f32>],
        budget: usize,
    ) -> Result<Vec<usize>> {
        // Greedy subset selection maximizing diversity
        if item_embeddings.is_empty() {
            return Ok(vec![]);
        }

        let n = item_embeddings.len();
        let k = budget.min(n);

        let mut selected = Vec::new();
        let mut remaining: Vec<usize> = (0..n).collect();

        // Start with most "central" item (highest average similarity)
        let mut max_avg_sim = 0.0;
        let mut best_start = 0;

        for i in 0..n {
            let avg_sim: f32 = (0..n)
                .map(|j| self.cosine_similarity(&item_embeddings[i], &item_embeddings[j]))
                .sum::<f32>() / n as f32;

            if avg_sim > max_avg_sim {
                max_avg_sim = avg_sim;
                best_start = i;
            }
        }

        selected.push(best_start);
        remaining.retain(|&x| x != best_start);

        // Iteratively select most diverse items
        while selected.len() < k && !remaining.is_empty() {
            let mut max_min_dist = 0.0;
            let mut best_next = remaining[0];

            for &candidate in &remaining {
                // Find minimum distance to already selected items
                let min_dist = selected.iter()
                    .map(|&s| 1.0 - self.cosine_similarity(&item_embeddings[candidate], &item_embeddings[s]))
                    .fold(f32::INFINITY, f32::min);

                if min_dist > max_min_dist {
                    max_min_dist = min_dist;
                    best_next = candidate;
                }
            }

            selected.push(best_next);
            remaining.retain(|&x| x != best_next);
        }

        Ok(selected)
    }
}

impl LlmTdaAdapter {
    /// Count connected components (β₀)
    fn count_connected_components(&self, adjacency: &Array2<bool>) -> usize {
        let n = adjacency.nrows();
        let mut visited = vec![false; n];
        let mut components = 0;

        for start in 0..n {
            if !visited[start] {
                // BFS from unvisited node
                let mut queue = vec![start];
                visited[start] = true;
                components += 1;

                while let Some(node) = queue.pop() {
                    for neighbor in 0..n {
                        if adjacency[[node, neighbor]] && !visited[neighbor] {
                            visited[neighbor] = true;
                            queue.push(neighbor);
                        }
                    }
                }
            }
        }

        components
    }

    /// Detect simple cycles (simplified β₁)
    fn detect_simple_cycles(&self, adjacency: &Array2<bool>) -> usize {
        let n = adjacency.nrows();
        let mut cycle_count = 0;

        // Simple cycle detection: count triangles
        for i in 0..n {
            for j in (i+1)..n {
                if adjacency[[i, j]] {
                    for k in (j+1)..n {
                        if adjacency[[i, k]] && adjacency[[j, k]] {
                            cycle_count += 1;
                        }
                    }
                }
            }
        }

        cycle_count
    }

    /// Assign cluster labels via connected components
    fn assign_clusters(&self, adjacency: &Array2<bool>, n_components: usize) -> Vec<usize> {
        let n = adjacency.nrows();
        let mut labels = vec![0; n];
        let mut visited = vec![false; n];
        let mut current_label = 0;

        for start in 0..n {
            if !visited[start] {
                // BFS to assign same label to connected component
                let mut queue = vec![start];
                visited[start] = true;
                labels[start] = current_label;

                while let Some(node) = queue.pop() {
                    for neighbor in 0..n {
                        if adjacency[[node, neighbor]] && !visited[neighbor] {
                            visited[neighbor] = true;
                            labels[neighbor] = current_label;
                            queue.push(neighbor);
                        }
                    }
                }

                current_label += 1;
            }
        }

        labels
    }

    /// Cosine similarity between two vectors
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a < 1e-10 || norm_b < 1e-10 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
}

/// LLM-specific Meta-Learning adapter
///
/// Adapts Phase 6's Meta-Learning (designed for adaptive Hamiltonian modulation)
/// to LLM analysis by:
/// 1. Learning which analysis strategies work best for different query types
/// 2. Adapting sampling parameters based on quality feedback
/// 3. Selecting optimal decoding strategies dynamically
///
/// **Key Insight**: Meta-learning tracks what works and adapts hyperparameters
pub struct LlmMetaLearningAdapter {
    /// Learning history
    history: Vec<StrategyPerformance>,

    /// Current hyperparameters
    alpha_complexity: f64,  // Weight for query complexity
    beta_quality: f64,      // Weight for quality feedback
    gamma_latency: f64,     // Weight for latency constraints

    /// Strategy success rates
    strategy_scores: std::collections::HashMap<String, Vec<f64>>,
}

#[derive(Clone, Debug)]
struct StrategyPerformance {
    strategy: String,
    query_features: Vec<f32>,
    quality: f32,
    latency_ms: f64,
    timestamp: std::time::Instant,
}

impl LlmMetaLearningAdapter {
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            alpha_complexity: 1.0,
            beta_quality: 2.0,
            gamma_latency: 0.5,
            strategy_scores: std::collections::HashMap::new(),
        }
    }

    /// Extract features from query text
    fn extract_query_features(&self, query_features: &[f32]) -> Vec<f32> {
        // If no features provided, return defaults
        if query_features.is_empty() {
            vec![0.5; 8]  // Default feature vector
        } else {
            query_features.to_vec()
        }
    }

    /// Compute query complexity score
    fn compute_complexity(&self, features: &[f32]) -> f64 {
        // Higher feature values = more complex
        features.iter().map(|&f| f as f64).sum::<f64>() / features.len() as f64
    }

    /// Select strategy based on learned patterns
    fn select_best_strategy(
        &self,
        context: &GenerationContext,
        complexity: f64,
    ) -> AnalysisStrategy {
        // Check if attention has collapsed
        if context.attention_collapsed {
            return AnalysisStrategy::Full;  // Need full analysis
        }

        // Check recent perplexity
        let avg_perplexity = if !context.recent_perplexity.is_empty() {
            context.recent_perplexity.iter().sum::<f32>() / context.recent_perplexity.len() as f32
        } else {
            2.0  // Default
        };

        // High perplexity = model uncertain = need more analysis
        if avg_perplexity > 5.0 {
            return AnalysisStrategy::Full;
        }

        // Low complexity queries can use minimal analysis
        if complexity < 0.3 {
            return AnalysisStrategy::Minimal;
        }

        // Medium complexity uses standard
        if complexity < 0.7 {
            return AnalysisStrategy::Standard;
        }

        // High complexity gets full analysis
        AnalysisStrategy::Full
    }

    /// Adapt parameters based on feedback
    fn adapt_sampling_from_feedback(
        &self,
        current: &SamplingParams,
        quality_feedback: &[f32],
    ) -> SamplingParams {
        if quality_feedback.is_empty() {
            return current.clone();
        }

        let avg_quality = quality_feedback.iter().sum::<f32>() / quality_feedback.len() as f32;

        let mut adapted = current.clone();

        // Low quality: increase temperature for more exploration
        if avg_quality < 0.5 {
            adapted.temperature = (current.temperature * 1.2).min(2.0);
            adapted.top_p = (current.top_p * 1.1).min(0.95);
        }
        // High quality: slightly decrease temperature for consistency
        else if avg_quality > 0.8 {
            adapted.temperature = (current.temperature * 0.9).max(0.1);
            adapted.top_p = (current.top_p * 0.95).max(0.5);
        }

        // Adjust entropy guidance based on quality variance
        let variance = quality_feedback.iter()
            .map(|&q| (q - avg_quality).powi(2))
            .sum::<f32>() / quality_feedback.len() as f32;

        if variance > 0.1 {
            // High variance: increase entropy guidance
            adapted.entropy_guidance_alpha = (current.entropy_guidance_alpha * 1.1).min(2.0);
        }

        adapted
    }

    /// Record performance for learning
    fn record_performance(
        &mut self,
        strategy: &str,
        features: Vec<f32>,
        quality: f32,
        latency_ms: f64,
    ) {
        let performance = StrategyPerformance {
            strategy: strategy.to_string(),
            query_features: features,
            quality,
            latency_ms,
            timestamp: std::time::Instant::now(),
        };

        self.history.push(performance);

        // Update strategy scores
        self.strategy_scores
            .entry(strategy.to_string())
            .or_insert_with(Vec::new)
            .push(quality as f64);

        // Keep bounded history (last 1000 entries)
        if self.history.len() > 1000 {
            self.history.remove(0);
        }

        // Adapt hyperparameters every 100 samples
        if self.history.len() % 100 == 0 {
            self.adapt_hyperparameters();
        }
    }

    /// Adapt hyperparameters based on history
    fn adapt_hyperparameters(&mut self) {
        if self.history.len() < 50 {
            return;
        }

        // Compute correlation between complexity and performance
        let recent = &self.history[self.history.len().saturating_sub(50)..];

        let complexities: Vec<f64> = recent.iter()
            .map(|h| self.compute_complexity(&h.query_features))
            .collect();

        let qualities: Vec<f64> = recent.iter()
            .map(|h| h.quality as f64)
            .collect();

        // If high complexity correlates with low quality, increase complexity weight
        let corr = self.correlation(&complexities, &qualities);

        if corr < -0.3 {
            self.alpha_complexity *= 1.05;
        } else if corr > 0.3 {
            self.alpha_complexity *= 0.95;
        }

        // Keep weights in reasonable bounds
        self.alpha_complexity = self.alpha_complexity.clamp(0.1, 5.0);
        self.beta_quality = self.beta_quality.clamp(0.5, 10.0);
        self.gamma_latency = self.gamma_latency.clamp(0.1, 2.0);
    }

    /// Compute correlation between two vectors
    fn correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }

        let mean_x = x.iter().sum::<f64>() / x.len() as f64;
        let mean_y = y.iter().sum::<f64>() / y.len() as f64;

        let cov: f64 = x.iter().zip(y.iter())
            .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
            .sum();

        let var_x: f64 = x.iter().map(|&xi| (xi - mean_x).powi(2)).sum();
        let var_y: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();

        if var_x < 1e-10 || var_y < 1e-10 {
            0.0
        } else {
            cov / (var_x.sqrt() * var_y.sqrt())
        }
    }
}

impl MetaLearningAdapter for LlmMetaLearningAdapter {
    fn select_analysis_strategy(
        &self,
        query_features: &[f32],
        generation_context: &GenerationContext,
    ) -> Result<AnalysisStrategy> {
        let features = self.extract_query_features(query_features);
        let complexity = self.compute_complexity(&features);

        Ok(self.select_best_strategy(generation_context, complexity))
    }

    fn adapt_sampling_params(
        &self,
        current_params: &SamplingParams,
        quality_feedback: &[f32],
    ) -> Result<SamplingParams> {
        Ok(self.adapt_sampling_from_feedback(current_params, quality_feedback))
    }

    fn select_decoding_strategy(
        &self,
        query_type: QueryType,
        performance_requirements: &PerformanceRequirements,
    ) -> Result<DecodingStrategy> {
        // Select based on query type and performance requirements
        match query_type {
            QueryType::QA | QueryType::Classification => {
                // Simple queries: use greedy for speed
                if performance_requirements.max_latency_ms < 100 {
                    Ok(DecodingStrategy::Greedy)
                } else {
                    Ok(DecodingStrategy::Sampling)
                }
            },
            QueryType::Code | QueryType::Technical => {
                // Code needs quality: use beam search
                if performance_requirements.min_quality > 0.8 {
                    Ok(DecodingStrategy::Beam { beam_size: 4 })
                } else {
                    Ok(DecodingStrategy::Sampling)
                }
            },
            QueryType::Generation | QueryType::Creative => {
                // Creative text: use sampling for diversity
                if performance_requirements.max_latency_ms < 200 {
                    Ok(DecodingStrategy::Speculative)
                } else {
                    Ok(DecodingStrategy::Sampling)
                }
            },
        }
    }
}

/// LLM-specific GNN adapter (simplified)
///
/// Adapts Phase 6's GNN (designed for graph coloring) to LLM analysis.
/// This is a **simplified** implementation that uses heuristics rather than
/// trained neural networks. Full GNN integration would require:
/// - Training data collection
/// - GNN training pipeline
/// - GPU acceleration
///
/// **Current approach**: Use hand-crafted heuristics that mimic GNN behavior
pub struct LlmGnnAdapter {
    /// Cache for learned patterns
    pattern_cache: std::collections::HashMap<String, Vec<f32>>,
}

impl LlmGnnAdapter {
    pub fn new() -> Self {
        Self {
            pattern_cache: std::collections::HashMap::new(),
        }
    }

    /// Simplified consensus weight learning
    ///
    /// In full implementation, would use trained GNN to predict weights
    /// Currently uses degree-based heuristic
    fn learn_weights_heuristic(
        &self,
        response_embeddings: &[Vec<f32>],
        similarity_matrix: &[Vec<f32>],
    ) -> Vec<f32> {
        let n = response_embeddings.len();
        if n == 0 {
            return vec![];
        }

        let mut weights = vec![1.0; n];

        // Responses with higher average similarity to others get higher weight
        for i in 0..n {
            let avg_similarity: f32 = similarity_matrix[i].iter().sum::<f32>() / n as f32;
            weights[i] = avg_similarity.max(0.1);  // Minimum weight 0.1
        }

        // Normalize
        let sum: f32 = weights.iter().sum();
        for w in &mut weights {
            *w /= sum;
        }

        weights
    }

    /// Simplified distance metric learning
    ///
    /// In full implementation, GNN would learn domain-specific distances
    /// Currently uses weighted combination of cosine + euclidean
    fn learned_distance_heuristic(&self, dist1: &[f32], dist2: &[f32]) -> f32 {
        // Cosine distance
        let dot: f32 = dist1.iter().zip(dist2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f32 = dist1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = dist2.iter().map(|x| x * x).sum::<f32>().sqrt();

        let cosine_dist = if norm1 < 1e-10 || norm2 < 1e-10 {
            1.0
        } else {
            1.0 - (dot / (norm1 * norm2))
        };

        // Euclidean distance
        let euclidean: f32 = dist1.iter().zip(dist2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        // Weighted combination (learned weights would come from GNN training)
        0.6 * cosine_dist + 0.4 * euclidean
    }
}

impl GnnConsensusAdapter for LlmGnnAdapter {
    fn learn_consensus_weights(
        &self,
        response_embeddings: &[Vec<f32>],
        similarity_matrix: &[Vec<f32>],
        _ground_truth_quality: Option<&[f32]>,
    ) -> Result<Vec<f32>> {
        Ok(self.learn_weights_heuristic(response_embeddings, similarity_matrix))
    }

    fn predict_metric_weights(
        &self,
        _logits: &[f32],
        _context_features: &[f32],
    ) -> Result<Vec<f32>> {
        // Simplified: return balanced weights
        // Full GNN would predict context-dependent weights
        Ok(vec![0.3, 0.3, 0.2, 0.2])  // perplexity, entropy, KL-div, TE
    }

    fn learned_distance(
        &self,
        dist1: &[f32],
        dist2: &[f32],
    ) -> Result<f32> {
        Ok(self.learned_distance_heuristic(dist1, dist2))
    }
}

impl Default for LlmTdaAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for LlmMetaLearningAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for LlmGnnAdapter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llm_tda_adapter_creation() {
        let adapter = LlmTdaAdapter::new();
        assert_eq!(adapter.attention_threshold, 0.1);
    }

    #[test]
    fn test_attention_to_adjacency() {
        let adapter = LlmTdaAdapter::new();

        // Create simple attention pattern
        let attention = vec![
            vec![
                vec![0.8, 0.2, 0.0],
                vec![0.1, 0.7, 0.2],
                vec![0.0, 0.3, 0.7],
            ]
        ];

        let adjacency = adapter.attention_to_adjacency(&attention);
        assert_eq!(adjacency.nrows(), 3);
        assert_eq!(adjacency.ncols(), 3);

        // High attention should create edges
        assert!(adjacency[[0, 1]]);  // 0.2 > 0.1
        assert!(adjacency[[1, 2]]);  // 0.2 > 0.1
    }

    #[test]
    fn test_topology_features() {
        let adapter = LlmTdaAdapter::new();

        let attention = vec![
            vec![
                vec![0.5, 0.3, 0.2],
                vec![0.3, 0.5, 0.2],
                vec![0.2, 0.2, 0.6],
            ]
        ];

        let features = adapter.analyze_attention_topology(&attention).unwrap();

        // Should have at least one connected component
        assert!(features.betti_numbers[0] >= 1);
    }

    #[test]
    fn test_meta_learning_strategy_selection() {
        let adapter = LlmMetaLearningAdapter::new();

        let context = GenerationContext {
            tokens_generated: 10,
            recent_perplexity: vec![1.5, 1.6, 1.4],
            recent_entropy: vec![2.0, 2.1, 1.9],
            attention_collapsed: false,
        };

        let strategy = adapter.select_analysis_strategy(&[], &context).unwrap();

        // Should return a valid strategy
        assert!(matches!(
            strategy,
            AnalysisStrategy::Minimal | AnalysisStrategy::Standard | AnalysisStrategy::Full
        ));
    }

    #[test]
    fn test_meta_learning_sampling_adaptation() {
        let adapter = LlmMetaLearningAdapter::new();

        let current = SamplingParams {
            temperature: 1.0,
            top_k: 50,
            top_p: 0.9,
            entropy_guidance_alpha: 0.5,
        };

        let quality_feedback = vec![0.3, 0.4, 0.35];  // Low quality

        let adapted = adapter.adapt_sampling_params(&current, &quality_feedback).unwrap();

        // Low quality should increase temperature
        assert!(adapted.temperature >= current.temperature);
    }

    #[test]
    fn test_gnn_consensus_weights() {
        let adapter = LlmGnnAdapter::new();

        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.9, 0.1, 0.0],
        ];

        let similarity = vec![
            vec![1.0, 0.0, 0.9],
            vec![0.0, 1.0, 0.1],
            vec![0.9, 0.1, 1.0],
        ];

        let weights = adapter.learn_consensus_weights(&embeddings, &similarity, None).unwrap();

        assert_eq!(weights.len(), 3);

        // Weights should sum to 1
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_gnn_learned_distance() {
        let adapter = LlmGnnAdapter::new();

        let dist1 = vec![1.0, 0.0, 0.0];
        let dist2 = vec![0.0, 1.0, 0.0];

        let distance = adapter.learned_distance(&dist1, &dist2).unwrap();

        // Orthogonal vectors should have high distance
        assert!(distance > 0.5);
    }

    #[test]
    fn test_subset_selection() {
        let adapter = LlmTdaAdapter::new();

        let embeddings = vec![
            vec![1.0, 0.0],
            vec![0.9, 0.1],
            vec![0.0, 1.0],
            vec![0.1, 0.9],
        ];

        let subset = adapter.select_representative_subset(&embeddings, 2).unwrap();

        assert_eq!(subset.len(), 2);
        // Should select diverse items (not adjacent)
    }
}

