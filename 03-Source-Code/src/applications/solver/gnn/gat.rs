//! Graph Attention Network (GAT) - Worker 4
//!
//! Multi-head attention mechanism for problem graph encoding.
//! Enables transfer learning across different problem types by learning
//! structural patterns in the problem embedding space.
//!
//! # Architecture
//!
//! **Input**: Problem embedding (128-dim vector)
//! **Output**: Enriched embedding with attention-weighted features
//!
//! **Attention Mechanism**:
//! α_ij = softmax(LeakyReLU(a^T [W h_i || W h_j]))
//! h'_i = σ(Σ_j α_ij W h_j)
//!
//! Where:
//! - h_i: Node feature vector
//! - W: Weight matrix
//! - a: Attention vector
//! - α_ij: Attention coefficient from node j to node i
//! - ||: Concatenation operator

use anyhow::Result;
use ndarray::{Array1, Array2};
use std::f64;

use super::super::problem_embedding::EMBEDDING_DIM;

/// Number of attention heads
pub const NUM_HEADS: usize = 8;

/// Hidden dimension per attention head
pub const HEAD_DIM: usize = 16;

/// Graph Attention Layer configuration
#[derive(Debug, Clone)]
pub struct GatConfig {
    /// Number of attention heads
    pub num_heads: usize,

    /// Hidden dimension per head
    pub head_dim: usize,

    /// Input feature dimension
    pub input_dim: usize,

    /// Output feature dimension
    pub output_dim: usize,

    /// Negative slope for LeakyReLU
    pub negative_slope: f64,

    /// Dropout rate (0.0 = no dropout)
    pub dropout_rate: f64,
}

impl Default for GatConfig {
    fn default() -> Self {
        Self {
            num_heads: NUM_HEADS,
            head_dim: HEAD_DIM,
            input_dim: EMBEDDING_DIM,
            output_dim: EMBEDDING_DIM,
            negative_slope: 0.2,
            dropout_rate: 0.1,
        }
    }
}

/// Single attention head
pub struct AttentionHead {
    /// Weight matrix W (head_dim × input_dim)
    weight: Array2<f64>,

    /// Attention vector a (2 × head_dim)
    attention: Array2<f64>,

    /// Configuration
    config: GatConfig,

    /// Random seed for reproducibility
    seed: u64,
}

impl AttentionHead {
    /// Create a new attention head with random initialization
    pub fn new(config: GatConfig, seed: u64) -> Self {
        let weight = Self::initialize_weights(config.head_dim, config.input_dim, seed);
        let attention = Self::initialize_weights(2, config.head_dim, seed + 1);

        Self {
            weight,
            attention,
            config,
            seed,
        }
    }

    /// Initialize weights using Xavier/Glorot initialization
    fn initialize_weights(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
        let scale = (2.0 / (rows + cols) as f64).sqrt();
        let mut weights = Array2::zeros((rows, cols));

        for i in 0..rows {
            for j in 0..cols {
                let idx = (i * cols + j) as u64;
                let u = ((seed + idx) as f64 / u64::MAX as f64) * 2.0 - 1.0;
                weights[[i, j]] = u * scale;
            }
        }

        weights
    }

    /// Compute attention coefficients between two nodes
    pub fn compute_attention(
        &self,
        source_features: &Array1<f64>,
        target_features: &Array1<f64>,
    ) -> f64 {
        // Transform features: W h_i and W h_j
        let source_transformed = self.weight.dot(source_features);
        let target_transformed = self.weight.dot(target_features);

        // Concatenate [W h_i || W h_j]
        let mut concat = Array1::zeros(self.config.head_dim * 2);
        for i in 0..self.config.head_dim {
            concat[i] = source_transformed[i];
            concat[self.config.head_dim + i] = target_transformed[i];
        }

        // Compute attention score: a^T [W h_i || W h_j]
        let attention_flat = self.attention.row(0).to_owned();
        let score = attention_flat.iter().zip(concat.iter()).map(|(a, c)| a * c).sum();

        // Apply LeakyReLU
        self.leaky_relu(score)
    }

    /// LeakyReLU activation
    fn leaky_relu(&self, x: f64) -> f64 {
        if x >= 0.0 {
            x
        } else {
            self.config.negative_slope * x
        }
    }

    /// Apply attention to aggregate neighbor features
    pub fn aggregate(
        &self,
        node_features: &Array1<f64>,
        neighbor_features: &[Array1<f64>],
        attention_weights: &[f64],
    ) -> Array1<f64> {
        let mut aggregated = Array1::zeros(self.config.head_dim);

        // Weighted sum of transformed neighbor features
        for (neighbor, &weight) in neighbor_features.iter().zip(attention_weights.iter()) {
            let transformed = self.weight.dot(neighbor);
            aggregated = aggregated + transformed * weight;
        }

        // Apply non-linearity (ELU activation)
        aggregated.mapv(|x: f64| if x >= 0.0 { x } else { x.exp() - 1.0 })
    }
}

/// Multi-head Graph Attention Layer
pub struct GraphAttentionLayer {
    /// Multiple attention heads
    heads: Vec<AttentionHead>,

    /// Output projection matrix
    output_projection: Array2<f64>,

    /// Configuration
    config: GatConfig,
}

impl GraphAttentionLayer {
    /// Create a new GAT layer
    pub fn new(config: GatConfig, seed: u64) -> Self {
        let heads: Vec<AttentionHead> = (0..config.num_heads)
            .map(|i| AttentionHead::new(config.clone(), seed + i as u64 * 1000))
            .collect();

        // Output projection: (output_dim × (num_heads * head_dim))
        let projection_rows = config.output_dim;
        let projection_cols = config.num_heads * config.head_dim;
        let output_projection = AttentionHead::initialize_weights(
            projection_rows,
            projection_cols,
            seed + 10000,
        );

        Self {
            heads,
            output_projection,
            config,
        }
    }

    /// Forward pass: compute attention and aggregate features
    pub fn forward(
        &self,
        node_features: &Array1<f64>,
        neighbor_features: &[Array1<f64>],
    ) -> Result<Array1<f64>> {
        if neighbor_features.is_empty() {
            // No neighbors: return transformed self-features
            return self.forward_self_only(node_features);
        }

        let mut head_outputs = Vec::with_capacity(self.config.num_heads);

        // Process each attention head
        for head in &self.heads {
            // Compute attention coefficients for all neighbors
            let mut attention_scores: Vec<f64> = neighbor_features
                .iter()
                .map(|neighbor| head.compute_attention(node_features, neighbor))
                .collect();

            // Add self-attention
            attention_scores.push(head.compute_attention(node_features, node_features));

            // Softmax normalization
            let attention_weights = self.softmax(&attention_scores);

            // Create neighbor list including self
            let mut all_neighbors = neighbor_features.to_vec();
            all_neighbors.push(node_features.clone());

            // Aggregate features
            let head_output = head.aggregate(node_features, &all_neighbors, &attention_weights);
            head_outputs.push(head_output);
        }

        // Concatenate all head outputs
        let mut concat_output = Array1::zeros(self.config.num_heads * self.config.head_dim);
        for (i, head_output) in head_outputs.iter().enumerate() {
            let offset = i * self.config.head_dim;
            for j in 0..self.config.head_dim {
                concat_output[offset + j] = head_output[j];
            }
        }

        // Project to output dimension
        let output = self.output_projection.dot(&concat_output);

        Ok(output)
    }

    /// Forward pass for isolated node (no neighbors)
    fn forward_self_only(&self, node_features: &Array1<f64>) -> Result<Array1<f64>> {
        let mut head_outputs = Vec::with_capacity(self.config.num_heads);

        for head in &self.heads {
            let transformed = head.weight.dot(node_features);
            head_outputs.push(transformed);
        }

        // Concatenate
        let mut concat_output = Array1::zeros(self.config.num_heads * self.config.head_dim);
        for (i, head_output) in head_outputs.iter().enumerate() {
            let offset = i * self.config.head_dim;
            for j in 0..self.config.head_dim {
                concat_output[offset + j] = head_output[j];
            }
        }

        let output = self.output_projection.dot(&concat_output);
        Ok(output)
    }

    /// Softmax normalization
    fn softmax(&self, scores: &[f64]) -> Vec<f64> {
        if scores.is_empty() {
            return Vec::new();
        }

        // Numerical stability: subtract max
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum_exp: f64 = exp_scores.iter().sum();

        if sum_exp == 0.0 {
            // Fallback to uniform distribution
            vec![1.0 / scores.len() as f64; scores.len()]
        } else {
            exp_scores.iter().map(|&e| e / sum_exp).collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_attention_head_creation() {
        let config = GatConfig::default();
        let head = AttentionHead::new(config.clone(), 42);

        assert_eq!(head.weight.shape(), &[config.head_dim, config.input_dim]);
        assert_eq!(head.attention.shape(), &[2, config.head_dim]);
    }

    #[test]
    fn test_attention_computation() {
        let config = GatConfig::default();
        let head = AttentionHead::new(config, 42);

        let source = Array1::from_elem(EMBEDDING_DIM, 0.5);
        let target = Array1::from_elem(EMBEDDING_DIM, 0.3);

        let attention = head.compute_attention(&source, &target);
        assert!(attention.is_finite());
    }

    #[test]
    fn test_leaky_relu() {
        let config = GatConfig::default();
        let head = AttentionHead::new(config.clone(), 42);

        assert_eq!(head.leaky_relu(1.0), 1.0);
        assert_eq!(head.leaky_relu(-1.0), -config.negative_slope);
        assert_eq!(head.leaky_relu(0.0), 0.0);
    }

    #[test]
    fn test_gat_layer_creation() {
        let config = GatConfig::default();
        let layer = GraphAttentionLayer::new(config.clone(), 42);

        assert_eq!(layer.heads.len(), config.num_heads);
    }

    #[test]
    fn test_gat_forward_with_neighbors() {
        let config = GatConfig::default();
        let layer = GraphAttentionLayer::new(config, 42);

        let node = Array1::from_elem(EMBEDDING_DIM, 0.5);
        let neighbors = vec![
            Array1::from_elem(EMBEDDING_DIM, 0.3),
            Array1::from_elem(EMBEDDING_DIM, 0.7),
        ];

        let result = layer.forward(&node, &neighbors);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.len(), EMBEDDING_DIM);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_gat_forward_self_only() {
        let config = GatConfig::default();
        let layer = GraphAttentionLayer::new(config, 42);

        let node = Array1::from_elem(EMBEDDING_DIM, 0.5);
        let neighbors = vec![];

        let result = layer.forward(&node, &neighbors);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.len(), EMBEDDING_DIM);
    }

    #[test]
    fn test_softmax_normalization() {
        let config = GatConfig::default();
        let layer = GraphAttentionLayer::new(config, 42);

        let scores = vec![1.0, 2.0, 3.0];
        let weights = layer.softmax(&scores);

        // Should sum to 1.0
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Should be monotonically increasing for increasing scores
        assert!(weights[0] < weights[1]);
        assert!(weights[1] < weights[2]);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let config = GatConfig::default();
        let layer = GraphAttentionLayer::new(config, 42);

        // Large scores that could cause overflow
        let scores = vec![1000.0, 1001.0, 1002.0];
        let weights = layer.softmax(&scores);

        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(weights.iter().all(|&w| w.is_finite()));
    }
}
