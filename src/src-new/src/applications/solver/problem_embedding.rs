//! Problem Embedding System
//!
//! Converts diverse problem types into fixed-size feature vectors for:
//! - Problem similarity comparison
//! - Transfer learning via GNN
//! - Solution pattern matching
//! - Meta-learning across domains
//!
//! # Architecture
//!
//! Each problem type has a specialized embedding function that extracts:
//! - Structural features (size, connectivity, sparsity)
//! - Statistical features (mean, variance, distribution)
//! - Domain-specific features (financial metrics, graph properties)
//!
//! All embeddings are normalized to a fixed dimension (default: 128).

use anyhow::Result;
use ndarray::{Array1, Array2};
use std::collections::HashMap;

use super::{Problem, ProblemData, ProblemType};

/// Dimension of problem embeddings
pub const EMBEDDING_DIM: usize = 128;

/// Problem embedding - fixed-size feature vector representing a problem
#[derive(Debug, Clone)]
pub struct ProblemEmbedding {
    /// Feature vector (fixed size)
    pub features: Array1<f64>,

    /// Original problem type
    pub problem_type: ProblemType,

    /// Problem size/dimension
    pub dimension: usize,

    /// Additional metadata
    pub metadata: HashMap<String, f64>,
}

impl ProblemEmbedding {
    /// Create a new problem embedding
    pub fn new(features: Array1<f64>, problem_type: ProblemType, dimension: usize) -> Self {
        assert_eq!(features.len(), EMBEDDING_DIM, "Features must be {} dimensional", EMBEDDING_DIM);

        Self {
            features,
            problem_type,
            dimension,
            metadata: HashMap::new(),
        }
    }

    /// Compute cosine similarity with another embedding
    pub fn cosine_similarity(&self, other: &ProblemEmbedding) -> f64 {
        let dot_product = self.features.dot(&other.features);
        let norm_self = self.features.dot(&self.features).sqrt();
        let norm_other = other.features.dot(&other.features).sqrt();

        if norm_self == 0.0 || norm_other == 0.0 {
            0.0
        } else {
            dot_product / (norm_self * norm_other)
        }
    }

    /// Compute Euclidean distance to another embedding
    pub fn euclidean_distance(&self, other: &ProblemEmbedding) -> f64 {
        let diff = &self.features - &other.features;
        diff.dot(&diff).sqrt()
    }

    /// Check if problems are from the same type
    pub fn same_type(&self, other: &ProblemEmbedding) -> bool {
        self.problem_type == other.problem_type
    }
}

/// Problem embedding generator
pub struct ProblemEmbedder {
    /// Normalization statistics per problem type
    normalization_stats: HashMap<ProblemType, (Array1<f64>, Array1<f64>)>, // (mean, std)
}

impl ProblemEmbedder {
    /// Create a new problem embedder
    pub fn new() -> Self {
        Self {
            normalization_stats: HashMap::new(),
        }
    }

    /// Embed a problem into a fixed-size feature vector
    pub fn embed(&self, problem: &Problem) -> Result<ProblemEmbedding> {
        let raw_features = match &problem.data {
            ProblemData::Graph { adjacency_matrix, .. } => {
                self.embed_graph(adjacency_matrix)
            }
            ProblemData::Portfolio { assets, .. } => {
                self.embed_portfolio(assets)
            }
            ProblemData::Continuous { variables, bounds, .. } => {
                self.embed_continuous(variables.len(), bounds)
            }
            ProblemData::TimeSeries { series, horizon, .. } => {
                self.embed_timeseries(series, *horizon)
            }
            ProblemData::Discrete { variables, domains, .. } => {
                self.embed_discrete(variables.len(), domains)
            }
            ProblemData::Tabular { features, targets, .. } => {
                self.embed_tabular(features, targets.as_ref())
            }
        }?;

        // Normalize features
        let normalized = self.normalize_features(&raw_features);

        let dimension = problem.dimension().unwrap_or(0);

        Ok(ProblemEmbedding::new(
            normalized,
            problem.problem_type.clone(),
            dimension,
        ))
    }

    /// Embed a graph problem
    fn embed_graph(&self, adjacency: &Array2<bool>) -> Result<Array1<f64>> {
        let n = adjacency.nrows();
        let mut features = vec![0.0; EMBEDDING_DIM];

        // Basic structural features
        features[0] = n as f64; // Number of nodes

        // Edge count and density
        let edge_count: usize = adjacency.iter().filter(|&&x| x).count() / 2;
        features[1] = edge_count as f64;
        features[2] = edge_count as f64 / ((n * (n - 1)) as f64 / 2.0); // Density

        // Degree distribution
        let degrees: Vec<usize> = (0..n)
            .map(|i| adjacency.row(i).iter().filter(|&&x| x).count())
            .collect();

        features[3] = degrees.iter().sum::<usize>() as f64 / n as f64; // Mean degree
        features[4] = Self::compute_variance(&degrees); // Degree variance

        let max_degree = *degrees.iter().max().unwrap_or(&0);
        let min_degree = *degrees.iter().min().unwrap_or(&0);
        features[5] = max_degree as f64;
        features[6] = min_degree as f64;

        // Clustering coefficient (local)
        features[7] = self.estimate_clustering_coefficient(adjacency);

        // Graph diameter estimate (for small graphs)
        if n <= 100 {
            features[8] = self.estimate_diameter(adjacency);
        }

        // Connected components (simplified)
        features[9] = if edge_count > 0 { 1.0 } else { n as f64 };

        Ok(Array1::from_vec(features))
    }

    /// Embed a portfolio optimization problem
    fn embed_portfolio(&self, assets: &[super::AssetSpec]) -> Result<Array1<f64>> {
        let mut features = vec![0.0; EMBEDDING_DIM];

        let n_assets = assets.len();
        features[0] = n_assets as f64;

        if n_assets == 0 {
            return Ok(Array1::from_vec(features));
        }

        // Collect historical returns
        let returns: Vec<Vec<f64>> = assets.iter()
            .map(|a| a.historical_returns.clone())
            .collect();

        // Mean return statistics
        let mean_returns: Vec<f64> = returns.iter()
            .map(|r| if r.is_empty() { 0.0 } else { r.iter().sum::<f64>() / r.len() as f64 })
            .collect();

        features[1] = mean_returns.iter().sum::<f64>() / n_assets as f64; // Mean return
        features[2] = Self::compute_variance(&mean_returns.iter()
            .map(|&x| (x * 1000.0) as usize).collect::<Vec<_>>()); // Return variance

        // Volatility statistics
        let volatilities: Vec<f64> = returns.iter()
            .map(|r| {
                if r.len() < 2 { return 0.0; }
                let mean = r.iter().sum::<f64>() / r.len() as f64;
                let variance = r.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (r.len() - 1) as f64;
                variance.sqrt()
            })
            .collect();

        features[3] = volatilities.iter().sum::<f64>() / n_assets as f64; // Mean volatility
        features[4] = volatilities.iter().cloned().fold(f64::NEG_INFINITY, f64::max); // Max volatility
        features[5] = volatilities.iter().cloned().fold(f64::INFINITY, f64::min); // Min volatility

        // Price statistics
        let prices: Vec<f64> = assets.iter().map(|a| a.current_price).collect();
        features[6] = prices.iter().sum::<f64>() / n_assets as f64; // Mean price
        features[7] = prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max); // Max price
        features[8] = prices.iter().cloned().fold(f64::INFINITY, f64::min); // Min price

        // History length
        let history_lengths: Vec<usize> = returns.iter().map(|r| r.len()).collect();
        features[9] = history_lengths.iter().sum::<usize>() as f64 / n_assets as f64;

        Ok(Array1::from_vec(features))
    }

    /// Embed a continuous optimization problem
    fn embed_continuous(&self, dimension: usize, bounds: &[(f64, f64)]) -> Result<Array1<f64>> {
        let mut features = vec![0.0; EMBEDDING_DIM];

        features[0] = dimension as f64;

        if bounds.is_empty() {
            return Ok(Array1::from_vec(features));
        }

        // Bound statistics
        let lower_bounds: Vec<f64> = bounds.iter().map(|(l, _)| *l).collect();
        let upper_bounds: Vec<f64> = bounds.iter().map(|(_, u)| *u).collect();
        let ranges: Vec<f64> = bounds.iter().map(|(l, u)| u - l).collect();

        features[1] = lower_bounds.iter().sum::<f64>() / dimension as f64;
        features[2] = upper_bounds.iter().sum::<f64>() / dimension as f64;
        features[3] = ranges.iter().sum::<f64>() / dimension as f64; // Mean range
        features[4] = ranges.iter().cloned().fold(f64::NEG_INFINITY, f64::max); // Max range
        features[5] = ranges.iter().cloned().fold(f64::INFINITY, f64::min); // Min range

        // Check for unbounded dimensions
        let unbounded_count = bounds.iter()
            .filter(|(l, u)| l.is_infinite() || u.is_infinite())
            .count();
        features[6] = unbounded_count as f64;

        Ok(Array1::from_vec(features))
    }

    /// Embed a time series forecasting problem
    fn embed_timeseries(&self, series: &Array1<f64>, horizon: usize) -> Result<Array1<f64>> {
        let mut features = vec![0.0; EMBEDDING_DIM];

        let n = series.len();
        features[0] = n as f64;
        features[1] = horizon as f64;

        if n == 0 {
            return Ok(Array1::from_vec(features));
        }

        // Statistical features
        let mean = series.mean().unwrap_or(0.0);
        let std = series.std(0.0);

        features[2] = mean;
        features[3] = std;
        features[4] = series.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        features[5] = series.iter().cloned().fold(f64::INFINITY, f64::min);

        // Trend features
        if n > 1 {
            let trend = (series[n - 1] - series[0]) / n as f64;
            features[6] = trend;

            // Autocorrelation at lag 1
            if n > 2 {
                features[7] = self.compute_autocorrelation(series, 1);
            }
        }

        Ok(Array1::from_vec(features))
    }

    /// Embed a discrete optimization problem
    fn embed_discrete(&self, dimension: usize, domains: &[Vec<i64>]) -> Result<Array1<f64>> {
        let mut features = vec![0.0; EMBEDDING_DIM];

        features[0] = dimension as f64;

        if domains.is_empty() {
            return Ok(Array1::from_vec(features));
        }

        // Domain size statistics
        let domain_sizes: Vec<usize> = domains.iter().map(|d| d.len()).collect();
        features[1] = domain_sizes.iter().sum::<usize>() as f64 / dimension as f64;
        features[2] = *domain_sizes.iter().max().unwrap_or(&0) as f64;
        features[3] = *domain_sizes.iter().min().unwrap_or(&0) as f64;

        // Total search space size (log scale for large spaces)
        let search_space_log: f64 = domain_sizes.iter()
            .map(|&size| (size as f64).ln())
            .sum();
        features[4] = search_space_log;

        Ok(Array1::from_vec(features))
    }

    /// Embed a tabular problem
    fn embed_tabular(&self, features_matrix: &Array2<f64>, targets: Option<&Array1<f64>>) -> Result<Array1<f64>> {
        let mut features = vec![0.0; EMBEDDING_DIM];

        let (n_samples, n_features) = features_matrix.dim();
        features[0] = n_samples as f64;
        features[1] = n_features as f64;

        // Feature statistics
        for (i, col_idx) in (0..n_features.min(10)).enumerate() {
            let column = features_matrix.column(col_idx);
            let mean = column.mean().unwrap_or(0.0);
            let std = column.std(0.0);

            features[2 + i * 2] = mean;
            features[3 + i * 2] = std;
        }

        // Target statistics (if regression)
        if let Some(target_vec) = targets {
            features[22] = target_vec.mean().unwrap_or(0.0);
            features[23] = target_vec.std(0.0);
        }

        Ok(Array1::from_vec(features))
    }

    /// Normalize features to [0, 1] range
    fn normalize_features(&self, features: &Array1<f64>) -> Array1<f64> {
        let mut normalized = features.clone();

        // Min-max normalization
        let min = features.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = features.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if max > min {
            for x in normalized.iter_mut() {
                *x = (*x - min) / (max - min);
            }
        }

        normalized
    }

    /// Compute variance of integer values
    fn compute_variance(values: &[usize]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mean = values.iter().sum::<usize>() as f64 / values.len() as f64;
        values.iter()
            .map(|&x| {
                let diff = x as f64 - mean;
                diff * diff
            })
            .sum::<f64>() / values.len() as f64
    }

    /// Estimate clustering coefficient (simplified)
    fn estimate_clustering_coefficient(&self, adjacency: &Array2<bool>) -> f64 {
        let n = adjacency.nrows();
        if n < 3 {
            return 0.0;
        }

        // Sample a few nodes for efficiency
        let sample_size = 10.min(n);
        let mut total_coeff = 0.0;

        for i in 0..sample_size {
            let neighbors: Vec<usize> = (0..n)
                .filter(|&j| adjacency[[i, j]])
                .collect();

            if neighbors.len() < 2 {
                continue;
            }

            // Count triangles
            let mut triangles = 0;
            for &u in &neighbors {
                for &v in &neighbors {
                    if u < v && adjacency[[u, v]] {
                        triangles += 1;
                    }
                }
            }

            let k = neighbors.len();
            let possible_triangles = k * (k - 1) / 2;
            if possible_triangles > 0 {
                total_coeff += triangles as f64 / possible_triangles as f64;
            }
        }

        total_coeff / sample_size as f64
    }

    /// Estimate graph diameter (simplified BFS)
    fn estimate_diameter(&self, adjacency: &Array2<bool>) -> f64 {
        let n = adjacency.nrows();
        if n == 0 {
            return 0.0;
        }

        // Simple BFS from node 0
        let mut distances = vec![std::usize::MAX; n];
        distances[0] = 0;
        let mut queue = vec![0];
        let mut head = 0;

        while head < queue.len() {
            let u = queue[head];
            head += 1;

            for v in 0..n {
                if adjacency[[u, v]] && distances[v] == std::usize::MAX {
                    distances[v] = distances[u] + 1;
                    queue.push(v);
                }
            }
        }

        *distances.iter().filter(|&&d| d != std::usize::MAX).max().unwrap_or(&0) as f64
    }

    /// Compute autocorrelation at a given lag
    fn compute_autocorrelation(&self, series: &Array1<f64>, lag: usize) -> f64 {
        let n = series.len();
        if lag >= n {
            return 0.0;
        }

        let mean = series.mean().unwrap_or(0.0);
        let variance = series.iter().map(|&x| (x - mean).powi(2)).sum::<f64>();

        if variance == 0.0 {
            return 0.0;
        }

        let covariance: f64 = (0..n - lag)
            .map(|i| (series[i] - mean) * (series[i + lag] - mean))
            .sum();

        covariance / variance
    }
}

impl Default for ProblemEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::problem::ObjectiveFunction;

    #[test]
    fn test_graph_embedding() {
        let adjacency = Array2::from_shape_fn((4, 4), |(i, j)| i != j && (i + j) % 2 == 0);
        let problem = Problem::new(
            ProblemType::GraphProblem,
            "Test graph".to_string(),
            ProblemData::Graph {
                adjacency_matrix: adjacency,
                node_labels: None,
                edge_weights: None,
            },
        );

        let embedder = ProblemEmbedder::new();
        let embedding = embedder.embed(&problem).unwrap();

        assert_eq!(embedding.features.len(), EMBEDDING_DIM);
        assert_eq!(embedding.problem_type, ProblemType::GraphProblem);
        assert_eq!(embedding.dimension, 4);
    }

    #[test]
    fn test_continuous_embedding() {
        let problem = Problem::new(
            ProblemType::ContinuousOptimization,
            "Test continuous".to_string(),
            ProblemData::Continuous {
                variables: vec!["x".to_string(), "y".to_string()],
                bounds: vec![(-5.0, 5.0), (-3.0, 3.0)],
                objective: ObjectiveFunction::Minimize("x^2 + y^2".to_string()),
            },
        );

        let embedder = ProblemEmbedder::new();
        let embedding = embedder.embed(&problem).unwrap();

        assert_eq!(embedding.features.len(), EMBEDDING_DIM);
        assert_eq!(embedding.dimension, 2);
    }

    #[test]
    fn test_cosine_similarity() {
        let features1 = Array1::from_vec(vec![1.0; EMBEDDING_DIM]);
        let features2 = Array1::from_vec(vec![1.0; EMBEDDING_DIM]);
        let features3 = Array1::from_vec(vec![-1.0; EMBEDDING_DIM]);

        let emb1 = ProblemEmbedding::new(features1, ProblemType::GraphProblem, 10);
        let emb2 = ProblemEmbedding::new(features2, ProblemType::GraphProblem, 10);
        let emb3 = ProblemEmbedding::new(features3, ProblemType::GraphProblem, 10);

        // Identical embeddings should have similarity 1.0
        assert!((emb1.cosine_similarity(&emb2) - 1.0).abs() < 1e-6);

        // Opposite embeddings should have similarity -1.0
        assert!((emb1.cosine_similarity(&emb3) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance() {
        let features1 = Array1::from_vec(vec![0.0; EMBEDDING_DIM]);
        let features2 = Array1::from_vec(vec![1.0; EMBEDDING_DIM]);

        let emb1 = ProblemEmbedding::new(features1, ProblemType::GraphProblem, 10);
        let emb2 = ProblemEmbedding::new(features2, ProblemType::GraphProblem, 10);

        // Distance should be sqrt(128) since all dimensions differ by 1
        let expected_distance = (EMBEDDING_DIM as f64).sqrt();
        let actual_distance = emb1.euclidean_distance(&emb2);

        assert!((actual_distance - expected_distance).abs() < 1e-6);
    }
}
