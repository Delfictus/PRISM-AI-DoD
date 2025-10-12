//! Solution Pattern Storage - Worker 4
//!
//! Database for storing and retrieving successful (problem, solution) pairs.
//! Enables GNN-based transfer learning by finding similar past problems
//! and adapting their solutions to new problems.
//!
//! # Features
//!
//! - Store problem embeddings with their solutions
//! - Similarity search (cosine, euclidean, hybrid)
//! - Pattern matching across problem types
//! - Success rate tracking
//! - GPU-ready batch retrieval

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use ndarray::Array1;
use std::time::{SystemTime, UNIX_EPOCH};

use super::problem_embedding::{ProblemEmbedding, EMBEDDING_DIM};
use super::{Problem, Solution, ProblemType};

/// Similarity metric for pattern matching
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SimilarityMetric {
    /// Cosine similarity (angle-based)
    Cosine,
    /// Euclidean distance (L2 norm)
    Euclidean,
    /// Hybrid: weighted combination
    Hybrid,
}

/// A stored pattern: problem embedding + solution + metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolutionPattern {
    /// Unique pattern ID
    pub id: String,
    /// Problem embedding (128-dimensional feature vector)
    pub problem_embedding: Vec<f64>,
    /// Problem type
    pub problem_type: ProblemType,
    /// Solution data (serialized)
    pub solution_data: Vec<f64>,
    /// Objective value achieved
    pub objective_value: f64,
    /// Algorithm used
    pub algorithm: String,
    /// Computation time (ms)
    pub computation_time_ms: f64,
    /// Solution quality metrics
    pub quality_score: f64,
    /// Confidence score
    pub confidence: f64,
    /// Timestamp when pattern was stored
    pub timestamp: u64,
    /// Number of times this pattern was successfully reused
    pub reuse_count: usize,
    /// Success rate when applied to similar problems
    pub success_rate: f64,
}

impl SolutionPattern {
    /// Create a new solution pattern
    pub fn new(
        problem_embedding: ProblemEmbedding,
        solution: Solution,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let id = format!("pattern_{}_{}_{}",
            problem_embedding.problem_type.description().replace(" ", "_"),
            timestamp,
            rand::random::<u32>()
        );

        let quality_score = solution.metrics.quality_score;

        Self {
            id,
            problem_embedding: problem_embedding.features.to_vec(),
            problem_type: problem_embedding.problem_type,
            solution_data: solution.solution_vector.clone(),
            objective_value: solution.objective_value,
            algorithm: solution.algorithm_used,
            computation_time_ms: solution.computation_time_ms,
            quality_score,
            confidence: solution.confidence,
            timestamp,
            reuse_count: 0,
            success_rate: 1.0, // Initially assume 100% success
        }
    }

    /// Calculate similarity to a problem embedding
    pub fn similarity_to(
        &self,
        embedding: &ProblemEmbedding,
        metric: SimilarityMetric,
    ) -> f64 {
        let stored_embedding = Array1::from_vec(self.problem_embedding.clone());

        match metric {
            SimilarityMetric::Cosine => {
                // Cosine similarity: dot(a, b) / (||a|| * ||b||)
                let dot_product = stored_embedding.dot(&embedding.features);
                let norm_stored = stored_embedding.dot(&stored_embedding).sqrt();
                let norm_query = embedding.features.dot(&embedding.features).sqrt();

                if norm_stored == 0.0 || norm_query == 0.0 {
                    0.0
                } else {
                    dot_product / (norm_stored * norm_query)
                }
            }
            SimilarityMetric::Euclidean => {
                // Convert distance to similarity: 1 / (1 + distance)
                let diff = &stored_embedding - &embedding.features;
                let distance = diff.dot(&diff).sqrt();
                1.0 / (1.0 + distance)
            }
            SimilarityMetric::Hybrid => {
                // Weighted combination: 60% cosine + 40% euclidean
                let cosine = self.similarity_to(embedding, SimilarityMetric::Cosine);
                let euclidean = self.similarity_to(embedding, SimilarityMetric::Euclidean);
                0.6 * cosine + 0.4 * euclidean
            }
        }
    }

    /// Record successful reuse of this pattern
    pub fn record_success(&mut self) {
        self.reuse_count += 1;
        // Update success rate with exponential moving average
        self.success_rate = 0.9 * self.success_rate + 0.1 * 1.0;
    }

    /// Record failed reuse of this pattern
    pub fn record_failure(&mut self) {
        self.reuse_count += 1;
        // Update success rate with exponential moving average
        self.success_rate = 0.9 * self.success_rate + 0.1 * 0.0;
    }

    /// Get effectiveness score (combines quality, confidence, and success rate)
    pub fn effectiveness_score(&self) -> f64 {
        let recency_factor = if self.reuse_count > 0 {
            (self.reuse_count as f64).ln() + 1.0
        } else {
            1.0
        };

        (self.quality_score * 0.4 + self.confidence * 0.3 + self.success_rate * 0.3) * recency_factor
    }
}

/// Query configuration for pattern search
#[derive(Debug, Clone)]
pub struct PatternQuery {
    /// Problem embedding to match against
    pub embedding: ProblemEmbedding,
    /// Maximum number of patterns to return
    pub top_k: usize,
    /// Minimum similarity threshold
    pub min_similarity: f64,
    /// Similarity metric to use
    pub metric: SimilarityMetric,
    /// Filter by problem type (None = all types)
    pub filter_type: Option<ProblemType>,
    /// Prefer patterns with high reuse count
    pub prefer_proven: bool,
}

impl Default for PatternQuery {
    fn default() -> Self {
        Self {
            embedding: ProblemEmbedding {
                features: Array1::zeros(EMBEDDING_DIM),
                problem_type: ProblemType::Unknown,
                dimension: 0,
                metadata: HashMap::new(),
            },
            top_k: 5,
            min_similarity: 0.7,
            metric: SimilarityMetric::Hybrid,
            filter_type: None,
            prefer_proven: true,
        }
    }
}

/// Result of a pattern search
#[derive(Debug, Clone)]
pub struct PatternMatch {
    pub pattern: SolutionPattern,
    pub similarity: f64,
    pub effectiveness: f64,
    pub rank: usize,
}

/// Solution pattern database
pub struct PatternDatabase {
    /// All stored patterns
    patterns: Vec<SolutionPattern>,
    /// Index by problem type for faster lookup
    type_index: HashMap<ProblemType, Vec<usize>>,
    /// Maximum patterns to store (LRU eviction)
    max_patterns: usize,
}

impl PatternDatabase {
    /// Create a new pattern database
    pub fn new(max_patterns: usize) -> Self {
        Self {
            patterns: Vec::new(),
            type_index: HashMap::new(),
            max_patterns,
        }
    }

    /// Store a new solution pattern
    pub fn store(&mut self, pattern: SolutionPattern) -> Result<String> {
        // Check capacity and evict if necessary
        if self.patterns.len() >= self.max_patterns {
            self.evict_least_effective()?;
        }

        let pattern_id = pattern.id.clone();
        let problem_type = pattern.problem_type.clone();

        // Add to patterns list
        let pattern_idx = self.patterns.len();
        self.patterns.push(pattern);

        // Update type index
        self.type_index
            .entry(problem_type)
            .or_insert_with(Vec::new)
            .push(pattern_idx);

        Ok(pattern_id)
    }

    /// Query for similar patterns
    pub fn query(&self, query: PatternQuery) -> Vec<PatternMatch> {
        // Determine which patterns to search
        let search_indices: Vec<usize> = if let Some(filter_type) = &query.filter_type {
            // Search only patterns of specific type
            self.type_index
                .get(filter_type)
                .map(|indices| indices.clone())
                .unwrap_or_default()
        } else {
            // Search all patterns
            (0..self.patterns.len()).collect()
        };

        // Calculate similarities
        let mut matches: Vec<(usize, f64, f64)> = search_indices
            .iter()
            .filter_map(|&idx| {
                let pattern = &self.patterns[idx];
                let similarity = pattern.similarity_to(&query.embedding, query.metric);

                if similarity >= query.min_similarity {
                    let effectiveness = if query.prefer_proven {
                        pattern.effectiveness_score()
                    } else {
                        1.0
                    };
                    Some((idx, similarity, effectiveness))
                } else {
                    None
                }
            })
            .collect();

        // Sort by combined score: similarity * effectiveness
        matches.sort_by(|a, b| {
            let score_a = a.1 * a.2;
            let score_b = b.1 * b.2;
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take top K
        matches
            .into_iter()
            .take(query.top_k)
            .enumerate()
            .map(|(rank, (idx, similarity, effectiveness))| PatternMatch {
                pattern: self.patterns[idx].clone(),
                similarity,
                effectiveness,
                rank: rank + 1,
            })
            .collect()
    }

    /// Update pattern after reuse
    pub fn update_pattern(&mut self, pattern_id: &str, success: bool) -> Result<()> {
        let pattern = self.patterns
            .iter_mut()
            .find(|p| p.id == pattern_id)
            .ok_or_else(|| anyhow!("Pattern not found: {}", pattern_id))?;

        if success {
            pattern.record_success();
        } else {
            pattern.record_failure();
        }

        Ok(())
    }

    /// Get database statistics
    pub fn stats(&self) -> DatabaseStats {
        let mut type_counts: HashMap<ProblemType, usize> = HashMap::new();
        let mut total_reuse = 0;

        for pattern in &self.patterns {
            *type_counts.entry(pattern.problem_type.clone()).or_insert(0) += 1;
            total_reuse += pattern.reuse_count;
        }

        let avg_quality = if self.patterns.is_empty() {
            0.0
        } else {
            self.patterns.iter().map(|p| p.quality_score).sum::<f64>() / self.patterns.len() as f64
        };

        DatabaseStats {
            total_patterns: self.patterns.len(),
            patterns_by_type: type_counts,
            total_reuse_count: total_reuse,
            average_quality: avg_quality,
            capacity: self.max_patterns,
        }
    }

    /// Evict least effective pattern (LRU with quality weighting)
    fn evict_least_effective(&mut self) -> Result<()> {
        if self.patterns.is_empty() {
            return Ok(());
        }

        // Find pattern with lowest effectiveness score
        let (worst_idx, worst_type) = self.patterns
            .iter()
            .enumerate()
            .min_by(|a, b| {
                let score_a = a.1.effectiveness_score();
                let score_b = b.1.effectiveness_score();
                score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, p)| (idx, p.problem_type.clone()))
            .unwrap();

        // Remove from patterns list
        self.patterns.remove(worst_idx);

        // Update type index
        if let Some(indices) = self.type_index.get_mut(&worst_type) {
            if let Some(pos) = indices.iter().position(|&idx| idx == worst_idx) {
                indices.remove(pos);
            }
            // Adjust indices after removal
            for idx in indices.iter_mut() {
                if *idx > worst_idx {
                    *idx -= 1;
                }
            }
        }

        Ok(())
    }

    /// Clear all patterns
    pub fn clear(&mut self) {
        self.patterns.clear();
        self.type_index.clear();
    }

    /// Get pattern by ID
    pub fn get_pattern(&self, pattern_id: &str) -> Option<&SolutionPattern> {
        self.patterns.iter().find(|p| p.id == pattern_id)
    }

    /// Export patterns to JSON
    pub fn export_json(&self) -> Result<String> {
        serde_json::to_string_pretty(&self.patterns)
            .map_err(|e| anyhow!("Failed to serialize patterns: {}", e))
    }

    /// Import patterns from JSON
    pub fn import_json(&mut self, json: &str) -> Result<usize> {
        let patterns: Vec<SolutionPattern> = serde_json::from_str(json)
            .map_err(|e| anyhow!("Failed to deserialize patterns: {}", e))?;

        let count = patterns.len();
        for pattern in patterns {
            self.store(pattern)?;
        }

        Ok(count)
    }
}

/// Database statistics
#[derive(Debug, Clone)]
pub struct DatabaseStats {
    pub total_patterns: usize,
    pub patterns_by_type: HashMap<ProblemType, usize>,
    pub total_reuse_count: usize,
    pub average_quality: f64,
    pub capacity: usize,
}

impl DatabaseStats {
    pub fn utilization(&self) -> f64 {
        self.total_patterns as f64 / self.capacity as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::applications::solver::problem::{Problem, ProblemData};
    use crate::applications::solver::solution::{Solution, SolutionMetrics};
    use crate::applications::solver::problem_embedding::ProblemEmbedder;
    use ndarray::Array2;

    fn create_test_pattern(problem_type: ProblemType, quality: f64) -> SolutionPattern {
        let problem = match problem_type {
            ProblemType::GraphProblem => {
                Problem::new(
                    problem_type.clone(),
                    ProblemData::Graph {
                        adjacency_matrix: Array2::from_shape_fn((5, 5), |(i, j)| i != j),
                        node_count: 5,
                    },
                )
            }
            _ => {
                Problem::new(
                    problem_type.clone(),
                    ProblemData::Continuous {
                        dimension: 10,
                        bounds: vec![(0.0, 1.0); 10],
                    },
                )
            }
        };

        let embedder = ProblemEmbedder::new();
        let embedding = embedder.embed(&problem).unwrap();

        let solution = Solution::new(
            problem_type,
            1.0,
            vec![0.5; 10],
            "TestAlgorithm".to_string(),
            100.0,
        )
        .with_metrics(SolutionMetrics {
            iterations: 100,
            convergence_rate: 0.95,
            is_optimal: false,
            optimality_gap: Some(0.01),
            constraints_satisfied: 10,
            total_constraints: 10,
            quality_score: quality,
        });

        SolutionPattern::new(embedding, solution)
    }

    #[test]
    fn test_pattern_creation() {
        let pattern = create_test_pattern(ProblemType::GraphProblem, 0.85);

        assert_eq!(pattern.problem_type, ProblemType::GraphProblem);
        assert_eq!(pattern.problem_embedding.len(), EMBEDDING_DIM);
        assert_eq!(pattern.quality_score, 0.85);
        assert_eq!(pattern.reuse_count, 0);
        assert_eq!(pattern.success_rate, 1.0);
    }

    #[test]
    fn test_database_store_and_query() {
        let mut db = PatternDatabase::new(100);

        // Store patterns
        let pattern1 = create_test_pattern(ProblemType::GraphProblem, 0.9);
        let pattern2 = create_test_pattern(ProblemType::GraphProblem, 0.7);
        let pattern3 = create_test_pattern(ProblemType::ContinuousOptimization, 0.8);

        db.store(pattern1).unwrap();
        db.store(pattern2).unwrap();
        db.store(pattern3).unwrap();

        // Query for graph problems
        let problem = Problem::new(
            ProblemType::GraphProblem,
            ProblemData::Graph {
                adjacency_matrix: Array2::from_shape_fn((5, 5), |(i, j)| i != j),
                node_count: 5,
            },
        );

        let embedder = ProblemEmbedder::new();
        let embedding = embedder.embed(&problem).unwrap();

        let query = PatternQuery {
            embedding,
            top_k: 2,
            min_similarity: 0.0,
            metric: SimilarityMetric::Cosine,
            filter_type: Some(ProblemType::GraphProblem),
            prefer_proven: false,
        };

        let matches = db.query(query);
        assert_eq!(matches.len(), 2);
        assert!(matches[0].similarity >= matches[1].similarity);
    }

    #[test]
    fn test_pattern_success_tracking() {
        let mut pattern = create_test_pattern(ProblemType::GraphProblem, 0.8);

        let initial_success_rate = pattern.success_rate;
        pattern.record_success();
        assert_eq!(pattern.reuse_count, 1);
        assert!(pattern.success_rate >= initial_success_rate);

        pattern.record_failure();
        assert_eq!(pattern.reuse_count, 2);
    }

    #[test]
    fn test_database_eviction() {
        let mut db = PatternDatabase::new(3);

        // Store 4 patterns (should trigger eviction)
        for i in 0..4 {
            let quality = 0.5 + (i as f64 * 0.1);
            let pattern = create_test_pattern(ProblemType::GraphProblem, quality);
            db.store(pattern).unwrap();
        }

        // Should have only 3 patterns
        assert_eq!(db.patterns.len(), 3);
    }

    #[test]
    fn test_similarity_metrics() {
        let pattern1 = create_test_pattern(ProblemType::GraphProblem, 0.8);
        let pattern2 = create_test_pattern(ProblemType::GraphProblem, 0.9);

        let embedder = ProblemEmbedder::new();
        let problem = Problem::new(
            ProblemType::GraphProblem,
            ProblemData::Graph {
                adjacency_matrix: Array2::from_shape_fn((5, 5), |(i, j)| i != j),
                node_count: 5,
            },
        );
        let embedding = embedder.embed(&problem).unwrap();

        let cosine_sim = pattern1.similarity_to(&embedding, SimilarityMetric::Cosine);
        let euclidean_sim = pattern1.similarity_to(&embedding, SimilarityMetric::Euclidean);
        let hybrid_sim = pattern1.similarity_to(&embedding, SimilarityMetric::Hybrid);

        assert!(cosine_sim >= 0.0 && cosine_sim <= 1.0);
        assert!(euclidean_sim >= 0.0 && euclidean_sim <= 1.0);
        assert!(hybrid_sim >= 0.0 && hybrid_sim <= 1.0);
    }

    #[test]
    fn test_database_stats() {
        let mut db = PatternDatabase::new(100);

        db.store(create_test_pattern(ProblemType::GraphProblem, 0.9)).unwrap();
        db.store(create_test_pattern(ProblemType::GraphProblem, 0.8)).unwrap();
        db.store(create_test_pattern(ProblemType::ContinuousOptimization, 0.85)).unwrap();

        let stats = db.stats();
        assert_eq!(stats.total_patterns, 3);
        assert_eq!(stats.patterns_by_type.len(), 2);
        assert!(stats.average_quality > 0.8);
    }
}
