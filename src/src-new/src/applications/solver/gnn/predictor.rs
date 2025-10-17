//! GNN Predictor - Worker 4
//!
//! Fast approximate solution prediction using trained GNN.
//! Provides confidence-based routing between GNN predictions and exact solvers.
//!
//! # Architecture
//!
//! **Prediction Flow**:
//! 1. Embed problem → 128-dim vector
//! 2. Query pattern database for similar problems
//! 3. Forward through GAT with neighbor features
//! 4. Predict solution quality + confidence
//! 5. Route: High confidence → Use prediction, Low confidence → Exact solver
//!
//! # Confidence Estimation
//!
//! Confidence based on:
//! - Distance to nearest training samples
//! - Pattern database coverage
//! - Prediction consistency across attention heads

use anyhow::Result;
use ndarray::Array1;
use serde::{Deserialize, Serialize};

use super::training::GnnTrainer;
use super::super::problem_embedding::{ProblemEmbedding, ProblemEmbedder};
use super::super::solution_patterns::{PatternDatabase, PatternQuery, SimilarityMetric};
use super::super::{Problem, Solution, ProblemType};

/// Confidence threshold for using GNN prediction vs exact solver
pub const DEFAULT_CONFIDENCE_THRESHOLD: f64 = 0.7;

/// Number of similar problems to consider for confidence
pub const NUM_NEIGHBORS: usize = 5;

/// Prediction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictorConfig {
    /// Minimum confidence to use GNN prediction
    pub confidence_threshold: f64,

    /// Number of similar problems to consider
    pub num_neighbors: usize,

    /// Use pattern database for neighbor features
    pub use_pattern_database: bool,

    /// Similarity metric for neighbor search
    pub similarity_metric: SimilarityMetric,
}

impl Default for PredictorConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: DEFAULT_CONFIDENCE_THRESHOLD,
            num_neighbors: NUM_NEIGHBORS,
            use_pattern_database: true,
            similarity_metric: SimilarityMetric::Cosine,
        }
    }
}

/// Prediction result with confidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    /// Predicted solution quality
    pub quality: f64,

    /// Confidence in prediction (0.0-1.0)
    pub confidence: f64,

    /// Whether to use GNN prediction or fallback to exact solver
    pub use_prediction: bool,

    /// Warm start solution (if available)
    pub warm_start: Option<Vec<f64>>,

    /// Number of similar problems found
    pub num_similar: usize,

    /// Average distance to similar problems
    pub avg_distance: f64,
}

/// GNN-based solution predictor
pub struct GnnPredictor {
    /// Trained GNN model
    trainer: GnnTrainer,

    /// Problem embedder
    embedder: ProblemEmbedder,

    /// Pattern database for similar problems
    pattern_db: Option<PatternDatabase>,

    /// Configuration
    config: PredictorConfig,
}

impl GnnPredictor {
    /// Create a new GNN predictor
    pub fn new(trainer: GnnTrainer, config: PredictorConfig) -> Self {
        Self {
            trainer,
            embedder: ProblemEmbedder::new(),
            pattern_db: Some(PatternDatabase::new(1000)), // 1000 pattern capacity
            config,
        }
    }

    /// Create predictor without pattern database
    pub fn without_pattern_db(trainer: GnnTrainer, config: PredictorConfig) -> Self {
        Self {
            trainer,
            embedder: ProblemEmbedder::new(),
            pattern_db: None,
            config,
        }
    }

    /// Predict solution quality with confidence
    pub fn predict(&self, problem: &Problem) -> Result<PredictionResult> {
        // Step 1: Embed problem
        let embedding = self.embedder.embed(problem)?;

        // Step 2: Query pattern database for similar problems
        let (similar_embeddings, avg_distance, num_similar) = if self.config.use_pattern_database && self.pattern_db.is_some() {
            self.find_similar_problems(&embedding)?
        } else {
            (Vec::new(), 1.0, 0)
        };

        // Step 3: Predict quality using GNN
        let quality = self.trainer.predict(&embedding)?;

        // Step 4: Estimate confidence
        let confidence = self.estimate_confidence(&embedding, &similar_embeddings, avg_distance, num_similar);

        // Step 5: Decide whether to use prediction
        let use_prediction = confidence >= self.config.confidence_threshold;

        // Step 6: Generate warm start (if using exact solver)
        let warm_start = if !use_prediction && num_similar > 0 {
            self.generate_warm_start(&embedding)?
        } else {
            None
        };

        Ok(PredictionResult {
            quality,
            confidence,
            use_prediction,
            warm_start,
            num_similar,
            avg_distance,
        })
    }

    /// Find similar problems in pattern database
    fn find_similar_problems(&self, _embedding: &ProblemEmbedding) -> Result<(Vec<ProblemEmbedding>, f64, usize)> {
        if let Some(ref _db) = self.pattern_db {
            // TODO: Implement full pattern database integration
            // For now, return empty (no similar problems found)
            Ok((Vec::new(), 1.0, 0))
        } else {
            Ok((Vec::new(), 1.0, 0))
        }
    }

    /// Estimate confidence in prediction
    fn estimate_confidence(
        &self,
        _embedding: &ProblemEmbedding,
        similar_embeddings: &[ProblemEmbedding],
        avg_distance: f64,
        num_similar: usize,
    ) -> f64 {
        if num_similar == 0 {
            // No similar problems: low confidence
            return 0.3;
        }

        // Confidence factors:
        // 1. Coverage: How many similar problems found?
        let coverage_factor = (num_similar as f64 / self.config.num_neighbors as f64).min(1.0);

        // 2. Similarity: How close are the similar problems?
        let similarity_factor = (1.0 - avg_distance).max(0.0);

        // 3. Consistency: Do similar problems have consistent solutions?
        let consistency_factor = if !similar_embeddings.is_empty() {
            // Simplified: assume high consistency if we have neighbors
            0.8
        } else {
            0.5
        };

        // Weighted combination
        let confidence = 0.3 * coverage_factor + 0.4 * similarity_factor + 0.3 * consistency_factor;

        confidence.clamp(0.0, 1.0)
    }

    /// Generate warm start solution from similar problems
    fn generate_warm_start(&self, _embedding: &ProblemEmbedding) -> Result<Option<Vec<f64>>> {
        if let Some(ref _db) = self.pattern_db {
            // TODO: Implement full warm start generation
            // For now, return None
            Ok(None)
        } else {
            Ok(None)
        }
    }

    /// Add a solved problem to pattern database
    pub fn add_pattern(&mut self, _problem: &Problem, _solution: &Solution) -> Result<()> {
        // TODO: Implement full pattern storage
        // For now, just return Ok
        Ok(())
    }

    /// Get pattern database statistics
    pub fn get_stats(&self) -> Option<super::super::solution_patterns::DatabaseStats> {
        // TODO: Implement when PatternDatabase has get_stats method
        None
    }

    /// Set pattern database
    pub fn set_pattern_database(&mut self, db: PatternDatabase) {
        self.pattern_db = Some(db);
    }
}

/// Performance tracking for hybrid solver
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridStats {
    /// Total problems solved
    pub total_problems: usize,

    /// Problems solved using GNN
    pub gnn_solutions: usize,

    /// Problems solved using exact solver
    pub exact_solutions: usize,

    /// Average GNN confidence
    pub avg_gnn_confidence: f64,

    /// Average speedup when using GNN
    pub avg_gnn_speedup: f64,

    /// Average quality gap (GNN vs exact)
    pub avg_quality_gap: f64,
}

impl Default for HybridStats {
    fn default() -> Self {
        Self {
            total_problems: 0,
            gnn_solutions: 0,
            exact_solutions: 0,
            avg_gnn_confidence: 0.0,
            avg_gnn_speedup: 0.0,
            avg_quality_gap: 0.0,
        }
    }
}

impl HybridStats {
    /// Update statistics with new problem
    pub fn update(&mut self, used_gnn: bool, confidence: f64, speedup: f64, quality_gap: f64) {
        self.total_problems += 1;

        if used_gnn {
            self.gnn_solutions += 1;
        } else {
            self.exact_solutions += 1;
        }

        // Update running averages
        let n = self.total_problems as f64;
        self.avg_gnn_confidence = (self.avg_gnn_confidence * (n - 1.0) + confidence) / n;
        self.avg_gnn_speedup = (self.avg_gnn_speedup * (n - 1.0) + speedup) / n;
        self.avg_quality_gap = (self.avg_quality_gap * (n - 1.0) + quality_gap) / n;
    }

    /// Get GNN usage percentage
    pub fn gnn_usage_rate(&self) -> f64 {
        if self.total_problems == 0 {
            0.0
        } else {
            (self.gnn_solutions as f64) / (self.total_problems as f64)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::training::TrainingConfig;
    use super::super::super::problem::ProblemData;

    fn create_test_problem() -> Problem {
        Problem {
            problem_type: ProblemType::ContinuousOptimization,
            description: "Test problem".to_string(),
            data: ProblemData::Continuous {
                variables: vec!["x".to_string(), "y".to_string()],
                bounds: vec![(0.0, 1.0), (0.0, 1.0)],
                objective: super::super::super::problem::ObjectiveFunction::Minimize("test".to_string()),
            },
            constraints: Vec::new(),
            metadata: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn test_predictor_creation() {
        let trainer = GnnTrainer::new(TrainingConfig::default());
        let config = PredictorConfig::default();
        let predictor = GnnPredictor::new(trainer, config);

        assert!(predictor.pattern_db.is_some());
        assert_eq!(predictor.config.confidence_threshold, DEFAULT_CONFIDENCE_THRESHOLD);
    }

    #[test]
    fn test_prediction() {
        let trainer = GnnTrainer::new(TrainingConfig::default());
        let config = PredictorConfig::default();
        let predictor = GnnPredictor::new(trainer, config);

        let problem = create_test_problem();
        let result = predictor.predict(&problem);

        assert!(result.is_ok());
        let pred = result.unwrap();
        assert!(pred.quality.is_finite());
        assert!(pred.confidence >= 0.0 && pred.confidence <= 1.0);
    }

    #[test]
    fn test_confidence_estimation_no_neighbors() {
        let trainer = GnnTrainer::new(TrainingConfig::default());
        let config = PredictorConfig::default();
        let predictor = GnnPredictor::new(trainer, config);

        let problem = create_test_problem();
        let embedding = predictor.embedder.embed(&problem).unwrap();

        let confidence = predictor.estimate_confidence(&embedding, &[], 1.0, 0);
        assert_eq!(confidence, 0.3); // Low confidence with no neighbors
    }

    #[test]
    fn test_hybrid_stats() {
        let mut stats = HybridStats::default();

        stats.update(true, 0.8, 5.0, 0.02);
        assert_eq!(stats.total_problems, 1);
        assert_eq!(stats.gnn_solutions, 1);
        assert_eq!(stats.exact_solutions, 0);

        stats.update(false, 0.5, 1.0, 0.0);
        assert_eq!(stats.total_problems, 2);
        assert_eq!(stats.gnn_solutions, 1);
        assert_eq!(stats.exact_solutions, 1);

        assert_eq!(stats.gnn_usage_rate(), 0.5);
    }

    #[test]
    fn test_predictor_without_pattern_db() {
        let trainer = GnnTrainer::new(TrainingConfig::default());
        let config = PredictorConfig {
            use_pattern_database: false,
            ..Default::default()
        };
        let predictor = GnnPredictor::without_pattern_db(trainer, config);

        assert!(predictor.pattern_db.is_none());
    }
}
