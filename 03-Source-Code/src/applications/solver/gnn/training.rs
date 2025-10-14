//! GNN Training Infrastructure - Worker 4
//!
//! Training loop and optimization for Graph Attention Networks.
//! Enables learning from problem-solution patterns for transfer learning.
//!
//! # Training Objective
//!
//! **Loss Function**:
//! L = MSE(predicted_quality, actual_quality) + λ × RankingLoss
//!
//! Where:
//! - MSE: Mean squared error for solution quality
//! - RankingLoss: Pairwise ranking loss to preserve solution ordering
//! - λ: Ranking loss weight
//!
//! # Training Strategy
//!
//! 1. **Data Collection**: Gather (problem, solution, quality) tuples
//! 2. **Mini-batch Training**: Process batches for gradient stability
//! 3. **Early Stopping**: Monitor validation loss
//! 4. **Checkpointing**: Save best model weights

use anyhow::Result;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use super::gat::{GraphAttentionLayer, GatConfig};
use super::super::problem_embedding::ProblemEmbedding;

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f64,

    /// Batch size
    pub batch_size: usize,

    /// Maximum number of epochs
    pub max_epochs: usize,

    /// Early stopping patience (epochs without improvement)
    pub patience: usize,

    /// Ranking loss weight
    pub ranking_loss_weight: f64,

    /// Validation split (0.0-1.0)
    pub validation_split: f64,

    /// Random seed
    pub seed: u64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            batch_size: 32,
            max_epochs: 100,
            patience: 10,
            ranking_loss_weight: 0.1,
            validation_split: 0.2,
            seed: 42,
        }
    }
}

/// Training sample: (problem embedding, solution quality, metadata)
#[derive(Debug, Clone)]
pub struct TrainingSample {
    /// Problem embedding
    pub problem: ProblemEmbedding,

    /// Solution quality score (higher is better)
    pub quality: f64,

    /// Optional: Actual solution for analysis
    pub solution: Option<Vec<f64>>,
}

/// Training history for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingHistory {
    /// Training loss per epoch
    pub train_losses: Vec<f64>,

    /// Validation loss per epoch
    pub val_losses: Vec<f64>,

    /// Best validation loss achieved
    pub best_val_loss: f64,

    /// Epoch where best loss occurred
    pub best_epoch: usize,

    /// Total epochs trained
    pub epochs_trained: usize,
}

/// GNN Trainer
pub struct GnnTrainer {
    /// GAT layer
    gat_layer: GraphAttentionLayer,

    /// Prediction head (linear layer: embedding_dim → 1)
    prediction_weights: Array1<f64>,

    /// Training configuration
    config: TrainingConfig,

    /// Training history
    history: TrainingHistory,
}

impl GnnTrainer {
    /// Create a new GNN trainer
    pub fn new(config: TrainingConfig) -> Self {
        let gat_config = GatConfig::default();
        let gat_layer = GraphAttentionLayer::new(gat_config.clone(), config.seed);

        // Initialize prediction head
        let prediction_weights = Self::initialize_prediction_head(gat_config.output_dim, config.seed);

        Self {
            gat_layer,
            prediction_weights,
            config,
            history: TrainingHistory {
                train_losses: Vec::new(),
                val_losses: Vec::new(),
                best_val_loss: f64::INFINITY,
                best_epoch: 0,
                epochs_trained: 0,
            },
        }
    }

    /// Initialize prediction head weights
    fn initialize_prediction_head(dim: usize, seed: u64) -> Array1<f64> {
        let scale = (2.0 / dim as f64).sqrt();
        let mut weights = Array1::zeros(dim);

        for i in 0..dim {
            let u = ((seed + i as u64) as f64 / u64::MAX as f64) * 2.0 - 1.0;
            weights[i] = u * scale;
        }

        weights
    }

    /// Train the GNN on a dataset
    pub fn train(&mut self, samples: Vec<TrainingSample>) -> Result<TrainingHistory> {
        if samples.is_empty() {
            anyhow::bail!("Cannot train on empty dataset");
        }

        // Split into train and validation
        let split_idx = ((samples.len() as f64) * (1.0 - self.config.validation_split)) as usize;
        let (train_samples, val_samples) = samples.split_at(split_idx);

        if val_samples.is_empty() {
            anyhow::bail!("Validation set is empty, increase dataset size or reduce validation_split");
        }

        let mut epochs_without_improvement = 0;
        let mut best_weights = self.prediction_weights.clone();

        // Training loop
        for epoch in 0..self.config.max_epochs {
            // Train for one epoch
            let train_loss = self.train_epoch(train_samples)?;

            // Validate
            let val_loss = self.validate(val_samples)?;

            // Update history
            self.history.train_losses.push(train_loss);
            self.history.val_losses.push(val_loss);
            self.history.epochs_trained = epoch + 1;

            // Check for improvement
            if val_loss < self.history.best_val_loss {
                self.history.best_val_loss = val_loss;
                self.history.best_epoch = epoch;
                best_weights = self.prediction_weights.clone();
                epochs_without_improvement = 0;
            } else {
                epochs_without_improvement += 1;
            }

            // Early stopping
            if epochs_without_improvement >= self.config.patience {
                break;
            }
        }

        // Restore best weights
        self.prediction_weights = best_weights;

        Ok(self.history.clone())
    }

    /// Train for one epoch
    fn train_epoch(&mut self, samples: &[TrainingSample]) -> Result<f64> {
        let mut total_loss = 0.0;
        let num_batches = (samples.len() + self.config.batch_size - 1) / self.config.batch_size;

        for batch_idx in 0..num_batches {
            let start = batch_idx * self.config.batch_size;
            let end = (start + self.config.batch_size).min(samples.len());
            let batch = &samples[start..end];

            // Forward pass
            let (predictions, targets) = self.forward_batch(batch)?;

            // Compute loss
            let mse_loss = self.compute_mse_loss(&predictions, &targets);
            let ranking_loss = self.compute_ranking_loss(&predictions, &targets);
            let batch_loss = mse_loss + self.config.ranking_loss_weight * ranking_loss;

            total_loss += batch_loss;

            // Backward pass (simplified gradient descent)
            self.backward_batch(batch, &predictions, &targets)?;
        }

        Ok(total_loss / num_batches as f64)
    }

    /// Validate on validation set
    fn validate(&self, samples: &[TrainingSample]) -> Result<f64> {
        let (predictions, targets) = self.forward_batch(samples)?;
        let mse_loss = self.compute_mse_loss(&predictions, &targets);
        let ranking_loss = self.compute_ranking_loss(&predictions, &targets);

        Ok(mse_loss + self.config.ranking_loss_weight * ranking_loss)
    }

    /// Forward pass for a batch
    fn forward_batch(&self, batch: &[TrainingSample]) -> Result<(Vec<f64>, Vec<f64>)> {
        let mut predictions = Vec::with_capacity(batch.len());
        let mut targets = Vec::with_capacity(batch.len());

        for sample in batch {
            let prediction = self.predict_quality(&sample.problem)?;
            predictions.push(prediction);
            targets.push(sample.quality);
        }

        Ok((predictions, targets))
    }

    /// Predict solution quality for a problem
    fn predict_quality(&self, problem: &ProblemEmbedding) -> Result<f64> {
        // Forward through GAT (no neighbors for now, using self-attention only)
        let gat_output = self.gat_layer.forward(&problem.features, &[])?;

        // Linear prediction head
        let quality = self.prediction_weights.dot(&gat_output);

        Ok(quality)
    }

    /// Compute mean squared error loss
    fn compute_mse_loss(&self, predictions: &[f64], targets: &[f64]) -> f64 {
        if predictions.is_empty() {
            return 0.0;
        }

        let mse: f64 = predictions
            .iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| (pred - target).powi(2))
            .sum();

        mse / predictions.len() as f64
    }

    /// Compute pairwise ranking loss
    fn compute_ranking_loss(&self, predictions: &[f64], targets: &[f64]) -> f64 {
        if predictions.len() < 2 {
            return 0.0;
        }

        let mut total_loss = 0.0;
        let mut num_pairs = 0;

        // Pairwise ranking: if target_i > target_j, then pred_i should > pred_j
        for i in 0..predictions.len() {
            for j in (i + 1)..predictions.len() {
                if targets[i] != targets[j] {
                    let target_diff = targets[i] - targets[j];
                    let pred_diff = predictions[i] - predictions[j];

                    // Hinge loss: max(0, -sign(target_diff) * pred_diff + margin)
                    let margin = 0.1;
                    let sign = if target_diff > 0.0 { 1.0 } else { -1.0 };
                    let loss = (0.0_f64).max(-sign * pred_diff + margin);

                    total_loss += loss;
                    num_pairs += 1;
                }
            }
        }

        if num_pairs > 0 {
            total_loss / num_pairs as f64
        } else {
            0.0
        }
    }

    /// Backward pass (simplified gradient descent)
    fn backward_batch(
        &mut self,
        batch: &[TrainingSample],
        predictions: &[f64],
        targets: &[f64],
    ) -> Result<()> {
        // Simplified gradient computation (approximate)
        let mut weight_gradients: Array1<f64> = Array1::zeros(self.prediction_weights.len());

        for (sample, (&pred, &target)) in batch.iter().zip(predictions.iter().zip(targets.iter())) {
            let error = pred - target;

            // Gradient through prediction head (approximate)
            let gat_output = self.gat_layer.forward(&sample.problem.features, &[])?;

            for i in 0..self.prediction_weights.len() {
                weight_gradients[i] += error * gat_output[i];
            }
        }

        // Average gradients
        weight_gradients = weight_gradients / (batch.len() as f64);

        // Update weights (gradient descent)
        for i in 0..self.prediction_weights.len() {
            self.prediction_weights[i] -= self.config.learning_rate * weight_gradients[i];
        }

        Ok(())
    }

    /// Predict quality for a new problem
    pub fn predict(&self, problem: &ProblemEmbedding) -> Result<f64> {
        self.predict_quality(problem)
    }

    /// Get training history
    pub fn get_history(&self) -> &TrainingHistory {
        &self.history
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn create_test_samples(n: usize) -> Vec<TrainingSample> {
        use crate::applications::solver::ProblemType;

        (0..n)
            .map(|i| {
                let features = Array1::from_elem(128, (i as f64) / (n as f64));
                TrainingSample {
                    problem: ProblemEmbedding {
                        features,
                        problem_type: ProblemType::ContinuousOptimization,
                        dimension: 128,
                        metadata: std::collections::HashMap::new(),
                    },
                    quality: (i as f64) / (n as f64), // Quality correlates with features
                    solution: None,
                }
            })
            .collect()
    }

    #[test]
    fn test_trainer_creation() {
        let config = TrainingConfig::default();
        let trainer = GnnTrainer::new(config);

        assert_eq!(trainer.prediction_weights.len(), 128);
        assert_eq!(trainer.history.epochs_trained, 0);
    }

    #[test]
    fn test_training_basic() {
        let mut config = TrainingConfig::default();
        config.max_epochs = 5;
        config.batch_size = 8;

        let mut trainer = GnnTrainer::new(config);
        let samples = create_test_samples(50);

        let result = trainer.train(samples);
        assert!(result.is_ok());

        let history = result.unwrap();
        assert!(history.epochs_trained > 0);
        assert!(!history.train_losses.is_empty());
        assert!(!history.val_losses.is_empty());
    }

    #[test]
    fn test_prediction() {
        use crate::applications::solver::ProblemType;

        let config = TrainingConfig::default();
        let trainer = GnnTrainer::new(config);

        let features = Array1::from_elem(128, 0.5);
        let problem = ProblemEmbedding {
            features,
            problem_type: ProblemType::ContinuousOptimization,
            dimension: 128,
            metadata: std::collections::HashMap::new(),
        };

        let result = trainer.predict(&problem);
        assert!(result.is_ok());

        let quality = result.unwrap();
        assert!(quality.is_finite());
    }

    #[test]
    fn test_mse_loss_computation() {
        let config = TrainingConfig::default();
        let trainer = GnnTrainer::new(config);

        let predictions = vec![1.0, 2.0, 3.0];
        let targets = vec![1.1, 2.2, 2.9];

        let loss = trainer.compute_mse_loss(&predictions, &targets);
        assert!(loss > 0.0);
        assert!(loss < 0.1); // Should be small for close predictions
    }

    #[test]
    fn test_ranking_loss_computation() {
        let config = TrainingConfig::default();
        let trainer = GnnTrainer::new(config);

        // Perfect ranking
        let predictions1 = vec![1.0, 2.0, 3.0];
        let targets1 = vec![1.0, 2.0, 3.0];
        let loss1 = trainer.compute_ranking_loss(&predictions1, &targets1);

        // Reversed ranking
        let predictions2 = vec![3.0, 2.0, 1.0];
        let targets2 = vec![1.0, 2.0, 3.0];
        let loss2 = trainer.compute_ranking_loss(&predictions2, &targets2);

        // Perfect ranking should have lower loss
        assert!(loss1 < loss2);
    }

    #[test]
    fn test_early_stopping() {
        use crate::applications::solver::ProblemType;

        let mut config = TrainingConfig::default();
        config.max_epochs = 100;
        config.patience = 3;
        config.batch_size = 10;

        let mut trainer = GnnTrainer::new(config);

        // Create samples with constant quality (no learning possible)
        let samples: Vec<TrainingSample> = (0..50)
            .map(|i| TrainingSample {
                problem: ProblemEmbedding {
                    features: Array1::from_elem(128, (i as f64) / 50.0),
                    problem_type: ProblemType::ContinuousOptimization,
                    dimension: 128,
                    metadata: std::collections::HashMap::new(),
                },
                quality: 0.5, // Constant
                solution: None,
            })
            .collect();

        let result = trainer.train(samples);
        assert!(result.is_ok());

        let history = result.unwrap();
        // Should stop early due to no improvement
        assert!(history.epochs_trained < 100);
    }
}
