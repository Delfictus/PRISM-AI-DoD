//! Graph Neural Network (GNN) Module - Worker 4
//!
//! Transfer learning across problem domains using Graph Attention Networks.
//! Enables fast approximate solutions by learning from previous problem-solution patterns.
//!
//! # Architecture
//!
//! **3-Layer Design**:
//! 1. **Encoding Layer**: Problem → Embedding (via ProblemEmbedder)
//! 2. **Transfer Layer**: GAT with multi-head attention
//! 3. **Prediction Layer**: Embedding → Solution quality estimate
//!
//! # Usage
//!
//! ```ignore
//! use prism_ai::applications::solver::gnn::{GnnTrainer, TrainingConfig, TrainingSample};
//!
//! // Create trainer
//! let config = TrainingConfig::default();
//! let mut trainer = GnnTrainer::new(config);
//!
//! // Train on historical data
//! let samples = vec![/* problem-solution pairs */];
//! trainer.train(samples)?;
//!
//! // Predict quality for new problem
//! let quality = trainer.predict(&problem_embedding)?;
//! ```

pub mod gat;
pub mod training;

pub use gat::{AttentionHead, GraphAttentionLayer, GatConfig, NUM_HEADS, HEAD_DIM};
pub use training::{GnnTrainer, TrainingConfig, TrainingSample, TrainingHistory};
