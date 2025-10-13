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
//! use prism_ai::applications::solver::gnn::{GnnTrainer, GnnPredictor, TrainingConfig};
//!
//! // Train GNN
//! let config = TrainingConfig::default();
//! let mut trainer = GnnTrainer::new(config);
//! trainer.train(samples)?;
//!
//! // Create predictor with confidence routing
//! let pred_config = PredictorConfig::default();
//! let predictor = GnnPredictor::new(trainer, pred_config);
//!
//! // Predict with confidence
//! let result = predictor.predict(&problem)?;
//! if result.use_prediction {
//!     // Use GNN prediction
//! } else {
//!     // Fallback to exact solver
//! }
//! ```

pub mod gat;
pub mod training;
pub mod predictor;

pub use gat::{AttentionHead, GraphAttentionLayer, GatConfig, NUM_HEADS, HEAD_DIM};
pub use training::{GnnTrainer, TrainingConfig, TrainingSample, TrainingHistory};
pub use predictor::{GnnPredictor, PredictorConfig, PredictionResult, HybridStats, DEFAULT_CONFIDENCE_THRESHOLD};
