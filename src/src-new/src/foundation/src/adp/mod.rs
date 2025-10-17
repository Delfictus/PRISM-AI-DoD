//! Adaptive Decision Processor (ADP)
//!
//! Implements reinforcement learning and adaptive decision making
//! for PRCT parameter optimization. Based on CSF's C-Logic ADP module.

pub mod reinforcement;
pub mod decision_processor;

pub use reinforcement::{ReinforcementLearner, RlConfig, RlStats, State, Action};
pub use decision_processor::{AdaptiveDecisionProcessor, Decision, AdpStats};
