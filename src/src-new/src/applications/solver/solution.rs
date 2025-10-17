//! Solution Types for Universal Solver

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::ProblemType;

/// Solution with explanation and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Solution {
    /// Problem type that was solved
    pub problem_type: ProblemType,

    /// Objective value achieved
    pub objective_value: f64,

    /// Solution vector (interpretation depends on problem type)
    pub solution_vector: Vec<f64>,

    /// Algorithm used
    pub algorithm_used: String,

    /// Computation time in milliseconds
    pub computation_time_ms: f64,

    /// Human-readable explanation
    pub explanation: String,

    /// Confidence score (0-1)
    pub confidence: f64,

    /// Additional metrics
    pub metrics: SolutionMetrics,

    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Solution metrics and statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolutionMetrics {
    /// Number of iterations
    pub iterations: usize,

    /// Convergence rate
    pub convergence_rate: f64,

    /// Whether solution is optimal (proven)
    pub is_optimal: bool,

    /// Optimality gap (if known)
    pub optimality_gap: Option<f64>,

    /// Number of constraints satisfied
    pub constraints_satisfied: usize,

    /// Total constraints
    pub total_constraints: usize,

    /// Solution quality (problem-specific)
    pub quality_score: f64,
}

impl Default for SolutionMetrics {
    fn default() -> Self {
        Self {
            iterations: 0,
            convergence_rate: 0.0,
            is_optimal: false,
            optimality_gap: None,
            constraints_satisfied: 0,
            total_constraints: 0,
            quality_score: 0.0,
        }
    }
}

impl Solution {
    /// Create a new solution
    pub fn new(
        problem_type: ProblemType,
        objective_value: f64,
        solution_vector: Vec<f64>,
        algorithm_used: String,
        computation_time_ms: f64,
    ) -> Self {
        Self {
            problem_type,
            objective_value,
            solution_vector,
            algorithm_used,
            computation_time_ms,
            explanation: String::new(),
            confidence: 1.0,
            metrics: SolutionMetrics::default(),
            metadata: HashMap::new(),
        }
    }

    /// Set explanation
    pub fn with_explanation(mut self, explanation: String) -> Self {
        self.explanation = explanation;
        self
    }

    /// Set confidence
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Set metrics
    pub fn with_metrics(mut self, metrics: SolutionMetrics) -> Self {
        self.metrics = metrics;
        self
    }

    /// Check if solution is feasible (all constraints satisfied)
    pub fn is_feasible(&self) -> bool {
        self.metrics.constraints_satisfied == self.metrics.total_constraints
    }

    /// Get solution summary
    pub fn summary(&self) -> String {
        format!(
            "{} Solution:\n\
             Objective: {:.6}\n\
             Algorithm: {}\n\
             Time: {:.2}ms\n\
             Confidence: {:.1}%\n\
             Feasible: {}",
            self.problem_type.description(),
            self.objective_value,
            self.algorithm_used,
            self.computation_time_ms,
            self.confidence * 100.0,
            if self.is_feasible() { "Yes" } else { "No" }
        )
    }
}

impl std::fmt::Display for Solution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.summary())?;
        if !self.explanation.is_empty() {
            write!(f, "\n\nExplanation:\n{}", self.explanation)?;
        }
        Ok(())
    }
}
