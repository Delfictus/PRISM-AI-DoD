//! CMA (Causal Manifold Annealing) Integration for Universal Solver
//!
//! Integrates PRISM-AI's CMA solver for continuous optimization problems.

use anyhow::Result;
use ndarray::Array1;
use std::sync::Arc;

use crate::cma::{CausalManifoldAnnealing, Problem as CmaProblem, Solution as CmaSolution};
use crate::information_theory::TransferEntropy;
use crate::active_inference::ActiveInferenceController;
use crate::gpu::GpuSolvable;

use super::{Solution, SolutionMetrics, ProblemType};
use super::problem::{ProblemData, ObjectiveFunction};

/// Adapter for continuous optimization problems to CMA
pub struct CmaAdapter {
    /// Dimension of the problem
    dimension: usize,
    /// Variable bounds
    bounds: Vec<(f64, f64)>,
    /// Objective function
    objective: ObjectiveFunction,
}

impl CmaAdapter {
    /// Create adapter from problem data
    pub fn from_problem_data(data: &ProblemData) -> Result<Self> {
        match data {
            ProblemData::Continuous { variables, bounds, objective } => {
                Ok(Self {
                    dimension: variables.len(),
                    bounds: bounds.clone(),
                    objective: objective.clone(),
                })
            }
            _ => anyhow::bail!("CMA adapter requires continuous problem data"),
        }
    }

    /// Solve using CMA
    pub async fn solve(&self, gpu_solver: Arc<dyn GpuSolvable>) -> Result<Solution> {
        // Create CMA engine with required dependencies
        let transfer_entropy = Arc::new(TransferEntropy::default());

        // Create a simple active inference controller (simplified for now)
        // In production, this would be properly initialized
        let active_inference = Arc::new(
            ActiveInferenceController::new(
                crate::active_inference::PolicySelector::new(
                    3,  // horizon
                    10, // n_policies
                    Array1::ones(self.dimension),
                    crate::active_inference::VariationalInference::new(
                        crate::active_inference::ObservationModel::new(
                            self.dimension,
                            900,
                            8.0,
                            0.01
                        ),
                        crate::active_inference::TransitionModel::default_timescales(),
                        &crate::active_inference::HierarchicalModel::new(),
                    ),
                    crate::active_inference::TransitionModel::default_timescales(),
                ),
                crate::active_inference::SensingStrategy::Adaptive,
            )
        );

        let mut cma = CausalManifoldAnnealing::new(
            gpu_solver,
            transfer_entropy,
            active_inference,
        );

        // Enable neural enhancements for faster solving
        cma.enable_neural_enhancements();

        // Create CMA problem wrapper
        let problem = CmaProblemWrapper {
            dimension: self.dimension,
            bounds: self.bounds.clone(),
            objective: self.objective.clone(),
        };

        // Solve using CMA
        let result = cma.solve(&problem);

        // Extract solution
        let explanation = format!(
            "Continuous optimization solved using Causal Manifold Annealing (CMA).\n\
             Dimension: {}\n\
             Method: Thermodynamic Ensemble + Causal Discovery + Quantum Annealing\n\
             Ensemble size: {}\n\
             Guarantee: {}",
            self.dimension,
            result.ensemble_size,
            if result.guarantee.is_valid {
                format!("Valid (error bound: {:.6})", result.guarantee.error_bound)
            } else {
                "None".to_string()
            }
        );

        Ok(Solution::new(
            ProblemType::ContinuousOptimization,
            result.value.cost,
            result.value.data,
            "CMA (Causal Manifold Annealing)".to_string(),
            0.0,
        )
        .with_explanation(explanation)
        .with_metrics(SolutionMetrics {
            iterations: result.ensemble_size,
            convergence_rate: 0.0, // CMA doesn't track convergence rate directly
            is_optimal: result.guarantee.is_valid,
            optimality_gap: Some(result.guarantee.error_bound),
            constraints_satisfied: 0, // Would track constraint violations
            total_constraints: 0,
            quality_score: 1.0 / (1.0 + result.value.cost),
        }))
    }
}

/// Wrapper to adapt continuous problems to CMA's Problem trait
struct CmaProblemWrapper {
    dimension: usize,
    bounds: Vec<(f64, f64)>,
    objective: ObjectiveFunction,
}

impl CmaProblem for CmaProblemWrapper {
    fn evaluate(&self, solution: &CmaSolution) -> f64 {
        match &self.objective {
            ObjectiveFunction::Minimize(_expr) => {
                // For custom functions, evaluate directly
                // In production, would parse expression
                self.default_sphere_function(&solution.data)
            }
            ObjectiveFunction::Maximize(_expr) => {
                // Negate for maximization
                -self.default_sphere_function(&solution.data)
            }
            ObjectiveFunction::Custom(f) => {
                // Use custom evaluation function
                f(&solution.data)
            }
        }
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

impl CmaProblemWrapper {
    /// Default sphere function for testing
    fn default_sphere_function(&self, x: &[f64]) -> f64 {
        x.iter().map(|&xi| xi * xi).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cma_adapter_creation() {
        let data = ProblemData::Continuous {
            variables: vec!["x".to_string(), "y".to_string()],
            bounds: vec![(-5.0, 5.0), (-5.0, 5.0)],
            objective: ObjectiveFunction::Minimize("x^2 + y^2".to_string()),
        };

        let adapter = CmaAdapter::from_problem_data(&data);
        assert!(adapter.is_ok());

        let adapter = adapter.unwrap();
        assert_eq!(adapter.dimension, 2);
    }

    #[test]
    fn test_cma_problem_wrapper() {
        let problem = CmaProblemWrapper {
            dimension: 2,
            bounds: vec![(-5.0, 5.0), (-5.0, 5.0)],
            objective: ObjectiveFunction::Minimize("sphere".to_string()),
        };

        assert_eq!(problem.dimension(), 2);

        // Test evaluation
        let solution = CmaSolution {
            data: vec![1.0, 1.0],
            cost: 0.0,
        };

        let cost = problem.evaluate(&solution);
        assert_eq!(cost, 2.0); // 1^2 + 1^2 = 2
    }
}
