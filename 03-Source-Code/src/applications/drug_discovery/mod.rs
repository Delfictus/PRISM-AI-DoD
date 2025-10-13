//! Drug Discovery with Active Inference
//!
//! Uses Worker 1's Active Inference to optimize drug molecules:
//! - Minimize free energy = maximize binding affinity + optimize properties
//! - Explore chemical space efficiently
//! - Predict drug-target interactions
//!
//! GPU-accelerated via Worker 2's kernels

use anyhow::Result;
use ndarray::{Array1, Array2};
use crate::active_inference::GenerativeModel;

pub mod molecular;
pub mod optimization;
pub mod prediction;

/// Drug discovery configuration
#[derive(Debug, Clone)]
pub struct DrugDiscoveryConfig {
    /// Number of optimization iterations
    pub max_iterations: usize,
    /// Learning rate for molecular optimization
    pub learning_rate: f64,
    /// Target binding affinity (lower is better, kcal/mol)
    pub target_affinity: f64,
    /// Use GPU acceleration
    pub use_gpu: bool,
}

impl Default for DrugDiscoveryConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            learning_rate: 0.01,
            target_affinity: -10.0, // -10 kcal/mol is good binding
            use_gpu: true,
        }
    }
}

/// Drug discovery controller using Active Inference
pub struct DrugDiscoveryController {
    config: DrugDiscoveryConfig,
    generative_model: GenerativeModel,
}

impl DrugDiscoveryController {
    /// Create new drug discovery controller
    pub fn new(config: DrugDiscoveryConfig) -> Result<Self> {
        let generative_model = GenerativeModel::new();

        Ok(Self {
            config,
            generative_model,
        })
    }

    /// Optimize molecule to maximize binding affinity
    ///
    /// Uses Active Inference to minimize free energy:
    /// F = E[ln Q(molecule|target)] - ln P(molecule,target)
    ///
    /// Where:
    /// - Q is the approximate posterior (our belief about good molecules)
    /// - P is the true distribution (actual binding affinity)
    pub fn optimize_molecule(
        &mut self,
        initial_molecule: &molecular::Molecule,
        target_protein: &molecular::Protein,
    ) -> Result<optimization::OptimizationResult> {
        let mut optimizer = optimization::MolecularOptimizer::new(
            self.config.clone(),
            &mut self.generative_model,
        );

        optimizer.optimize(initial_molecule, target_protein)
    }

    /// Predict drug-target binding affinity
    pub fn predict_binding(
        &self,
        molecule: &molecular::Molecule,
        target: &molecular::Protein,
    ) -> Result<prediction::BindingPrediction> {
        let predictor = prediction::BindingPredictor::new();
        predictor.predict(molecule, target)
    }

    /// Screen library of molecules against target
    pub fn screen_library(
        &self,
        molecules: &[molecular::Molecule],
        target: &molecular::Protein,
    ) -> Result<Vec<prediction::BindingPrediction>> {
        let predictor = prediction::BindingPredictor::new();

        molecules
            .iter()
            .map(|mol| predictor.predict(mol, target))
            .collect()
    }
}

/// Platform version for drug discovery module
pub const VERSION: &str = "0.1.0";
pub const MODULE_NAME: &str = "Drug Discovery (Active Inference)";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_controller_creation() {
        let config = DrugDiscoveryConfig::default();
        let controller = DrugDiscoveryController::new(config);
        assert!(controller.is_ok());
    }

    #[test]
    fn test_config_defaults() {
        let config = DrugDiscoveryConfig::default();
        assert_eq!(config.max_iterations, 1000);
        assert_eq!(config.target_affinity, -10.0);
        assert!(config.use_gpu);
    }
}
