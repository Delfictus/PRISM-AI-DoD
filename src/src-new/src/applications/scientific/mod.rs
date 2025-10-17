//! Scientific Discovery Applications Module - Worker 7
//!
//! Tools for scientific research and experimental design:
//! - Experiment optimization using Active Inference
//! - Parameter space exploration
//! - Hypothesis testing with Bayesian inference
//! - Data analysis and pattern discovery

pub mod experiment_design;
pub mod parameter_optimization;
pub mod hypothesis_testing;

pub use experiment_design::{ExperimentDesigner, ExperimentConfig, ExperimentResult};
pub use parameter_optimization::{ParameterOptimizer, OptimizationConfig, ParameterSpace};
pub use hypothesis_testing::{HypothesisTester, Hypothesis, BayesianEvidence};

/// Scientific discovery module configuration
#[derive(Debug, Clone)]
pub struct ScientificConfig {
    /// Use GPU acceleration for computations
    pub use_gpu: bool,
    /// Confidence level for statistical tests
    pub confidence_level: f64,
    /// Maximum number of experiments
    pub max_experiments: usize,
}

impl Default for ScientificConfig {
    fn default() -> Self {
        Self {
            use_gpu: true,
            confidence_level: 0.95,
            max_experiments: 1000,
        }
    }
}

/// Unified scientific discovery system
pub struct ScientificDiscovery {
    config: ScientificConfig,
    experiment_designer: ExperimentDesigner,
    parameter_optimizer: ParameterOptimizer,
    hypothesis_tester: HypothesisTester,
}

impl ScientificDiscovery {
    /// Create new scientific discovery system
    pub fn new(config: ScientificConfig) -> anyhow::Result<Self> {
        let experiment_designer = ExperimentDesigner::new()?;
        let parameter_optimizer = ParameterOptimizer::new()?;
        let hypothesis_tester = HypothesisTester::new(config.confidence_level);

        Ok(Self {
            config,
            experiment_designer,
            parameter_optimizer,
            hypothesis_tester,
        })
    }

    /// Design optimal next experiment using Active Inference
    pub fn design_next_experiment(
        &mut self,
        current_knowledge: &ExperimentResult,
        constraints: &ExperimentConfig,
    ) -> anyhow::Result<ExperimentConfig> {
        self.experiment_designer.design_next(current_knowledge, constraints)
    }

    /// Optimize parameters for a scientific model
    pub fn optimize_parameters(
        &mut self,
        parameter_space: &ParameterSpace,
        observations: &[f64],
    ) -> anyhow::Result<Vec<f64>> {
        self.parameter_optimizer.optimize(parameter_space, observations)
    }

    /// Test hypothesis against experimental data
    pub fn test_hypothesis(
        &self,
        hypothesis: &Hypothesis,
        data: &[f64],
    ) -> anyhow::Result<BayesianEvidence> {
        self.hypothesis_tester.test(hypothesis, data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scientific_config_default() {
        let config = ScientificConfig::default();
        assert!(config.use_gpu);
        assert_eq!(config.confidence_level, 0.95);
        assert_eq!(config.max_experiments, 1000);
    }

    #[test]
    fn test_scientific_discovery_creation() {
        let config = ScientificConfig::default();
        let system = ScientificDiscovery::new(config);
        assert!(system.is_ok());
    }
}
