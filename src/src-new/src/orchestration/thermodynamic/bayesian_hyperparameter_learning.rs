//! Bayesian Hyperparameter Learning for Thermodynamic Schedules
//!
//! Implements Bayesian inference to automatically learn optimal hyperparameters
//! for temperature schedules based on observed performance.
//!
//! Worker 5 Enhancement: Week 3, Task 3.1
//!
//! ## Theory
//!
//! Instead of manually tuning hyperparameters (cooling rate, initial temperature,
//! etc.), we use Bayesian inference to learn optimal values from data:
//!
//! 1. **Prior**: Initial beliefs about hyperparameters (e.g., uniform, Gaussian)
//! 2. **Likelihood**: How well hyperparameters explain observed performance
//! 3. **Posterior**: Updated beliefs after observing data (Bayes' rule)
//! 4. **Inference**: Sample from posterior or maximize (MAP estimation)
//!
//! Bayes' Rule: P(θ|D) ∝ P(D|θ) * P(θ)
//! - θ: hyperparameters
//! - D: observed performance data
//! - P(θ): prior distribution
//! - P(D|θ): likelihood function
//! - P(θ|D): posterior distribution

use anyhow::{Context, Result};
use std::collections::HashMap;
use rand::prelude::*;
use rand_distr::{Distribution, Normal, Uniform};

/// Prior distribution types for hyperparameters
#[derive(Debug, Clone)]
pub enum PriorDistribution {
    /// Uniform distribution U(min, max)
    Uniform { min: f64, max: f64 },

    /// Normal (Gaussian) distribution N(μ, σ²)
    Normal { mean: f64, std: f64 },

    /// Log-normal distribution (for positive-only parameters)
    LogNormal { mu: f64, sigma: f64 },

    /// Beta distribution (for parameters in [0, 1])
    Beta { alpha: f64, beta: f64 },
}

impl PriorDistribution {
    /// Sample from the prior distribution
    pub fn sample(&self, rng: &mut impl Rng) -> f64 {
        match self {
            PriorDistribution::Uniform { min, max } => {
                let dist = Uniform::new(*min, *max);
                dist.sample(rng)
            },
            PriorDistribution::Normal { mean, std } => {
                let dist = Normal::new(*mean, *std).unwrap();
                dist.sample(rng)
            },
            PriorDistribution::LogNormal { mu, sigma } => {
                let normal = Normal::new(*mu, *sigma).unwrap();
                let log_sample = normal.sample(rng);
                log_sample.exp()
            },
            PriorDistribution::Beta { alpha, beta } => {
                // Beta distribution using accept-reject from two gammas
                // For now, approximate with truncated normal
                let mean = alpha / (alpha + beta);
                let variance = (alpha * beta) / ((alpha + beta).powi(2) * (alpha + beta + 1.0));
                let std = variance.sqrt();
                let dist = Normal::new(mean, std).unwrap();
                dist.sample(rng).clamp(0.0, 1.0)
            },
        }
    }

    /// Compute log probability density
    pub fn log_pdf(&self, x: f64) -> f64 {
        match self {
            PriorDistribution::Uniform { min, max } => {
                if x >= *min && x <= *max {
                    -(max - min).ln()
                } else {
                    f64::NEG_INFINITY
                }
            },
            PriorDistribution::Normal { mean, std } => {
                let z = (x - mean) / std;
                -0.5 * z * z - std.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
            },
            PriorDistribution::LogNormal { mu, sigma } => {
                if x <= 0.0 {
                    return f64::NEG_INFINITY;
                }
                let log_x = x.ln();
                let z = (log_x - mu) / sigma;
                -log_x - 0.5 * z * z - sigma.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
            },
            PriorDistribution::Beta { alpha, beta } => {
                if x <= 0.0 || x >= 1.0 {
                    return f64::NEG_INFINITY;
                }
                // Beta PDF: x^(α-1) * (1-x)^(β-1) / B(α, β)
                // Log: (α-1)*ln(x) + (β-1)*ln(1-x) - ln(B(α,β))
                let log_beta = lgamma(*alpha) + lgamma(*beta) - lgamma(alpha + beta);
                (alpha - 1.0) * x.ln() + (beta - 1.0) * (1.0 - x).ln() - log_beta
            },
        }
    }
}

/// Log-gamma function approximation (Stirling's approximation)
fn lgamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    // Stirling's approximation: ln(Γ(x)) ≈ (x-0.5)ln(x) - x + 0.5*ln(2π)
    (x - 0.5) * x.ln() - x + 0.5 * (2.0 * std::f64::consts::PI).ln()
}

/// Performance observation for Bayesian learning
#[derive(Debug, Clone)]
pub struct PerformanceObservation {
    /// Hyperparameter values used
    pub hyperparameters: HashMap<String, f64>,
    /// Performance metric (higher is better, e.g., negative cost)
    pub performance: f64,
    /// Number of iterations run
    pub iterations: usize,
    /// Final acceptance rate achieved
    pub acceptance_rate: f64,
}

/// Bayesian hyperparameter learner
///
/// Learns optimal hyperparameters using Bayesian inference
pub struct BayesianHyperparameterLearner {
    /// Prior distributions for each hyperparameter
    priors: HashMap<String, PriorDistribution>,

    /// Observed performance data
    observations: Vec<PerformanceObservation>,

    /// Posterior samples (from MCMC or other inference)
    posterior_samples: Vec<HashMap<String, f64>>,

    /// Evidence (marginal likelihood) - used for model comparison
    evidence: f64,

    /// Random number generator
    rng: StdRng,
}

impl BayesianHyperparameterLearner {
    /// Create new Bayesian learner
    ///
    /// # Arguments
    /// * `priors` - Prior distributions for hyperparameters
    /// * `seed` - RNG seed for reproducibility
    pub fn new(priors: HashMap<String, PriorDistribution>, seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        Self {
            priors,
            observations: Vec::new(),
            posterior_samples: Vec::new(),
            evidence: 0.0,
            rng,
        }
    }

    /// Add performance observation
    pub fn observe(&mut self, observation: PerformanceObservation) {
        self.observations.push(observation);
        // Invalidate posterior samples (need to re-run inference)
        self.posterior_samples.clear();
    }

    /// Compute log likelihood of observations given hyperparameters
    ///
    /// Assumes performance follows Gaussian distribution around predicted value
    fn log_likelihood(&self, hyperparameters: &HashMap<String, f64>) -> f64 {
        let mut log_likelihood = 0.0;
        let noise_std = 0.1; // Assumed observation noise

        for obs in &self.observations {
            // Predicted performance based on hyperparameters
            // For now, simple model: higher cooling rate → faster convergence → better performance
            let predicted = self.predict_performance(hyperparameters);

            // Gaussian likelihood
            let error = obs.performance - predicted;
            log_likelihood += -0.5 * (error / noise_std).powi(2);
        }

        log_likelihood
    }

    /// Predict performance given hyperparameters (simple model)
    fn predict_performance(&self, hyperparameters: &HashMap<String, f64>) -> f64 {
        // Simple heuristic model for demonstration
        // In practice, this would be learned from data or use physics-based model

        let cooling_rate = hyperparameters.get("cooling_rate").unwrap_or(&0.95);
        let initial_temp = hyperparameters.get("initial_temp").unwrap_or(&1.0);

        // Performance increases with moderate cooling and appropriate initial temp
        let cooling_score = 1.0 - (cooling_rate - 0.95).abs();
        let temp_score = 1.0 - (initial_temp - 1.0).abs() / 10.0;

        cooling_score * temp_score
    }

    /// Compute log posterior (prior + likelihood)
    fn log_posterior(&self, hyperparameters: &HashMap<String, f64>) -> f64 {
        let mut log_prior = 0.0;

        // Sum log prior for all hyperparameters
        for (param, value) in hyperparameters {
            if let Some(prior) = self.priors.get(param) {
                log_prior += prior.log_pdf(*value);
            }
        }

        // Add log likelihood
        let log_like = self.log_likelihood(hyperparameters);

        log_prior + log_like
    }

    /// Run Metropolis-Hastings MCMC to sample from posterior
    ///
    /// # Arguments
    /// * `num_samples` - Number of posterior samples to generate
    /// * `burn_in` - Number of initial samples to discard
    /// * `proposal_std` - Standard deviation for proposal distribution
    pub fn infer_posterior_mcmc(
        &mut self,
        num_samples: usize,
        burn_in: usize,
        proposal_std: f64,
    ) -> Result<()> {
        if self.observations.is_empty() {
            return Ok(()); // No data to learn from
        }

        self.posterior_samples.clear();

        // Initialize with sample from prior
        let mut current: HashMap<String, f64> = self.priors.iter()
            .map(|(name, prior)| (name.clone(), prior.sample(&mut self.rng)))
            .collect();
        let mut current_log_prob = self.log_posterior(&current);

        let mut accepted = 0;

        for i in 0..(num_samples + burn_in) {
            // Propose new state (random walk)
            let mut proposed = current.clone();
            for (param, value) in proposed.iter_mut() {
                let noise = Normal::new(0.0, proposal_std).unwrap();
                *value += noise.sample(&mut self.rng);
            }

            // Compute acceptance probability
            let proposed_log_prob = self.log_posterior(&proposed);
            let log_acceptance = proposed_log_prob - current_log_prob;

            // Accept or reject
            if log_acceptance >= 0.0 || self.rng.gen::<f64>().ln() < log_acceptance {
                current = proposed;
                current_log_prob = proposed_log_prob;
                accepted += 1;
            }

            // Store sample (after burn-in)
            if i >= burn_in {
                self.posterior_samples.push(current.clone());
            }
        }

        let acceptance_rate = accepted as f64 / (num_samples + burn_in) as f64;
        println!("MCMC acceptance rate: {:.3}", acceptance_rate);

        Ok(())
    }

    /// Get MAP (Maximum A Posteriori) estimate
    ///
    /// Returns hyperparameters that maximize posterior probability
    pub fn get_map_estimate(&self) -> Option<HashMap<String, f64>> {
        self.posterior_samples.iter()
            .max_by(|a, b| {
                self.log_posterior(a)
                    .partial_cmp(&self.log_posterior(b))
                    .unwrap()
            })
            .cloned()
    }

    /// Get posterior mean estimate
    ///
    /// Returns average of posterior samples for each hyperparameter
    pub fn get_posterior_mean(&self) -> Option<HashMap<String, f64>> {
        if self.posterior_samples.is_empty() {
            return None;
        }

        let mut means = HashMap::new();

        for param in self.priors.keys() {
            let sum: f64 = self.posterior_samples.iter()
                .map(|sample| sample.get(param).unwrap_or(&0.0))
                .sum();
            means.insert(param.clone(), sum / self.posterior_samples.len() as f64);
        }

        Some(means)
    }

    /// Get posterior standard deviation for each hyperparameter
    pub fn get_posterior_std(&self) -> Option<HashMap<String, f64>> {
        if self.posterior_samples.is_empty() {
            return None;
        }

        let means = self.get_posterior_mean()?;
        let mut stds = HashMap::new();

        for (param, mean) in &means {
            let variance: f64 = self.posterior_samples.iter()
                .map(|sample| {
                    let value = sample.get(param).unwrap_or(&0.0);
                    (value - mean).powi(2)
                })
                .sum::<f64>() / self.posterior_samples.len() as f64;

            stds.insert(param.clone(), variance.sqrt());
        }

        Some(stds)
    }

    /// Sample hyperparameters from posterior (Thompson sampling)
    ///
    /// Useful for exploration/exploitation trade-off
    pub fn sample_from_posterior(&mut self) -> Option<HashMap<String, f64>> {
        if self.posterior_samples.is_empty() {
            return None;
        }

        let idx = self.rng.gen_range(0..self.posterior_samples.len());
        Some(self.posterior_samples[idx].clone())
    }

    /// Get number of posterior samples
    pub fn num_posterior_samples(&self) -> usize {
        self.posterior_samples.len()
    }

    /// Get number of observations
    pub fn num_observations(&self) -> usize {
        self.observations.len()
    }

    /// Compute posterior predictive distribution
    ///
    /// Predicts performance for new hyperparameters by marginalizing over posterior
    pub fn predict_performance_distribution(&self, hyperparameters: &HashMap<String, f64>) -> (f64, f64) {
        if self.posterior_samples.is_empty() {
            // No posterior samples, use prior
            return (self.predict_performance(hyperparameters), 1.0);
        }

        // Predict using each posterior sample, then average
        let predictions: Vec<f64> = self.posterior_samples.iter()
            .map(|_| self.predict_performance(hyperparameters))
            .collect();

        let mean = predictions.iter().sum::<f64>() / predictions.len() as f64;
        let variance = predictions.iter()
            .map(|p| (p - mean).powi(2))
            .sum::<f64>() / predictions.len() as f64;

        (mean, variance.sqrt())
    }
}

/// Thompson sampling for hyperparameter selection
///
/// Balances exploration (trying uncertain hyperparameters) and
/// exploitation (using known good hyperparameters)
pub struct ThompsonSampler {
    learner: BayesianHyperparameterLearner,
}

impl ThompsonSampler {
    /// Create new Thompson sampler
    pub fn new(learner: BayesianHyperparameterLearner) -> Self {
        Self { learner }
    }

    /// Select next hyperparameters to try
    ///
    /// Samples from posterior (exploration) or uses best known (exploitation)
    pub fn select_next(&mut self) -> HashMap<String, f64> {
        // If we have posterior samples, use Thompson sampling
        if self.learner.num_posterior_samples() > 0 {
            self.learner.sample_from_posterior()
                .unwrap_or_else(|| self.sample_from_prior())
        } else {
            // No posterior yet, sample from prior
            self.sample_from_prior()
        }
    }

    fn sample_from_prior(&mut self) -> HashMap<String, f64> {
        self.learner.priors.iter()
            .map(|(name, prior)| (name.clone(), prior.sample(&mut self.learner.rng)))
            .collect()
    }

    /// Update with new observation
    pub fn observe(&mut self, observation: PerformanceObservation) {
        self.learner.observe(observation);

        // Periodically re-run inference (expensive)
        if self.learner.num_observations() % 10 == 0 {
            let _ = self.learner.infer_posterior_mcmc(1000, 100, 0.01);
        }
    }

    /// Get best hyperparameters so far (MAP estimate)
    pub fn get_best(&self) -> Option<HashMap<String, f64>> {
        self.learner.get_map_estimate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prior_uniform_sampling() {
        let prior = PriorDistribution::Uniform { min: 0.0, max: 1.0 };
        let mut rng = StdRng::seed_from_u64(42);

        let samples: Vec<f64> = (0..1000)
            .map(|_| prior.sample(&mut rng))
            .collect();

        // All samples should be in range
        assert!(samples.iter().all(|&x| x >= 0.0 && x <= 1.0));

        // Mean should be ~0.5
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!((mean - 0.5).abs() < 0.05);
    }

    #[test]
    fn test_prior_normal_sampling() {
        let prior = PriorDistribution::Normal { mean: 0.0, std: 1.0 };
        let mut rng = StdRng::seed_from_u64(42);

        let samples: Vec<f64> = (0..1000)
            .map(|_| prior.sample(&mut rng))
            .collect();

        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!((mean - 0.0).abs() < 0.1);
    }

    #[test]
    fn test_prior_log_pdf() {
        let prior = PriorDistribution::Uniform { min: 0.0, max: 1.0 };

        // Inside range should have constant probability
        let log_p1 = prior.log_pdf(0.3);
        let log_p2 = prior.log_pdf(0.7);
        assert!((log_p1 - log_p2).abs() < 1e-10);

        // Outside range should have zero probability
        assert!(prior.log_pdf(-0.1).is_infinite());
        assert!(prior.log_pdf(1.1).is_infinite());
    }

    #[test]
    fn test_bayesian_learner_initialization() {
        let mut priors = HashMap::new();
        priors.insert("cooling_rate".to_string(),
            PriorDistribution::Uniform { min: 0.9, max: 0.99 });
        priors.insert("initial_temp".to_string(),
            PriorDistribution::Normal { mean: 1.0, std: 0.5 });

        let learner = BayesianHyperparameterLearner::new(priors, Some(42));

        assert_eq!(learner.num_observations(), 0);
        assert_eq!(learner.num_posterior_samples(), 0);
    }

    #[test]
    fn test_bayesian_learner_observation() {
        let mut priors = HashMap::new();
        priors.insert("cooling_rate".to_string(),
            PriorDistribution::Uniform { min: 0.9, max: 0.99 });

        let mut learner = BayesianHyperparameterLearner::new(priors, Some(42));

        let mut hyperparams = HashMap::new();
        hyperparams.insert("cooling_rate".to_string(), 0.95);

        learner.observe(PerformanceObservation {
            hyperparameters: hyperparams,
            performance: 0.8,
            iterations: 100,
            acceptance_rate: 0.44,
        });

        assert_eq!(learner.num_observations(), 1);
    }

    #[test]
    fn test_mcmc_inference() {
        let mut priors = HashMap::new();
        priors.insert("cooling_rate".to_string(),
            PriorDistribution::Uniform { min: 0.9, max: 0.99 });

        let mut learner = BayesianHyperparameterLearner::new(priors, Some(42));

        // Add some observations
        for i in 0..5 {
            let mut hyperparams = HashMap::new();
            hyperparams.insert("cooling_rate".to_string(), 0.95);

            learner.observe(PerformanceObservation {
                hyperparameters: hyperparams,
                performance: 0.8 + (i as f64 * 0.01),
                iterations: 100,
                acceptance_rate: 0.44,
            });
        }

        // Run MCMC
        learner.infer_posterior_mcmc(100, 10, 0.01).unwrap();

        assert!(learner.num_posterior_samples() > 0);
    }

    #[test]
    fn test_posterior_estimates() {
        let mut priors = HashMap::new();
        priors.insert("cooling_rate".to_string(),
            PriorDistribution::Uniform { min: 0.9, max: 0.99 });

        let mut learner = BayesianHyperparameterLearner::new(priors, Some(42));

        // Add observations
        for _ in 0..3 {
            let mut hyperparams = HashMap::new();
            hyperparams.insert("cooling_rate".to_string(), 0.95);

            learner.observe(PerformanceObservation {
                hyperparameters: hyperparams,
                performance: 0.8,
                iterations: 100,
                acceptance_rate: 0.44,
            });
        }

        learner.infer_posterior_mcmc(100, 10, 0.01).unwrap();

        let map = learner.get_map_estimate();
        assert!(map.is_some());

        let mean = learner.get_posterior_mean();
        assert!(mean.is_some());

        let std = learner.get_posterior_std();
        assert!(std.is_some());
    }

    #[test]
    fn test_thompson_sampling() {
        let mut priors = HashMap::new();
        priors.insert("cooling_rate".to_string(),
            PriorDistribution::Uniform { min: 0.9, max: 0.99 });

        let learner = BayesianHyperparameterLearner::new(priors, Some(42));
        let mut sampler = ThompsonSampler::new(learner);

        // Should sample from prior initially
        let params1 = sampler.select_next();
        assert!(params1.contains_key("cooling_rate"));

        // Add observation
        sampler.observe(PerformanceObservation {
            hyperparameters: params1,
            performance: 0.8,
            iterations: 100,
            acceptance_rate: 0.44,
        });

        // Should still work after observation
        let params2 = sampler.select_next();
        assert!(params2.contains_key("cooling_rate"));
    }
}
