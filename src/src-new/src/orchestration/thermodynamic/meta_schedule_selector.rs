//! Meta-Learning for Temperature Schedule Selection
//!
//! Implements a meta-learning system that learns to recommend optimal
//! temperature schedules based on problem characteristics.
//!
//! Worker 5 Enhancement: Week 3, Task 3.2
//!
//! ## Theory
//!
//! Different optimization problems respond better to different temperature
//! schedules. Instead of manually choosing schedules, meta-learning:
//!
//! 1. **Extracts problem features**: Size, structure, energy landscape
//! 2. **Tracks performance**: Which schedules work for which problems
//! 3. **Learns mapping**: Problem features → Best schedule
//! 4. **Recommends**: Predicts best schedule for new problems
//!
//! This is "learning to optimize" - using historical data to guide
//! future optimization attempts.

use anyhow::{Context, Result};
use std::collections::HashMap;

/// Problem features extracted for meta-learning
///
/// These features characterize the optimization problem to help
/// predict which schedule will work best
#[derive(Debug, Clone)]
pub struct ProblemFeatures {
    /// Problem dimensionality (number of variables)
    pub dimensionality: usize,

    /// Estimated problem size (e.g., number of LLM choices)
    pub problem_size: usize,

    /// Energy landscape ruggedness (estimated from initial samples)
    /// Range: [0, 1] where 0 = smooth, 1 = very rugged
    pub ruggedness: f64,

    /// Multi-modality indicator (estimated number of local optima)
    pub estimated_local_optima: usize,

    /// Budget constraint (available iterations/time)
    pub budget: f64,

    /// Quality requirement (how good solution must be)
    /// Range: [0, 1] where 0 = any solution, 1 = near-optimal required
    pub quality_requirement: f64,

    /// Problem domain/type (optional identifier)
    pub domain: Option<String>,
}

impl ProblemFeatures {
    /// Create features from problem parameters
    pub fn new(
        dimensionality: usize,
        problem_size: usize,
        budget: f64,
        quality_requirement: f64,
    ) -> Self {
        Self {
            dimensionality,
            problem_size,
            ruggedness: 0.5, // Default: medium ruggedness
            estimated_local_optima: 10, // Default estimate
            budget,
            quality_requirement,
            domain: None,
        }
    }

    /// Estimate ruggedness from initial energy samples
    ///
    /// Ruggedness = standard deviation of energy changes / mean energy
    pub fn estimate_ruggedness(&mut self, energy_samples: &[f64]) {
        if energy_samples.len() < 2 {
            return;
        }

        // Compute energy differences
        let diffs: Vec<f64> = energy_samples.windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .collect();

        let mean_diff = diffs.iter().sum::<f64>() / diffs.len() as f64;
        let mean_energy = energy_samples.iter().sum::<f64>() / energy_samples.len() as f64;

        if mean_energy.abs() > 1e-10 {
            self.ruggedness = (mean_diff / mean_energy.abs()).min(1.0);
        }
    }

    /// Convert to feature vector for machine learning
    pub fn to_vector(&self) -> Vec<f64> {
        vec![
            self.dimensionality as f64,
            self.problem_size as f64,
            self.ruggedness,
            self.estimated_local_optima as f64,
            self.budget,
            self.quality_requirement,
        ]
    }

    /// Compute similarity to another problem (cosine similarity)
    pub fn similarity(&self, other: &ProblemFeatures) -> f64 {
        let v1 = self.to_vector();
        let v2 = other.to_vector();

        let dot: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f64 = v1.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm2: f64 = v2.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm1 > 0.0 && norm2 > 0.0 {
            dot / (norm1 * norm2)
        } else {
            0.0
        }
    }
}

/// Schedule type identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScheduleType {
    Simple,
    SimulatedAnnealing,
    ParallelTempering,
    HamiltonianMC,
    BayesianOptimization,
    MultiObjective,
}

impl ScheduleType {
    /// Get all available schedule types
    pub fn all() -> Vec<ScheduleType> {
        vec![
            ScheduleType::Simple,
            ScheduleType::SimulatedAnnealing,
            ScheduleType::ParallelTempering,
            ScheduleType::HamiltonianMC,
            ScheduleType::BayesianOptimization,
            ScheduleType::MultiObjective,
        ]
    }

    /// Get name of schedule
    pub fn name(&self) -> &'static str {
        match self {
            ScheduleType::Simple => "Simple",
            ScheduleType::SimulatedAnnealing => "SimulatedAnnealing",
            ScheduleType::ParallelTempering => "ParallelTempering",
            ScheduleType::HamiltonianMC => "HamiltonianMC",
            ScheduleType::BayesianOptimization => "BayesianOptimization",
            ScheduleType::MultiObjective => "MultiObjective",
        }
    }
}

/// Performance record for a schedule on a problem
#[derive(Debug, Clone)]
pub struct SchedulePerformanceRecord {
    /// Problem features
    pub features: ProblemFeatures,

    /// Schedule used
    pub schedule: ScheduleType,

    /// Final solution quality (higher is better)
    pub quality: f64,

    /// Iterations to convergence
    pub iterations: usize,

    /// Wall-clock time (seconds)
    pub time_seconds: f64,

    /// Final acceptance rate achieved
    pub acceptance_rate: f64,

    /// Whether it converged successfully
    pub converged: bool,
}

impl SchedulePerformanceRecord {
    /// Compute performance score (normalized)
    ///
    /// Combines quality, efficiency, and convergence
    pub fn performance_score(&self) -> f64 {
        let quality_score = self.quality;
        let efficiency_score = 1.0 / (1.0 + self.time_seconds / 100.0);
        let convergence_score = if self.converged { 1.0 } else { 0.5 };

        // Weighted combination
        0.5 * quality_score + 0.3 * efficiency_score + 0.2 * convergence_score
    }
}

/// Simple k-NN based recommendation model
///
/// Recommends schedules based on similarity to past problems
#[derive(Debug, Clone)]
pub struct KNNRecommender {
    /// Historical performance records
    history: Vec<SchedulePerformanceRecord>,

    /// Number of neighbors to consider
    k: usize,
}

impl KNNRecommender {
    /// Create new k-NN recommender
    pub fn new(k: usize) -> Self {
        Self {
            history: Vec::new(),
            k,
        }
    }

    /// Add performance record to history
    pub fn add_record(&mut self, record: SchedulePerformanceRecord) {
        self.history.push(record);
    }

    /// Recommend schedule for new problem
    ///
    /// Returns (schedule, confidence) where confidence ∈ [0, 1]
    pub fn recommend(&self, features: &ProblemFeatures) -> Option<(ScheduleType, f64)> {
        if self.history.is_empty() {
            return None;
        }

        // Find k nearest neighbors
        let mut similarities: Vec<(usize, f64)> = self.history.iter()
            .enumerate()
            .map(|(i, record)| {
                let sim = features.similarity(&record.features);
                (i, sim)
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let neighbors: Vec<(usize, f64)> = similarities.into_iter()
            .take(self.k.min(self.history.len()))
            .collect();

        // Vote on best schedule (weighted by similarity and performance)
        let mut schedule_scores: HashMap<ScheduleType, f64> = HashMap::new();

        for (idx, similarity) in neighbors {
            let record = &self.history[idx];
            let weight = similarity * record.performance_score();

            *schedule_scores.entry(record.schedule).or_insert(0.0) += weight;
        }

        // Return schedule with highest score
        schedule_scores.into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(schedule, score)| {
                // Confidence = normalized score
                let max_possible_score = self.k as f64;
                let confidence = (score / max_possible_score).min(1.0);
                (schedule, confidence)
            })
    }

    /// Get performance statistics for each schedule
    pub fn schedule_statistics(&self) -> HashMap<ScheduleType, ScheduleStats> {
        let mut stats: HashMap<ScheduleType, Vec<f64>> = HashMap::new();

        for record in &self.history {
            stats.entry(record.schedule)
                .or_insert_with(Vec::new)
                .push(record.performance_score());
        }

        stats.into_iter()
            .map(|(schedule, scores)| {
                let count = scores.len();
                let mean = scores.iter().sum::<f64>() / count as f64;
                let variance = scores.iter()
                    .map(|s| (s - mean).powi(2))
                    .sum::<f64>() / count as f64;

                (schedule, ScheduleStats {
                    count,
                    mean_performance: mean,
                    std_performance: variance.sqrt(),
                    best_performance: scores.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                })
            })
            .collect()
    }

    /// Number of records in history
    pub fn num_records(&self) -> usize {
        self.history.len()
    }
}

/// Statistics for a schedule type
#[derive(Debug, Clone)]
pub struct ScheduleStats {
    pub count: usize,
    pub mean_performance: f64,
    pub std_performance: f64,
    pub best_performance: f64,
}

/// Meta-schedule selector with multiple strategies
///
/// High-level interface for schedule selection
pub struct MetaScheduleSelector {
    /// k-NN recommender
    recommender: KNNRecommender,

    /// Exploration rate (epsilon-greedy)
    exploration_rate: f64,

    /// Random number generator
    rng: rand::rngs::StdRng,
}

impl MetaScheduleSelector {
    /// Create new meta-schedule selector
    ///
    /// # Arguments
    /// * `k` - Number of neighbors for k-NN
    /// * `exploration_rate` - Probability of random exploration (0-1)
    pub fn new(k: usize, exploration_rate: f64) -> Self {
        use rand::SeedableRng;
        Self {
            recommender: KNNRecommender::new(k),
            exploration_rate,
            rng: rand::rngs::StdRng::from_entropy(),
        }
    }

    /// Select schedule for problem (with exploration)
    ///
    /// Uses epsilon-greedy: exploit best known schedule or explore randomly
    pub fn select_schedule(&mut self, features: &ProblemFeatures) -> ScheduleType {
        use rand::Rng;

        // Epsilon-greedy exploration
        if self.rng.gen::<f64>() < self.exploration_rate {
            // Explore: random schedule
            let schedules = ScheduleType::all();
            let idx = self.rng.gen_range(0..schedules.len());
            schedules[idx]
        } else {
            // Exploit: use recommender
            self.recommender.recommend(features)
                .map(|(schedule, _confidence)| schedule)
                .unwrap_or(ScheduleType::SimulatedAnnealing) // Default fallback
        }
    }

    /// Recommend schedule without exploration (pure exploitation)
    pub fn recommend_schedule(&self, features: &ProblemFeatures) -> Option<(ScheduleType, f64)> {
        self.recommender.recommend(features)
    }

    /// Record performance of schedule on problem
    pub fn record_performance(&mut self, record: SchedulePerformanceRecord) {
        self.recommender.add_record(record);
    }

    /// Get performance statistics
    pub fn get_statistics(&self) -> HashMap<ScheduleType, ScheduleStats> {
        self.recommender.schedule_statistics()
    }

    /// Decay exploration rate over time (anneal exploration)
    pub fn decay_exploration(&mut self, decay_rate: f64) {
        self.exploration_rate *= decay_rate;
        self.exploration_rate = self.exploration_rate.max(0.01); // Minimum 1%
    }

    /// Get current exploration rate
    pub fn exploration_rate(&self) -> f64 {
        self.exploration_rate
    }

    /// Number of records in history
    pub fn num_records(&self) -> usize {
        self.recommender.num_records()
    }
}

/// Contextual bandit for schedule selection
///
/// Treats schedule selection as a contextual multi-armed bandit problem
pub struct ContextualBandit {
    /// Performance estimates for each schedule in each context
    q_values: HashMap<String, HashMap<ScheduleType, (f64, usize)>>, // (mean_reward, count)

    /// Exploration parameter (UCB)
    exploration_bonus: f64,
}

impl ContextualBandit {
    /// Create new contextual bandit
    ///
    /// # Arguments
    /// * `exploration_bonus` - UCB exploration parameter (typical: 2.0)
    pub fn new(exploration_bonus: f64) -> Self {
        Self {
            q_values: HashMap::new(),
            exploration_bonus,
        }
    }

    /// Select schedule using UCB (Upper Confidence Bound)
    ///
    /// Balances exploitation (high mean reward) and exploration (high uncertainty)
    pub fn select_schedule(&self, features: &ProblemFeatures) -> ScheduleType {
        let context = self.context_hash(features);

        if let Some(context_values) = self.q_values.get(&context) {
            let total_counts: usize = context_values.values().map(|(_, c)| c).sum();

            // Compute UCB for each schedule
            let mut best_schedule = ScheduleType::SimulatedAnnealing;
            let mut best_ucb = f64::NEG_INFINITY;

            for schedule in ScheduleType::all() {
                let (mean_reward, count) = context_values.get(&schedule)
                    .copied()
                    .unwrap_or((0.0, 0));

                let ucb = if count == 0 {
                    f64::INFINITY // Always try untried schedules first
                } else {
                    let bonus = self.exploration_bonus * ((total_counts as f64).ln() / count as f64).sqrt();
                    mean_reward + bonus
                };

                if ucb > best_ucb {
                    best_ucb = ucb;
                    best_schedule = schedule;
                }
            }

            best_schedule
        } else {
            // New context, pick random schedule
            ScheduleType::SimulatedAnnealing
        }
    }

    /// Update reward for schedule in context
    pub fn update(&mut self, features: &ProblemFeatures, schedule: ScheduleType, reward: f64) {
        let context = self.context_hash(features);
        let context_values = self.q_values.entry(context).or_insert_with(HashMap::new);

        let (mean, count) = context_values.entry(schedule).or_insert((0.0, 0));
        let new_count = *count + 1;
        let new_mean = (*mean * (*count as f64) + reward) / (new_count as f64);

        *context_values.get_mut(&schedule).unwrap() = (new_mean, new_count);
    }

    /// Create context identifier from features (discretize continuous features)
    fn context_hash(&self, features: &ProblemFeatures) -> String {
        format!(
            "dim:{}_size:{}_rug:{:.1}_qual:{:.1}",
            features.dimensionality / 10 * 10, // Round to nearest 10
            features.problem_size / 100 * 100, // Round to nearest 100
            (features.ruggedness * 10.0).round() / 10.0, // Round to 0.1
            (features.quality_requirement * 10.0).round() / 10.0, // Round to 0.1
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_problem_features() {
        let features = ProblemFeatures::new(10, 100, 1000.0, 0.9);

        assert_eq!(features.dimensionality, 10);
        assert_eq!(features.problem_size, 100);
        assert_eq!(features.budget, 1000.0);
        assert_eq!(features.quality_requirement, 0.9);
    }

    #[test]
    fn test_ruggedness_estimation() {
        let mut features = ProblemFeatures::new(5, 50, 100.0, 0.5);

        // Smooth landscape (small changes)
        let smooth_energies = vec![1.0, 1.1, 1.2, 1.1, 1.3];
        features.estimate_ruggedness(&smooth_energies);
        assert!(features.ruggedness < 0.5);

        // Rugged landscape (large changes)
        let rugged_energies = vec![1.0, 5.0, 0.5, 4.0, 1.5];
        features.estimate_ruggedness(&rugged_energies);
        assert!(features.ruggedness > 0.5);
    }

    #[test]
    fn test_feature_similarity() {
        let f1 = ProblemFeatures::new(10, 100, 1000.0, 0.9);
        let f2 = ProblemFeatures::new(10, 100, 1000.0, 0.9);
        let f3 = ProblemFeatures::new(100, 1000, 10000.0, 0.1);

        // Identical features should have similarity ~1
        let sim_same = f1.similarity(&f2);
        assert!(sim_same > 0.99);

        // Different features should have lower similarity
        let sim_diff = f1.similarity(&f3);
        assert!(sim_diff < sim_same);
    }

    #[test]
    fn test_schedule_types() {
        let schedules = ScheduleType::all();
        assert_eq!(schedules.len(), 6);

        assert_eq!(ScheduleType::SimulatedAnnealing.name(), "SimulatedAnnealing");
    }

    #[test]
    fn test_performance_score() {
        let record = SchedulePerformanceRecord {
            features: ProblemFeatures::new(10, 100, 1000.0, 0.9),
            schedule: ScheduleType::SimulatedAnnealing,
            quality: 0.9,
            iterations: 100,
            time_seconds: 10.0,
            acceptance_rate: 0.44,
            converged: true,
        };

        let score = record.performance_score();
        assert!(score > 0.0 && score <= 1.0);
    }

    #[test]
    fn test_knn_recommender() {
        let mut recommender = KNNRecommender::new(3);

        // Add some records
        for i in 0..5 {
            let features = ProblemFeatures::new(10 + i, 100, 1000.0, 0.9);
            recommender.add_record(SchedulePerformanceRecord {
                features,
                schedule: ScheduleType::SimulatedAnnealing,
                quality: 0.8 + i as f64 * 0.02,
                iterations: 100,
                time_seconds: 10.0,
                acceptance_rate: 0.44,
                converged: true,
            });
        }

        assert_eq!(recommender.num_records(), 5);

        // Recommend for similar problem
        let test_features = ProblemFeatures::new(12, 100, 1000.0, 0.9);
        let recommendation = recommender.recommend(&test_features);

        assert!(recommendation.is_some());
        let (schedule, confidence) = recommendation.unwrap();
        assert_eq!(schedule, ScheduleType::SimulatedAnnealing);
        assert!(confidence > 0.0 && confidence <= 1.0);
    }

    #[test]
    fn test_meta_selector() {
        let mut selector = MetaScheduleSelector::new(3, 0.1);

        let features = ProblemFeatures::new(10, 100, 1000.0, 0.9);

        // Should work even with no history (exploration)
        let schedule = selector.select_schedule(&features);
        assert!(ScheduleType::all().contains(&schedule));

        // Add some records
        selector.record_performance(SchedulePerformanceRecord {
            features: features.clone(),
            schedule: ScheduleType::ParallelTempering,
            quality: 0.95,
            iterations: 100,
            time_seconds: 10.0,
            acceptance_rate: 0.44,
            converged: true,
        });

        assert_eq!(selector.num_records(), 1);

        // Decay exploration
        let initial_rate = selector.exploration_rate();
        selector.decay_exploration(0.9);
        assert!(selector.exploration_rate() < initial_rate);
    }

    #[test]
    fn test_contextual_bandit() {
        let mut bandit = ContextualBandit::new(2.0);

        let features = ProblemFeatures::new(10, 100, 1000.0, 0.9);

        // Select initial schedule
        let schedule1 = bandit.select_schedule(&features);

        // Update with reward
        bandit.update(&features, schedule1, 0.8);

        // Select again (should consider updated values)
        let schedule2 = bandit.select_schedule(&features);

        // Both schedules should be valid
        assert!(ScheduleType::all().contains(&schedule1));
        assert!(ScheduleType::all().contains(&schedule2));
    }

    #[test]
    fn test_statistics() {
        let mut recommender = KNNRecommender::new(3);

        // Add varied records
        for schedule in [ScheduleType::SimulatedAnnealing, ScheduleType::ParallelTempering] {
            for i in 0..3 {
                let features = ProblemFeatures::new(10, 100, 1000.0, 0.9);
                recommender.add_record(SchedulePerformanceRecord {
                    features,
                    schedule,
                    quality: 0.7 + i as f64 * 0.1,
                    iterations: 100,
                    time_seconds: 10.0,
                    acceptance_rate: 0.44,
                    converged: true,
                });
            }
        }

        let stats = recommender.schedule_statistics();
        assert_eq!(stats.len(), 2);

        for (schedule, stat) in stats {
            assert_eq!(stat.count, 3);
            assert!(stat.mean_performance > 0.0);
            assert!(stat.best_performance >= stat.mean_performance);
        }
    }
}
