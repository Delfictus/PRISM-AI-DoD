# Worker 7 Best Practices Guide

**Constitution**: Worker 7 - Drug Discovery & Robotics
**Version**: 1.0.0
**Date**: 2025-10-13

## Table of Contents

1. [Overview](#overview)
2. [Architecture Patterns](#architecture-patterns)
3. [Performance Optimization](#performance-optimization)
4. [Drug Discovery Workflows](#drug-discovery-workflows)
5. [Robotics Motion Planning](#robotics-motion-planning)
6. [Information-Theoretic Design](#information-theoretic-design)
7. [Testing & Validation](#testing--validation)
8. [Production Deployment](#production-deployment)
9. [Common Pitfalls](#common-pitfalls)
10. [Advanced Topics](#advanced-topics)

---

## Overview

Worker 7 provides Active Inference-based controllers for drug discovery and robotics applications, built on PRISM-AI's neuromorphic and quantum computing foundation. This guide presents battle-tested patterns and practices for production use.

### Core Capabilities

- **Drug Discovery**: Molecular optimization, chemical space exploration, binding affinity prediction
- **Robotics**: Motion planning, trajectory optimization, collision avoidance
- **Scientific Discovery**: Experiment design, parameter optimization, information-theoretic metrics
- **Information Theory**: Differential entropy, mutual information, KL divergence, Expected Information Gain

### Key Design Principles

1. **Information-Theoretic Foundations**: All optimization uses rigorous information theory
2. **Active Inference**: Unified framework for perception, action, and learning
3. **Performance-Critical**: Optimized O(n log n) algorithms with parallelization
4. **Production-Ready**: Comprehensive error handling, logging, and monitoring

---

## Architecture Patterns

### 1. Controller-Based Design

**Pattern**: Use domain-specific controllers (`DrugDiscoveryController`, `RoboticsController`) as the primary interface.

```rust
use prism_ai::applications::{DrugDiscoveryController, DrugDiscoveryConfig};

#[tokio::main]
async fn main() -> Result<()> {
    let config = DrugDiscoveryConfig {
        descriptor_dim: 256,
        population_size: 50,
        num_generations: 20,
        mutation_rate: 0.1,
        target_binding_affinity: 0.9,
    };

    let controller = DrugDiscoveryController::new(config)?;

    // Use controller for optimization
    let result = controller.optimize_molecule(&descriptors).await?;

    Ok(())
}
```

**Why**: Controllers encapsulate Active Inference logic, providing clean APIs and consistent behavior.

**Anti-Pattern**: Directly manipulating internal state or bypassing controller methods.

### 2. Information Metrics Integration

**Pattern**: Use specialized metrics classes for information-theoretic calculations.

```rust
use prism_ai::applications::{
    ExperimentInformationMetrics,
    MolecularInformationMetrics,
    RoboticsInformationMetrics,
};

// For experiment design
let exp_metrics = ExperimentInformationMetrics::new()?;
let entropy = exp_metrics.differential_entropy(&samples)?;
let eig = exp_metrics.expected_information_gain(&prior, &posterior)?;

// For drug discovery
let mol_metrics = MolecularInformationMetrics::new();
let similarity = mol_metrics.molecular_similarity(&mol1, &mol2);
let diversity = mol_metrics.chemical_space_entropy(&library)?;

// For robotics
let rob_metrics = RoboticsInformationMetrics::new();
let traj_entropy = rob_metrics.trajectory_entropy(&trajectory)?;
let info_gain = rob_metrics.sensor_information_gain(prior_var, posterior_var);
```

**Why**: Domain-specific metrics provide semantically meaningful calculations with appropriate units and bounds.

### 3. Optimized vs Baseline Trade-Off

**Pattern**: Use optimized implementations for production, baseline for debugging.

```rust
// Production: Use optimized O(n log n) implementation
use prism_ai::applications::information_metrics_optimized::OptimizedExperimentInformationMetrics;

let metrics = OptimizedExperimentInformationMetrics::new()?;
let entropy = metrics.differential_entropy(&samples)?;  // 5-10x faster

// Development/Debugging: Use baseline O(n²) implementation
use prism_ai::applications::information_metrics::ExperimentInformationMetrics;

let metrics = ExperimentInformationMetrics::new()?;
let entropy = metrics.differential_entropy(&samples)?;  // Simpler, more predictable
```

**Decision Matrix**:

| Condition | Use Optimized | Use Baseline |
|-----------|---------------|--------------|
| n > 100 | ✓ Always | |
| Production deployment | ✓ Always | |
| Real-time requirements | ✓ Always | |
| n < 50 | | ✓ Consider (overhead may dominate) |
| Debugging numerical issues | | ✓ Simpler implementation |
| Teaching/learning | | ✓ Easier to understand |

---

## Performance Optimization

### 1. Dataset Size Considerations

**Rule of Thumb**: Choose implementation based on dataset size.

```rust
fn choose_metrics(n: usize) -> Box<dyn EntropyCalculator> {
    if n < 50 {
        // Small: Overhead may dominate
        Box::new(ExperimentInformationMetrics::new().unwrap())
    } else if n < 500 {
        // Medium: 5-10x speedup expected
        Box::new(OptimizedExperimentInformationMetrics::new().unwrap())
    } else {
        // Large: 10-20x speedup expected
        Box::new(OptimizedExperimentInformationMetrics::new().unwrap())
    }
}
```

**Benchmarking**: Always measure performance for your specific workload.

```rust
use std::time::Instant;

let start = Instant::now();
let entropy = metrics.differential_entropy(&samples)?;
let duration = start.elapsed();

log::info!("Entropy calculation: {:.2?} for n={}", duration, samples.nrows());
```

### 2. Parallel Computing

**Pattern**: Leverage rayon for CPU parallelization in optimized implementations.

```rust
use rayon::prelude::*;

// Optimized implementation automatically uses parallelization
let metrics = OptimizedExperimentInformationMetrics::new()?;

// Parallel entropy calculations for multiple datasets
let entropies: Vec<f64> = datasets.par_iter()
    .map(|data| metrics.differential_entropy(data).unwrap())
    .collect();

// Parallel mutual information with rayon::join
let (h_x, h_y) = rayon::join(
    || metrics.differential_entropy(x_samples),
    || metrics.differential_entropy(y_samples),
);
```

**Configuration**: Set thread pool size via environment variable:

```bash
export RAYON_NUM_THREADS=8  # Use 8 CPU cores
```

### 3. Memory Management

**Pattern**: Reuse allocations when processing multiple datasets.

```rust
// Good: Reuse metrics object
let metrics = OptimizedExperimentInformationMetrics::new()?;

for dataset in datasets {
    let entropy = metrics.differential_entropy(&dataset)?;
    // KD-tree is rebuilt, but metrics object is reused
}

// Bad: Recreate metrics object every iteration
for dataset in datasets {
    let metrics = OptimizedExperimentInformationMetrics::new()?;  // Unnecessary allocation
    let entropy = metrics.differential_entropy(&dataset)?;
}
```

**Pre-allocation**: Pre-allocate arrays when size is known.

```rust
use ndarray::Array2;

// Pre-allocate matrix
let mut results = Array2::zeros((num_experiments, descriptor_dim));

for (i, experiment) in experiments.iter().enumerate() {
    let descriptors = run_experiment(experiment)?;
    results.row_mut(i).assign(&descriptors);
}
```

### 4. Batch Processing

**Pattern**: Process multiple queries in batches for better cache utilization.

```rust
// Good: Batch processing
let all_descriptors = Array2::from_shape_vec((n, d), flat_data)?;
let entropy = metrics.differential_entropy(&all_descriptors)?;

// Bad: Individual processing
let mut sum_entropy = 0.0;
for descriptor in descriptors {
    let single = Array2::from_shape_vec((1, d), descriptor.to_vec())?;
    sum_entropy += metrics.differential_entropy(&single)?;  // Inefficient
}
```

---

## Drug Discovery Workflows

### 1. Molecular Library Screening

**Pattern**: Use Expected Information Gain for active learning.

```rust
use prism_ai::applications::{
    DrugDiscoveryController,
    information_metrics_optimized::OptimizedExperimentInformationMetrics,
};

async fn active_screening(
    library: &mut Vec<Molecule>,
    controller: &DrugDiscoveryController,
    budget: usize,
) -> Result<Vec<Molecule>> {
    let metrics = OptimizedExperimentInformationMetrics::new()?;
    let mut selected = Vec::new();

    for round in 0..budget {
        // Calculate EIG for each unscreened molecule
        let eig_scores = calculate_eig_scores(library, &metrics)?;

        // Select molecule with highest EIG
        let best_idx = eig_scores.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        // Perform experiment (measure binding affinity)
        let affinity = measure_binding_affinity(&library[best_idx]).await?;
        library[best_idx].set_affinity(affinity);

        selected.push(library[best_idx].clone());
    }

    Ok(selected)
}
```

**Why**: EIG-based selection minimizes experiments while maximizing information gain.

**Metrics**:
- **Efficiency**: Molecules screened / Total library size
- **Quality**: Best affinity found
- **Cost**: Total synthesis + assay costs

### 2. Chemical Space Exploration

**Pattern**: Monitor chemical space entropy to ensure diversity.

```rust
use prism_ai::applications::MolecularInformationMetrics;

fn ensure_diversity(library: &[Molecule], threshold: f64) -> Result<bool> {
    let metrics = MolecularInformationMetrics::new();

    let descriptors = stack_descriptors(library)?;
    let entropy = metrics.chemical_space_entropy(&descriptors)?;

    log::info!("Chemical space entropy: {:.4} nats", entropy);

    if entropy < threshold {
        log::warn!("Low chemical diversity! Consider expanding library.");
        Ok(false)
    } else {
        Ok(true)
    }
}
```

**Thresholds** (empirical guidelines):
- **High diversity**: H > 5.0 nats
- **Moderate diversity**: 3.0 < H < 5.0 nats
- **Low diversity**: H < 3.0 nats (consider expanding)

### 3. Binding Affinity Optimization

**Pattern**: Use Active Inference controller for molecular optimization.

```rust
use prism_ai::applications::DrugDiscoveryController;

async fn optimize_lead_compound(
    seed_molecule: &Molecule,
    controller: &DrugDiscoveryController,
    target_affinity: f64,
) -> Result<Molecule> {
    log::info!("Optimizing from seed: {} (affinity: {:.4})",
        seed_molecule.id, seed_molecule.affinity);

    let result = controller.optimize_molecule(&seed_molecule.descriptors).await?;

    log::info!("Optimization complete:");
    log::info!("  • Final affinity: {:.4}", result.binding_affinity);
    log::info!("  • Improvement: {:.2}%",
        ((result.binding_affinity - seed_molecule.affinity) / seed_molecule.affinity) * 100.0);

    if result.binding_affinity >= target_affinity {
        log::info!("  ✓ Target affinity achieved!");
    } else {
        log::warn!("  ⚠ Target affinity not reached");
    }

    Ok(result.to_molecule())
}
```

**Configuration Guidelines**:
- `population_size`: 30-100 (larger for complex molecules)
- `num_generations`: 10-50 (balance quality vs time)
- `mutation_rate`: 0.05-0.2 (higher for exploration)

### 4. Multi-Objective Optimization

**Pattern**: Balance multiple objectives (affinity, ADMET properties, synthesis cost).

```rust
struct MultiObjectiveScore {
    binding_affinity: f64,    // Maximize
    admet_score: f64,         // Maximize (drug-likeness)
    synthesis_cost: f64,      // Minimize
}

impl MultiObjectiveScore {
    fn weighted_score(&self, weights: &[f64; 3]) -> f64 {
        weights[0] * self.binding_affinity
            + weights[1] * self.admet_score
            - weights[2] * (self.synthesis_cost / 10000.0)  // Normalize cost
    }
}

fn select_best_candidate(
    molecules: &[Molecule],
    weights: &[f64; 3],
) -> Result<&Molecule> {
    molecules.iter()
        .max_by(|a, b| {
            let score_a = a.multi_objective_score().weighted_score(weights);
            let score_b = b.multi_objective_score().weighted_score(weights);
            score_a.partial_cmp(&score_b).unwrap()
        })
        .ok_or_else(|| anyhow::anyhow!("Empty molecule list"))
}
```

**Typical Weights**:
- **Lead optimization**: [0.6, 0.3, 0.1] (prioritize affinity)
- **Hit-to-lead**: [0.4, 0.4, 0.2] (balance affinity and ADMET)
- **Cost-constrained**: [0.3, 0.3, 0.4] (minimize synthesis cost)

---

## Robotics Motion Planning

### 1. Collision-Free Path Planning

**Pattern**: Use MotionPlanner with occupancy grids for collision avoidance.

```rust
use prism_ai::applications::MotionPlanner;
use ndarray::Array2;

async fn plan_collision_free_path(
    start: &Array1<f64>,
    goal: &Array1<f64>,
    obstacles: &[Obstacle],
    workspace_size: (f64, f64),
) -> Result<MotionPlan> {
    let planner = MotionPlanner::new(start.len())?;

    // Convert obstacles to occupancy grid
    let occupancy_grid = create_occupancy_grid(obstacles, workspace_size, 100)?;

    // Plan path
    let plan = planner.plan(start, goal, Some(&occupancy_grid)).await?;

    // Validate collision-free
    let collisions = count_collisions(&plan, obstacles);
    if collisions > 0 {
        log::warn!("Path contains {} collision points!", collisions);
    }

    Ok(plan)
}
```

**Occupancy Grid Resolution**:
- **Low (50×50)**: Fast planning, coarse obstacles
- **Medium (100×100)**: Balance speed and accuracy
- **High (200×200)**: Precise planning, slower

### 2. Trajectory Optimization

**Pattern**: Use information-theoretic metrics to evaluate trajectory quality.

```rust
use prism_ai::applications::RoboticsInformationMetrics;

fn evaluate_trajectory(
    trajectory: &Array2<f64>,
    metrics: &RoboticsInformationMetrics,
) -> Result<TrajectoryScore> {
    // Calculate trajectory entropy
    let entropy = metrics.trajectory_entropy(trajectory)?;

    // Calculate path smoothness (variance of accelerations)
    let smoothness = calculate_smoothness(trajectory);

    // Calculate safety margin
    let min_clearance = calculate_min_clearance(trajectory);

    Ok(TrajectoryScore {
        entropy,
        smoothness,
        safety: min_clearance,
    })
}

struct TrajectoryScore {
    entropy: f64,      // Higher = more exploratory
    smoothness: f64,   // Lower = smoother motion
    safety: f64,       // Higher = safer (more clearance)
}
```

**Interpretation**:
- **High entropy + low smoothness**: Erratic, exploratory motion
- **Low entropy + high smoothness**: Direct, efficient motion
- **Balanced**: Smooth exploration with safety margins

### 3. Real-Time Replanning

**Pattern**: Monitor environment and replan when obstacles change.

```rust
use tokio::time::{interval, Duration};

async fn reactive_planning(
    start: &Array1<f64>,
    goal: &Array1<f64>,
    planner: &MotionPlanner,
) -> Result<()> {
    let mut current_plan = planner.plan(start, goal, None).await?;
    let mut current_pos = start.clone();

    let mut update_timer = interval(Duration::from_millis(100));  // 10 Hz

    loop {
        update_timer.tick().await;

        // Check for dynamic obstacles
        let obstacles = detect_obstacles().await?;

        if obstacles_changed(&obstacles) {
            log::info!("Obstacles detected, replanning...");

            let replan_start = Instant::now();
            let occupancy = create_occupancy_grid(&obstacles, (10.0, 10.0), 100)?;
            current_plan = planner.plan(&current_pos, goal, Some(&occupancy)).await?;

            log::info!("Replanning complete: {:.2?}", replan_start.elapsed());
        }

        // Execute next waypoint
        if let Some(next_waypoint) = current_plan.waypoints.get(0) {
            execute_motion(&current_pos, next_waypoint).await?;
            current_pos = next_waypoint.clone();
            current_plan.waypoints.remove(0);
        }

        // Check if goal reached
        if euclidean_distance(&current_pos, goal) < 0.1 {
            log::info!("Goal reached!");
            break;
        }
    }

    Ok(())
}
```

**Performance Requirements**:
- **Replanning latency**: < 100ms for real-time systems
- **Update frequency**: 5-20 Hz depending on robot speed
- **Lookahead horizon**: 1-5 seconds of motion

### 4. Sensor Fusion with Information Gain

**Pattern**: Weight sensor measurements by information gain.

```rust
use prism_ai::applications::RoboticsInformationMetrics;

fn weighted_sensor_fusion(
    sensors: &[SensorReading],
    metrics: &RoboticsInformationMetrics,
) -> Result<State> {
    let mut weighted_state = State::zero();
    let mut total_weight = 0.0;

    for sensor in sensors {
        // Calculate information gain from this sensor
        let info_gain = metrics.sensor_information_gain(
            sensor.prior_variance,
            sensor.posterior_variance,
        );

        // Weight sensor reading by information gain
        let weight = info_gain.exp();  // Higher IG → higher weight
        weighted_state = weighted_state + sensor.reading * weight;
        total_weight += weight;
    }

    Ok(weighted_state / total_weight)
}
```

**Why**: Sensors with higher information gain get more weight, improving state estimation accuracy.

---

## Information-Theoretic Design

### 1. Choosing k for k-NN Entropy Estimation

**Pattern**: Select k based on dataset size and dimensionality.

```rust
fn choose_k_neighbors(n: usize, d: usize) -> usize {
    // Kozachenko-Leonenko estimator guidelines
    if n < 50 {
        3  // Minimum for stability
    } else if n < 200 {
        5  // Standard choice
    } else if d > 10 {
        // Higher dimensions need more neighbors
        (7 + d / 5).min(n / 10)
    } else {
        5
    }
}
```

**Rules of Thumb**:
- **Minimum**: k ≥ 3 for numerical stability
- **Standard**: k = 5 works well for most cases
- **High-dimensional**: k = 5 + d/10
- **Maximum**: k < n/10 to avoid boundary effects

### 2. Interpreting Entropy Values

**Context-Dependent Interpretation**:

```rust
fn interpret_entropy(h: f64, dim: usize, data_type: &str) -> String {
    match data_type {
        "molecular" => {
            if h > 5.0 {
                "High diversity - good chemical space coverage"
            } else if h > 3.0 {
                "Moderate diversity - acceptable"
            } else {
                "Low diversity - consider expanding library"
            }
        }
        "trajectory" => {
            if h > 4.0 {
                "Exploratory trajectory - high uncertainty"
            } else if h > 2.0 {
                "Balanced trajectory"
            } else {
                "Direct trajectory - low uncertainty"
            }
        }
        _ => "Entropy depends on context and units"
    }.to_string()
}
```

**Absolute Values**: Entropy is in nats (natural logarithm units). Convert to bits: `H_bits = H_nats / ln(2) ≈ H_nats * 1.44`.

### 3. Mutual Information Bounds

**Pattern**: Enforce information-theoretic inequalities.

```rust
fn validate_mutual_information(
    i_xy: f64,
    h_x: f64,
    h_y: f64,
) -> Result<f64> {
    // Non-negativity
    if i_xy < 0.0 {
        log::warn!("Negative MI detected: {:.6}, clamping to 0", i_xy);
        return Ok(0.0);
    }

    // Upper bound: I(X;Y) ≤ min(H(X), H(Y))
    let max_mi = h_x.min(h_y);
    if i_xy > max_mi {
        log::warn!("MI {:.6} exceeds bound {:.6}, clamping", i_xy, max_mi);
        return Ok(max_mi);
    }

    Ok(i_xy)
}
```

**Key Inequalities**:
- **Non-negativity**: I(X;Y) ≥ 0
- **Upper bound**: I(X;Y) ≤ min(H(X), H(Y))
- **Symmetry**: I(X;Y) = I(Y;X)
- **Chain rule**: I(X;Y,Z) = I(X;Y) + I(X;Z|Y)

### 4. Expected Information Gain for Experiment Design

**Pattern**: Select experiments that maximize information gain.

```rust
use prism_ai::applications::information_metrics_optimized::OptimizedExperimentInformationMetrics;

async fn optimal_experiment_design(
    prior_samples: &Array2<f64>,
    candidate_experiments: &[Experiment],
    metrics: &OptimizedExperimentInformationMetrics,
) -> Result<Experiment> {
    let mut best_experiment = None;
    let mut max_eig = f64::NEG_INFINITY;

    for experiment in candidate_experiments {
        // Simulate posterior distribution if this experiment were performed
        let posterior_samples = simulate_experiment_outcome(prior_samples, experiment).await?;

        // Calculate expected information gain
        let eig = metrics.expected_information_gain(prior_samples, &posterior_samples)?;

        if eig > max_eig {
            max_eig = eig;
            best_experiment = Some(experiment.clone());
        }
    }

    log::info!("Selected experiment with EIG: {:.4} nats", max_eig);

    best_experiment.ok_or_else(|| anyhow::anyhow!("No valid experiments"))
}
```

**Interpretation**:
- **High EIG (> 2 nats)**: Experiment will significantly reduce uncertainty
- **Medium EIG (0.5-2 nats)**: Moderate information gain
- **Low EIG (< 0.5 nats)**: Minimal new information, consider alternatives

---

## Testing & Validation

### 1. Unit Testing Information Metrics

**Pattern**: Validate against known distributions and mathematical properties.

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_entropy_uniform_distribution() {
        let metrics = OptimizedExperimentInformationMetrics::new().unwrap();

        // Uniform distribution on [0, 1]²
        let samples = generate_uniform_samples(1000, 2, 0.0, 1.0);

        let h = metrics.differential_entropy(&samples).unwrap();

        // Analytical entropy of uniform distribution: ln(volume)
        let expected = (1.0_f64).ln();  // ln(1) = 0 for unit square

        assert_relative_eq!(h, expected, epsilon = 0.5);  // Allow 0.5 nats tolerance
    }

    #[test]
    fn test_mutual_information_independence() {
        let metrics = OptimizedExperimentInformationMetrics::new().unwrap();

        // Independent random variables
        let x = generate_uniform_samples(500, 1, 0.0, 1.0);
        let y = generate_uniform_samples(500, 1, 0.0, 1.0);

        let i_xy = metrics.mutual_information(&x, &y).unwrap();

        // Independent variables should have MI ≈ 0
        assert!(i_xy < 0.1, "MI = {:.4} > 0.1 for independent variables", i_xy);
    }

    #[test]
    fn test_entropy_non_negativity() {
        let metrics = OptimizedExperimentInformationMetrics::new().unwrap();

        let samples = generate_gaussian_samples(200, 3);

        let h = metrics.differential_entropy(&samples).unwrap();

        assert!(h.is_finite(), "Entropy must be finite");
        // Note: Differential entropy can be negative for very peaked distributions
    }
}
```

### 2. Integration Testing Workflows

**Pattern**: Test end-to-end workflows with realistic scenarios.

```rust
#[tokio::test]
async fn test_drug_discovery_workflow() {
    let config = DrugDiscoveryConfig {
        descriptor_dim: 128,
        population_size: 30,
        num_generations: 10,
        mutation_rate: 0.1,
        target_binding_affinity: 0.8,
    };

    let controller = DrugDiscoveryController::new(config).unwrap();

    // Generate test molecule
    let seed_descriptors = Array1::from_vec(vec![0.5; 128]);

    // Run optimization
    let result = controller.optimize_molecule(&seed_descriptors).await.unwrap();

    // Validate results
    assert!(result.binding_affinity > 0.0);
    assert!(result.binding_affinity <= 1.0);
    assert!(result.descriptors.len() == 128);
}

#[tokio::test]
async fn test_motion_planning_collision_free() {
    let planner = MotionPlanner::new(2).unwrap();

    let start = Array1::from_vec(vec![0.0, 0.0]);
    let goal = Array1::from_vec(vec![10.0, 10.0]);

    // Create occupancy grid with obstacles
    let mut occupancy = Array2::ones((100, 100));
    add_circular_obstacle(&mut occupancy, (50, 50), 10);  // Obstacle at center

    let plan = planner.plan(&start, &goal, Some(&occupancy)).await.unwrap();

    // Validate plan
    assert!(plan.waypoints.len() > 0);
    assert_eq!(plan.waypoints[0], start);
    assert!(euclidean_distance(&plan.waypoints.last().unwrap(), &goal) < 1.0);

    // Validate collision-free
    for waypoint in &plan.waypoints {
        let (i, j) = waypoint_to_grid_index(waypoint, 100);
        assert_eq!(occupancy[[i, j]], 1.0, "Waypoint in collision!");
    }
}
```

### 3. Numerical Stability Testing

**Pattern**: Test extreme cases and edge conditions.

```rust
#[test]
fn test_entropy_numerical_stability() {
    let metrics = OptimizedExperimentInformationMetrics::new().unwrap();

    // Test with very small values
    let small_samples = Array2::from_elem((100, 2), 1e-10);
    assert!(metrics.differential_entropy(&small_samples).is_ok());

    // Test with very large values
    let large_samples = Array2::from_elem((100, 2), 1e10);
    assert!(metrics.differential_entropy(&large_samples).is_ok());

    // Test with mixed scales
    let mut mixed_samples = Array2::zeros((100, 2));
    mixed_samples.column_mut(0).fill(1e-6);
    mixed_samples.column_mut(1).fill(1e6);
    assert!(metrics.differential_entropy(&mixed_samples).is_ok());
}

#[test]
fn test_kl_divergence_identity() {
    let metrics = OptimizedExperimentInformationMetrics::new().unwrap();

    let samples = generate_gaussian_samples(200, 3);

    // KL(P || P) should be 0
    let kl = metrics.kl_divergence(&samples, &samples).unwrap();

    assert!(kl < 0.01, "KL(P || P) = {:.6} should be ≈ 0", kl);
}
```

### 4. Performance Regression Testing

**Pattern**: Monitor performance metrics to detect regressions.

```rust
#[test]
fn test_optimization_performance() {
    let metrics = OptimizedExperimentInformationMetrics::new().unwrap();
    let baseline_metrics = ExperimentInformationMetrics::new().unwrap();

    let samples = generate_gaussian_samples(500, 5);

    // Measure optimized performance
    let start = Instant::now();
    let _ = metrics.differential_entropy(&samples).unwrap();
    let optimized_duration = start.elapsed();

    // Measure baseline performance
    let start = Instant::now();
    let _ = baseline_metrics.differential_entropy(&samples).unwrap();
    let baseline_duration = start.elapsed();

    let speedup = baseline_duration.as_secs_f64() / optimized_duration.as_secs_f64();

    println!("Speedup: {:.2}x", speedup);
    assert!(speedup > 3.0, "Expected >3x speedup, got {:.2}x", speedup);
}
```

---

## Production Deployment

### 1. Error Handling

**Pattern**: Use `anyhow::Result` for rich error context.

```rust
use anyhow::{Result, Context};

async fn production_workflow(config: &Config) -> Result<Output> {
    let controller = DrugDiscoveryController::new(config.clone())
        .context("Failed to initialize drug discovery controller")?;

    let library = load_molecular_library(&config.library_path)
        .context(format!("Failed to load library from {}", config.library_path))?;

    let result = controller.optimize_molecule(&library[0].descriptors).await
        .context("Molecular optimization failed")?;

    save_results(&result, &config.output_path)
        .context("Failed to save results")?;

    Ok(result)
}
```

**Error Categories**:
- **Configuration errors**: Invalid parameters, missing files
- **Numerical errors**: Non-finite values, convergence failures
- **Resource errors**: Memory exhaustion, timeout
- **I/O errors**: File system, network issues

### 2. Logging and Monitoring

**Pattern**: Use structured logging for observability.

```rust
use log::{info, warn, error};

async fn monitored_optimization(
    controller: &DrugDiscoveryController,
    molecule: &Molecule,
) -> Result<OptimizationResult> {
    info!("Starting optimization for molecule: {}", molecule.id);
    info!("Initial affinity: {:.4}", molecule.affinity);

    let start = Instant::now();

    match controller.optimize_molecule(&molecule.descriptors).await {
        Ok(result) => {
            let duration = start.elapsed();

            info!("Optimization successful");
            info!("  • Duration: {:.2?}", duration);
            info!("  • Final affinity: {:.4}", result.binding_affinity);
            info!("  • Improvement: {:.2}%",
                ((result.binding_affinity - molecule.affinity) / molecule.affinity) * 100.0);

            // Monitor for anomalies
            if result.binding_affinity > 1.0 {
                warn!("Anomalous binding affinity > 1.0: {:.4}", result.binding_affinity);
            }

            if duration > Duration::from_secs(60) {
                warn!("Optimization took longer than expected: {:.2?}", duration);
            }

            Ok(result)
        }
        Err(e) => {
            error!("Optimization failed for molecule {}: {}", molecule.id, e);
            Err(e)
        }
    }
}
```

**Key Metrics to Log**:
- **Execution time**: Track performance
- **Input/output values**: Debug numerical issues
- **Resource usage**: Memory, CPU
- **Error rates**: Monitor system health

### 3. Configuration Management

**Pattern**: Use TOML configuration files for flexibility.

```rust
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, Deserialize, Serialize)]
struct WorkerConfig {
    drug_discovery: DrugDiscoverySettings,
    robotics: RoboticsSettings,
    information_theory: InfoTheorySettings,
}

#[derive(Debug, Deserialize, Serialize)]
struct DrugDiscoverySettings {
    descriptor_dim: usize,
    population_size: usize,
    num_generations: usize,
    mutation_rate: f64,
    target_binding_affinity: f64,
    #[serde(default = "default_library_path")]
    library_path: String,
}

fn default_library_path() -> String {
    "data/molecular_library.json".to_string()
}

fn load_config(path: &str) -> Result<WorkerConfig> {
    let contents = fs::read_to_string(path)
        .context(format!("Failed to read config file: {}", path))?;

    let config: WorkerConfig = toml::from_str(&contents)
        .context("Failed to parse TOML config")?;

    Ok(config)
}

// Usage
fn main() -> Result<()> {
    let config = load_config("worker7_config.toml")?;

    let controller = DrugDiscoveryController::new(DrugDiscoveryConfig {
        descriptor_dim: config.drug_discovery.descriptor_dim,
        population_size: config.drug_discovery.population_size,
        num_generations: config.drug_discovery.num_generations,
        mutation_rate: config.drug_discovery.mutation_rate,
        target_binding_affinity: config.drug_discovery.target_binding_affinity,
    })?;

    Ok(())
}
```

**Example `worker7_config.toml`**:

```toml
[drug_discovery]
descriptor_dim = 256
population_size = 50
num_generations = 20
mutation_rate = 0.1
target_binding_affinity = 0.9
library_path = "data/chembl_library.json"

[robotics]
state_dim = 3
action_dim = 3
planning_horizon = 100
max_velocity = 1.0
obstacle_threshold = 0.3

[information_theory]
use_optimized = true
k_neighbors = 5
parallel_threads = 8
```

### 4. Graceful Degradation

**Pattern**: Fall back to simpler methods when optimizations fail.

```rust
fn robust_entropy_calculation(samples: &Array2<f64>) -> Result<f64> {
    // Try optimized implementation first
    match OptimizedExperimentInformationMetrics::new()
        .and_then(|m| m.differential_entropy(samples))
    {
        Ok(entropy) if entropy.is_finite() => Ok(entropy),
        Ok(entropy) => {
            warn!("Non-finite entropy from optimized: {}", entropy);
            fallback_entropy(samples)
        }
        Err(e) => {
            warn!("Optimized entropy failed: {}, falling back", e);
            fallback_entropy(samples)
        }
    }
}

fn fallback_entropy(samples: &Array2<f64>) -> Result<f64> {
    // Use baseline implementation as fallback
    ExperimentInformationMetrics::new()?
        .differential_entropy(samples)
        .context("Fallback entropy calculation also failed")
}
```

---

## Common Pitfalls

### 1. Insufficient Sample Size

**Problem**: Using too few samples for information-theoretic calculations.

```rust
// ❌ Bad: Too few samples
let samples = Array2::from_shape_vec((10, 5), data)?;
let entropy = metrics.differential_entropy(&samples)?;  // Unreliable!

// ✓ Good: Adequate sample size
let samples = Array2::from_shape_vec((100, 5), data)?;
let entropy = metrics.differential_entropy(&samples)?;
```

**Rule of Thumb**: n > 10 * k_neighbors (typically n > 50).

### 2. Dimension Mismatch

**Problem**: Inconsistent dimensionality between datasets.

```rust
// ❌ Bad: Dimension mismatch
let x = Array2::from_shape_vec((100, 3), x_data)?;
let y = Array2::from_shape_vec((100, 2), y_data)?;  // Different d!
let mi = metrics.mutual_information(&x, &y)?;  // Error!

// ✓ Good: Consistent dimensions
let x = Array2::from_shape_vec((100, 3), x_data)?;
let y = Array2::from_shape_vec((100, 3), y_data)?;
let mi = metrics.mutual_information(&x, &y)?;
```

**Validation**: Always check dimensions before calculations.

### 3. Ignoring Numerical Bounds

**Problem**: Not validating information-theoretic properties.

```rust
// ❌ Bad: No validation
let mi = metrics.mutual_information(&x, &y)?;
// Use mi directly without checking bounds

// ✓ Good: Validate bounds
let mi = metrics.mutual_information(&x, &y)?;
let h_x = metrics.differential_entropy(&x)?;
let h_y = metrics.differential_entropy(&y)?;

if mi < 0.0 || mi > h_x.min(h_y) {
    warn!("MI outside valid bounds: {:.4}", mi);
    mi = mi.max(0.0).min(h_x.min(h_y));
}
```

### 4. Not Handling Async Properly

**Problem**: Blocking async functions or not awaiting properly.

```rust
// ❌ Bad: Blocking in async context
#[tokio::main]
async fn main() -> Result<()> {
    let result = controller.optimize_molecule(&descriptors);  // Missing .await!
    Ok(())
}

// ✓ Good: Proper async/await
#[tokio::main]
async fn main() -> Result<()> {
    let result = controller.optimize_molecule(&descriptors).await?;
    Ok(())
}
```

### 5. Memory Leaks in Long-Running Processes

**Problem**: Not cleaning up resources in long-running loops.

```rust
// ❌ Bad: Potential memory leak
loop {
    let metrics = OptimizedExperimentInformationMetrics::new()?;  // Created every iteration
    let entropy = metrics.differential_entropy(&samples)?;
    // metrics dropped but KD-tree memory may accumulate
}

// ✓ Good: Reuse objects
let metrics = OptimizedExperimentInformationMetrics::new()?;
loop {
    let entropy = metrics.differential_entropy(&samples)?;
    // metrics reused, KD-tree rebuilt efficiently
}
```

---

## Advanced Topics

### 1. Custom Information Metrics

**Pattern**: Implement domain-specific metrics using Worker 7's foundations.

```rust
use prism_ai::applications::information_metrics_optimized::OptimizedExperimentInformationMetrics;

struct ProteinInformationMetrics {
    base_metrics: OptimizedExperimentInformationMetrics,
}

impl ProteinInformationMetrics {
    pub fn new() -> Result<Self> {
        Ok(Self {
            base_metrics: OptimizedExperimentInformationMetrics::new()?,
        })
    }

    /// Calculate sequence entropy for protein sequences
    pub fn sequence_entropy(&self, sequences: &[String]) -> Result<f64> {
        // Convert sequences to numerical descriptors
        let descriptors = sequences_to_descriptors(sequences)?;

        // Use base metrics for calculation
        self.base_metrics.differential_entropy(&descriptors)
    }

    /// Calculate structural similarity information
    pub fn structural_similarity(&self, protein1: &Protein, protein2: &Protein) -> f64 {
        // Custom structural comparison using information theory
        let desc1 = protein1.structural_descriptors();
        let desc2 = protein2.structural_descriptors();

        // Use cosine similarity weighted by information content
        cosine_similarity(&desc1, &desc2)
    }
}
```

### 2. GPU Acceleration Integration

**Future Enhancement**: Integrate with PRISM-AI's GPU infrastructure.

```rust
// Pseudocode for future GPU integration

#[cfg(feature = "cuda")]
use prism_ai::gpu::GpuKernelExecutor;

pub struct GpuAcceleratedMetrics {
    executor: GpuKernelExecutor,
    base_metrics: OptimizedExperimentInformationMetrics,
}

impl GpuAcceleratedMetrics {
    pub fn new() -> Result<Self> {
        let executor = GpuKernelExecutor::new()?;
        let base_metrics = OptimizedExperimentInformationMetrics::new()?;

        Ok(Self { executor, base_metrics })
    }

    pub async fn differential_entropy_gpu(&self, samples: &Array2<f64>) -> Result<f64> {
        if samples.nrows() > 1000 {
            // Use GPU for large datasets
            self.gpu_entropy(samples).await
        } else {
            // Use CPU for small datasets (avoid transfer overhead)
            self.base_metrics.differential_entropy(samples)
        }
    }
}
```

### 3. Distributed Computing

**Pattern**: Scale to distributed systems for large-scale screening.

```rust
use tokio::task;

async fn distributed_screening(
    library: Vec<Molecule>,
    num_workers: usize,
) -> Result<Vec<ScoredMolecule>> {
    let chunk_size = library.len() / num_workers;
    let chunks: Vec<_> = library.chunks(chunk_size).collect();

    let mut tasks = Vec::new();

    for chunk in chunks {
        let chunk_owned = chunk.to_vec();
        tasks.push(task::spawn(async move {
            screen_molecules(chunk_owned).await
        }));
    }

    let results: Vec<_> = futures::future::try_join_all(tasks).await?;

    // Flatten results
    let all_results: Vec<_> = results.into_iter().flatten().collect();

    Ok(all_results)
}
```

### 4. Active Learning Strategies

**Advanced Pattern**: Implement sophisticated active learning beyond EIG.

```rust
enum AcquisitionStrategy {
    ExpectedImprovement,     // Maximize improvement over current best
    UpperConfidenceBound,    // Balance exploitation and exploration
    ExpectedInformationGain, // Maximize information (Worker 7 default)
    ThompsonSampling,        // Bayesian optimization
}

async fn adaptive_screening(
    library: &mut Vec<Molecule>,
    strategy: AcquisitionStrategy,
    budget: usize,
) -> Result<Vec<Molecule>> {
    match strategy {
        AcquisitionStrategy::ExpectedInformationGain => {
            eig_based_screening(library, budget).await
        }
        AcquisitionStrategy::ExpectedImprovement => {
            ei_based_screening(library, budget).await
        }
        // ... other strategies
    }
}
```

---

## Conclusion

This guide provides production-ready patterns for Worker 7 applications. Key takeaways:

1. **Use domain-specific controllers** for clean, maintainable code
2. **Choose optimized implementations** for production performance
3. **Validate information-theoretic bounds** for numerical stability
4. **Monitor and log comprehensively** for observability
5. **Test rigorously** including edge cases and performance regressions

For questions or contributions, refer to:
- **Source Code**: `03-Source-Code/src/applications/`
- **Examples**: `03-Source-Code/examples/worker7_*.rs`
- **Tests**: `03-Source-Code/tests/worker7_integration_test.rs`
- **Performance Report**: `03-Source-Code/PERFORMANCE_OPTIMIZATION_REPORT.md`

**Worker 7**: Advancing drug discovery and robotics through Active Inference and information theory! 🧬🤖
