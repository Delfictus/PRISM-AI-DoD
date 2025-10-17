//! Scientific Discovery Demo
//!
//! Demonstrates Worker 7's scientific discovery module:
//! - Experiment design optimization
//! - Bayesian parameter exploration
//! - Active learning for efficient discovery
//!
//! Run with: cargo run --example scientific_discovery_demo

use anyhow::Result;
use ndarray::{array, Array1};
use prism_ai::applications::{ScientificDiscovery, ScientificConfig};

fn main() -> Result<()> {
    println!("=== PRISM-AI Scientific Discovery Demo ===\n");

    // 1. Configure scientific discovery
    let config = ScientificConfig {
        max_experiments: 50,
        exploration_weight: 0.3,  // 30% exploration, 70% exploitation
        convergence_threshold: 0.01,
        batch_size: 5,
        use_active_learning: true,
    };

    println!("Configuration:");
    println!("  Max experiments: {}", config.max_experiments);
    println!("  Exploration weight: {:.0}%", config.exploration_weight * 100.0);
    println!("  Active learning: {}\n", config.use_active_learning);

    // 2. Initialize discovery system
    let mut discovery = ScientificDiscovery::new(config)?;
    println!("✓ Scientific discovery system initialized\n");

    // 3. Define parameter space
    // Example: Optimizing a chemical reaction
    // Parameters: [temperature (°C), pressure (bar), catalyst_amount (g), time (min)]
    let param_bounds = vec![
        (50.0, 200.0),    // Temperature: 50-200°C
        (1.0, 10.0),      // Pressure: 1-10 bar
        (0.1, 2.0),       // Catalyst: 0.1-2.0g
        (10.0, 120.0),    // Time: 10-120 min
    ];

    println!("Parameter space:");
    println!("  Temperature: {:.0}-{:.0}°C", param_bounds[0].0, param_bounds[0].1);
    println!("  Pressure: {:.1}-{:.1} bar", param_bounds[1].0, param_bounds[1].1);
    println!("  Catalyst amount: {:.1}-{:.1}g", param_bounds[2].0, param_bounds[2].1);
    println!("  Reaction time: {:.0}-{:.0} min\n", param_bounds[3].0, param_bounds[3].1);

    // 4. Define objective function (simulated experimental outcome)
    // In reality, this would be actual experiments
    let objective_function = |params: &Array1<f64>| -> f64 {
        let temp = params[0];
        let pressure = params[1];
        let catalyst = params[2];
        let time = params[3];

        // Simulated yield function (peak around 150°C, 5 bar, 1g catalyst, 60 min)
        let temp_factor = -((temp - 150.0) / 30.0).powi(2);
        let pressure_factor = -((pressure - 5.0) / 2.0).powi(2);
        let catalyst_factor = -((catalyst - 1.0) / 0.5).powi(2);
        let time_factor = -((time - 60.0) / 20.0).powi(2);

        // Yield percentage (0-100%)
        let base_yield = 85.0;
        let yield_value = base_yield * (temp_factor + pressure_factor +
                                        catalyst_factor + time_factor).exp();

        // Add some noise
        let noise = (params[0] * params[1]).sin() * 2.0;
        (yield_value + noise).max(0.0).min(100.0)
    };

    // 5. Run experiment design loop
    println!("--- Running Experiment Design ---\n");

    let mut best_yield = 0.0;
    let mut best_params = Array1::zeros(4);
    let mut experiment_count = 0;

    // Initial random sampling
    println!("Phase 1: Initial exploration (10 random experiments)");
    for i in 0..10 {
        let params = array![
            param_bounds[0].0 + rand::random::<f64>() * (param_bounds[0].1 - param_bounds[0].0),
            param_bounds[1].0 + rand::random::<f64>() * (param_bounds[1].1 - param_bounds[1].0),
            param_bounds[2].0 + rand::random::<f64>() * (param_bounds[2].1 - param_bounds[2].0),
            param_bounds[3].0 + rand::random::<f64>() * (param_bounds[3].1 - param_bounds[3].0),
        ];

        let yield_value = objective_function(&params);

        discovery.add_observation(params.clone(), yield_value)?;
        experiment_count += 1;

        if yield_value > best_yield {
            best_yield = yield_value;
            best_params = params.clone();
        }

        println!("  Experiment {}: T={:.1}°C, P={:.1}bar, C={:.2}g, t={:.0}min → Yield={:.1}%",
            i + 1, params[0], params[1], params[2], params[3], yield_value);
    }

    println!("\nBest from initial exploration: {:.1}% yield\n", best_yield);

    // Bayesian optimization phase
    println!("Phase 2: Bayesian optimization (active learning)");
    for i in 0..20 {
        // Suggest next experiment using acquisition function
        let next_params = discovery.suggest_next_experiment(&param_bounds)?;

        // "Run" the experiment
        let yield_value = objective_function(&next_params);

        // Update model
        discovery.add_observation(next_params.clone(), yield_value)?;
        experiment_count += 1;

        if yield_value > best_yield {
            best_yield = yield_value;
            best_params = next_params.clone();
            println!("  Experiment {}: T={:.1}°C, P={:.1}bar, C={:.2}g, t={:.0}min → Yield={:.1}% ⭐ NEW BEST",
                10 + i + 1, next_params[0], next_params[1], next_params[2], next_params[3], yield_value);
        } else if i % 5 == 0 {
            println!("  Experiment {}: Yield={:.1}%", 10 + i + 1, yield_value);
        }

        // Check convergence
        if i > 5 {
            let recent_improvement = discovery.get_improvement_rate()?;
            if recent_improvement < config.convergence_threshold {
                println!("\n✓ Converged after {} experiments (improvement < {:.2}%)",
                    experiment_count, config.convergence_threshold * 100.0);
                break;
            }
        }
    }

    // 6. Results
    println!("\n=== Discovery Results ===");
    println!("\nOptimal parameters found:");
    println!("  Temperature: {:.1}°C", best_params[0]);
    println!("  Pressure: {:.1} bar", best_params[1]);
    println!("  Catalyst amount: {:.2}g", best_params[2]);
    println!("  Reaction time: {:.0} min", best_params[3]);
    println!("\nMaximum yield: {:.2}%", best_yield);
    println!("Total experiments: {}", experiment_count);

    // 7. Compare to brute force
    let brute_force_experiments = 5_usize.pow(4); // 5 points per dimension
    let efficiency_gain = (brute_force_experiments as f64) / (experiment_count as f64);

    println!("\n--- Efficiency Analysis ---");
    println!("Active learning used: {} experiments", experiment_count);
    println!("Brute force grid search would need: {} experiments", brute_force_experiments);
    println!("Efficiency gain: {:.1}x faster", efficiency_gain);

    // 8. Active Inference insights
    println!("\n--- Active Inference Insights ---");
    println!("The discovery system used Active Inference to:");
    println!("  1. Balance exploration (trying new parameter regions)");
    println!("     and exploitation (refining near best results)");
    println!("  2. Minimize expected free energy = maximize information gain");
    println!("  3. Actively select experiments that reduce uncertainty");
    println!("  4. Learn a surrogate model (Gaussian Process) of the");
    println!("     experimental outcome surface");
    println!("\nThis approach discovers optimal parameters with {:.0}% fewer experiments",
        (1.0 - 1.0/efficiency_gain) * 100.0);

    println!("\n=== Demo Complete ===");

    Ok(())
}
