//! Trajectory Forecasting Demo
//!
//! Demonstrates Worker 7's advanced trajectory forecasting capabilities
//! using Worker 1's time series module (ARIMA/LSTM).
//!
//! Features demonstrated:
//! - Obstacle trajectory prediction with ARIMA
//! - Environment dynamics forecasting
//! - Multi-agent trajectory forecasting
//! - Uncertainty quantification

use prism_ai::applications::robotics::{
    AdvancedTrajectoryForecaster, TrajectoryForecastConfig,
    TrajectoryPoint, EnvironmentSnapshot, AgentState, ObstacleModel,
};
use ndarray::Array1;
use anyhow::Result;

fn main() -> Result<()> {
    println!("=== Worker 7: Advanced Trajectory Forecasting Demo ===\n");

    // Demo 1: Obstacle Trajectory Forecasting with ARIMA
    demo_obstacle_forecasting()?;

    // Demo 2: Environment Dynamics Prediction
    demo_environment_dynamics()?;

    // Demo 3: Multi-Agent Trajectory Forecasting
    demo_multi_agent()?;

    println!("\n=== Demo Complete ===");
    println!("Worker 7 trajectory forecasting integrates:");
    println!("  ✓ Worker 1's ARIMA/LSTM time series models");
    println!("  ✓ Uncertainty quantification");
    println!("  ✓ GPU-accelerated forecasting");
    println!("  ✓ Multi-agent coordination");

    Ok(())
}

fn demo_obstacle_forecasting() -> Result<()> {
    println!("Demo 1: Obstacle Trajectory Forecasting");
    println!("---------------------------------------");

    // Configure forecaster with ARIMA (fast, good for robotics)
    let config = TrajectoryForecastConfig {
        horizon: 3.0,  // 3 seconds ahead
        dt: 0.1,
        use_lstm: false,  // ARIMA is faster
        arima_order: (2, 1, 1),  // AR(2), I(1), MA(1)
        include_uncertainty: true,
        ..Default::default()
    };

    let mut forecaster = AdvancedTrajectoryForecaster::new(config)?;

    // Create historical trajectory (moving obstacle)
    println!("\n  Creating obstacle history (20 timesteps)...");
    let mut history = Vec::new();
    for i in 0..20 {
        let t = i as f64 * 0.1;
        // Obstacle moving with slight acceleration
        let x = t + 0.05 * t * t;
        let y = 0.5 * t;

        history.push(TrajectoryPoint {
            time: t,
            position: Array1::from_vec(vec![x, y]),
            velocity: Array1::from_vec(vec![1.0 + 0.1 * t, 0.5]),
        });
    }

    println!("  Historical trajectory: {} points from t=0.0 to t={:.1}s",
             history.len(), history.last().unwrap().time);

    // Forecast future trajectory
    println!("\n  Forecasting {} seconds ahead...", 3.0);
    let forecast = forecaster.forecast_obstacle_trajectory(&history)?;

    println!("  ✓ Forecast complete: {} future timesteps", forecast.points.len());
    println!("\n  Predicted positions:");
    for (i, point) in forecast.points.iter().take(5).enumerate() {
        println!("    t={:.2}s: pos=[{:.3}, {:.3}] ± {:.3}",
                 point.time, point.position[0], point.position[1],
                 forecast.uncertainty[i]);
    }

    println!("\n  Uncertainty growth:");
    println!("    Initial: {:.4}", forecast.uncertainty[0]);
    println!("    Final:   {:.4}", forecast.uncertainty.last().unwrap());

    Ok(())
}

fn demo_environment_dynamics() -> Result<()> {
    println!("\n\nDemo 2: Environment Dynamics Forecasting");
    println!("---------------------------------------");

    let config = TrajectoryForecastConfig {
        horizon: 2.0,
        dt: 0.2,
        use_lstm: false,
        ..Default::default()
    };

    let mut forecaster = AdvancedTrajectoryForecaster::new(config)?;

    // Create historical environment snapshots
    println!("\n  Creating environment history (15 snapshots, 2 obstacles)...");
    let mut snapshots = Vec::new();
    for i in 0..15 {
        let t = i as f64 * 0.1;

        // Obstacle 1: moving right
        let obstacle1 = ObstacleModel::new_dynamic(
            Array1::from_vec(vec![t, 1.0]),
            Array1::from_vec(vec![1.0, 0.0]),
            0.1,
        );

        // Obstacle 2: moving diagonally
        let obstacle2 = ObstacleModel::new_dynamic(
            Array1::from_vec(vec![t * 0.5, t * 0.5]),
            Array1::from_vec(vec![0.5, 0.5]),
            0.1,
        );

        snapshots.push(EnvironmentSnapshot {
            timestamp: t,
            obstacles: vec![obstacle1, obstacle2],
        });
    }

    println!("  Historical snapshots: {} from t=0.0 to t={:.1}s",
             snapshots.len(), snapshots.last().unwrap().timestamp);

    // Forecast environment dynamics
    println!("\n  Forecasting environment dynamics...");
    let forecasted = forecaster.forecast_environment_dynamics(&snapshots, 2.0)?;

    println!("  ✓ Environment forecast complete: {} future snapshots", forecasted.len());
    println!("\n  Predicted environment states:");
    for (i, snapshot) in forecasted.iter().take(3).enumerate() {
        println!("    t={:.2}s: {} obstacles", snapshot.timestamp, snapshot.obstacles.len());
        for (j, obs) in snapshot.obstacles.iter().enumerate() {
            println!("      Obstacle {}: pos=[{:.3}, {:.3}]",
                     j + 1, obs.position[0], obs.position[1]);
        }
    }

    Ok(())
}

fn demo_multi_agent() -> Result<()> {
    println!("\n\nDemo 3: Multi-Agent Trajectory Forecasting");
    println!("-----------------------------------------");

    let config = TrajectoryForecastConfig::default();
    let mut forecaster = AdvancedTrajectoryForecaster::new(config)?;

    // Create multiple agents
    println!("\n  Creating multi-agent scenario (3 agents)...");
    let agents = vec![
        AgentState {
            id: 0,
            position: Array1::from_vec(vec![0.0, 0.0]),
            velocity: Array1::from_vec(vec![1.0, 0.0]),
        },
        AgentState {
            id: 1,
            position: Array1::from_vec(vec![10.0, 0.0]),
            velocity: Array1::from_vec(vec![-1.0, 0.0]),
        },
        AgentState {
            id: 2,
            position: Array1::from_vec(vec![5.0, 5.0]),
            velocity: Array1::from_vec(vec![0.0, -1.0]),
        },
    ];

    println!("  Agent 0: pos=[0.0, 0.0], moving right");
    println!("  Agent 1: pos=[10.0, 0.0], moving left");
    println!("  Agent 2: pos=[5.0, 5.0], moving down");

    // Forecast multi-agent trajectories
    println!("\n  Forecasting multi-agent trajectories (5.0s horizon)...");
    let interactions = vec![];  // No historical interactions for this demo
    let trajectories = forecaster.forecast_multi_agent(&agents, &interactions, 5.0)?;

    println!("  ✓ Multi-agent forecast complete: {} trajectories", trajectories.len());
    println!("\n  Predicted agent positions:");

    for (agent_idx, trajectory) in trajectories.iter().enumerate() {
        println!("\n    Agent {}:", agent_idx);
        for (i, point) in trajectory.points.iter().step_by(10).take(3).enumerate() {
            println!("      t={:.1}s: pos=[{:.3}, {:.3}] ± {:.3}",
                     point.time, point.position[0], point.position[1],
                     trajectory.uncertainty[i * 10]);
        }
    }

    // Check for potential collisions
    println!("\n  Collision detection:");
    let collision_threshold = 1.0;
    let mut collisions_detected = 0;

    for t_idx in 0..trajectories[0].points.len() {
        for i in 0..trajectories.len() {
            for j in (i + 1)..trajectories.len() {
                let pos_i = &trajectories[i].points[t_idx].position;
                let pos_j = &trajectories[j].points[t_idx].position;

                let distance = ((pos_i[0] - pos_j[0]).powi(2) +
                               (pos_i[1] - pos_j[1]).powi(2)).sqrt();

                if distance < collision_threshold {
                    let t = trajectories[i].points[t_idx].time;
                    println!("    ⚠ Potential collision at t={:.2}s: Agent {} and Agent {}",
                             t, i, j);
                    println!("      Distance: {:.3}m (threshold: {}m)", distance, collision_threshold);
                    collisions_detected += 1;
                    break;
                }
            }
        }
    }

    if collisions_detected == 0 {
        println!("    ✓ No collisions detected");
    } else {
        println!("    Found {} potential collision events", collisions_detected);
    }

    Ok(())
}
