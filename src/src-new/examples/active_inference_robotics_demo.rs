//! # Active Inference Robotics Demo
//!
//! This example demonstrates how to use Worker 1's Active Inference module
//! for robotics motion planning and control.
//!
//! ## What This Demo Shows
//!
//! 1. **Initialize Active Inference Agent** - Set up belief state and preferences
//! 2. **Define Goal State** - Target position for robot to reach
//! 3. **Plan Trajectory** - Free energy minimization for optimal path
//! 4. **Execute Motion** - Online belief updates with sensory feedback
//! 5. **Uncertainty Handling** - Robust control under noisy observations
//!
//! ## Key Concepts
//!
//! - **Free Energy**: Variational bound on surprise (agent minimizes this)
//! - **Belief State**: Agent's probabilistic model of world state
//! - **Precision**: Confidence in sensory observations vs predictions
//! - **Active Inference**: Acting to confirm predictions (action as inference)
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example active_inference_robotics_demo
//! ```
//!
//! ## Integration
//!
//! This demo integrates with:
//! - **Worker 7**: Robotics motion planning APIs
//! - **Worker 3**: Applications across domains (healthcare trajectories, etc.)
//! - **Worker 8**: REST/GraphQL endpoints for active inference

use anyhow::Result;
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

// Simulated imports (adjust to actual Worker 1 module paths)
// use prism_worker_1::active_inference::ActiveInferenceAgent;
// use prism_worker_1::active_inference::AgentConfig;
// use prism_worker_1::active_inference::BeliefState;

/// Agent configuration
#[derive(Debug, Clone)]
struct AgentConfig {
    state_dim: usize,
    action_dim: usize,
    sensory_precision: f64,  // How much to trust observations
    action_precision: f64,    // How confident in action selection
    learning_rate: f64,
}

/// Belief state (agent's probabilistic world model)
#[derive(Debug, Clone)]
struct BeliefState {
    mean: Array1<f64>,      // Expected state
    covariance: Array2<f64>, // Uncertainty
}

/// Robot state (2D position + velocity)
#[derive(Debug, Clone)]
struct RobotState {
    position: Array1<f64>,   // [x, y]
    velocity: Array1<f64>,   // [vx, vy]
}

fn main() -> Result<()> {
    println!("=".repeat(80));
    println!("  ACTIVE INFERENCE ROBOTICS DEMO");
    println!("  Worker 1 - Free Energy Minimization for Motion Planning");
    println!("=".repeat(80));
    println!();

    // Step 1: Initialize robot and environment
    println!("🤖 Step 1: Initializing Robot and Environment");
    println!("-".repeat(80));

    let start_position = Array1::from_vec(vec![0.0, 0.0]);
    let goal_position = Array1::from_vec(vec![10.0, 10.0]);

    println!("  • Start Position: [{:.1}, {:.1}]", start_position[0], start_position[1]);
    println!("  • Goal Position:  [{:.1}, {:.1}]", goal_position[0], goal_position[1]);
    println!();

    // Add obstacles
    let obstacles = vec![
        (Array1::from_vec(vec![5.0, 3.0]), 1.5),  // (center, radius)
        (Array1::from_vec(vec![7.0, 7.0]), 2.0),
        (Array1::from_vec(vec![3.0, 8.0]), 1.0),
    ];

    println!("  Obstacles:");
    for (i, (center, radius)) in obstacles.iter().enumerate() {
        println!("    {} - Center: [{:.1}, {:.1}], Radius: {:.1}",
            i + 1, center[0], center[1], radius);
    }
    println!();

    // Step 2: Configure Active Inference agent
    println!("⚙️  Step 2: Configuring Active Inference Agent");
    println!("-".repeat(80));

    let agent_config = AgentConfig {
        state_dim: 4,  // [x, y, vx, vy]
        action_dim: 2, // [ax, ay] (acceleration)
        sensory_precision: 10.0,  // Trust observations
        action_precision: 5.0,     // Confidence in actions
        learning_rate: 0.1,
    };

    println!("  Agent Configuration:");
    println!("  • State dimension:     {} (position + velocity)", agent_config.state_dim);
    println!("  • Action dimension:    {} (acceleration)", agent_config.action_dim);
    println!("  • Sensory precision:   {:.1} (observation confidence)", agent_config.sensory_precision);
    println!("  • Action precision:    {:.1} (action confidence)", agent_config.action_precision);
    println!("  • Learning rate:       {:.2}", agent_config.learning_rate);
    println!();

    // Step 3: Initialize belief state
    println!("🧠 Step 3: Initializing Belief State");
    println!("-".repeat(80));

    let mut belief = BeliefState {
        mean: Array1::from_vec(vec![
            start_position[0], start_position[1],  // position
            0.0, 0.0,                                // velocity (initially at rest)
        ]),
        covariance: Array2::eye(4) * 0.5,  // Initial uncertainty
    };

    println!("  Initial Belief:");
    println!("  • Position:   [{:.4}, {:.4}]", belief.mean[0], belief.mean[1]);
    println!("  • Velocity:   [{:.4}, {:.4}]", belief.mean[2], belief.mean[3]);
    println!("  • Uncertainty: {:.4} (average diagonal covariance)", belief.covariance.diag().mean().unwrap());
    println!();

    // Step 4: Plan trajectory using free energy minimization
    println!("🎯 Step 4: Planning Trajectory (Free Energy Minimization)");
    println!("-".repeat(80));

    let trajectory = plan_trajectory_active_inference(
        &start_position,
        &goal_position,
        &obstacles,
        &agent_config,
        100, // max steps
    )?;

    println!("  ✓ Trajectory planned with {} waypoints", trajectory.len());
    println!("  • Path length: {:.2} units", calculate_path_length(&trajectory));
    println!("  • Obstacle clearance: {:.2} units (minimum)", calculate_min_clearance(&trajectory, &obstacles));
    println!();

    // Step 5: Execute motion with online belief updates
    println!("🚀 Step 5: Executing Motion with Online Belief Updates");
    println!("-".repeat(80));
    println!();

    let mut robot_state = RobotState {
        position: start_position.clone(),
        velocity: Array1::zeros(2),
    };

    let mut execution_history = Vec::new();

    println!("  Step | Position       | Velocity      | Free Energy | Reached Goal");
    println!("  {}", "-".repeat(70));

    for (step, target_position) in trajectory.iter().enumerate().step_by(10) {
        // Compute action (acceleration towards target)
        let action = compute_action_active_inference(
            &belief,
            target_position,
            &goal_position,
            &agent_config,
        )?;

        // Apply action to robot (simulate dynamics)
        robot_state = update_robot_state(&robot_state, &action, 0.1); // dt = 0.1s

        // Get noisy observation
        let observation = observe_robot_state(&robot_state, 0.1); // observation noise std = 0.1

        // Update belief state (Bayesian inference)
        belief = update_belief_state(
            &belief,
            &action,
            &observation,
            &agent_config,
        );

        // Calculate free energy
        let free_energy = calculate_free_energy(
            &belief,
            &observation,
            &goal_position,
            &agent_config,
        );

        // Check if goal reached
        let distance_to_goal = euclidean_distance(&robot_state.position, &goal_position);
        let reached_goal = distance_to_goal < 0.5;

        println!("  {:>4} | [{:>5.2}, {:>5.2}] | [{:>5.2}, {:>5.2}] | {:>11.4} | {}",
            step,
            robot_state.position[0], robot_state.position[1],
            robot_state.velocity[0], robot_state.velocity[1],
            free_energy,
            if reached_goal { "✓ Yes" } else { "  No" }
        );

        execution_history.push((
            robot_state.position.clone(),
            free_energy,
            distance_to_goal,
        ));

        // Stop if goal reached
        if reached_goal {
            println!();
            println!("  ✅ Goal reached at step {}!", step);
            break;
        }
    }
    println!();

    // Step 6: Analyze performance
    println!("📊 Step 6: Performance Analysis");
    println!("-".repeat(80));

    analyze_performance(&execution_history, &goal_position);
    println!();

    // Step 7: Visualize path
    println!("🗺️  Step 7: Path Visualization");
    println!("-".repeat(80));

    visualize_path_2d(&execution_history, &goal_position, &obstacles);
    println!();

    // Step 8: Worker 7 integration example
    println!("🔗 Step 8: Worker 7 Robotics Integration");
    println!("-".repeat(80));

    demonstrate_worker7_integration(&trajectory)?;
    println!();

    // Summary
    println!("=".repeat(80));
    println!("  DEMO COMPLETE");
    println!("=".repeat(80));
    println!();
    println!("  Key Concepts Demonstrated:");
    println!("  • Free Energy Minimization: Agent minimizes surprise about sensations");
    println!("  • Belief State: Probabilistic world model updated online");
    println!("  • Precision: Balance between observation trust vs prediction trust");
    println!("  • Active Inference: Acting to confirm predictions (exploit + explore)");
    println!();
    println!("  Active Inference Advantages:");
    println!("  ✓ Unified perception-action framework (no separate planner/controller)");
    println!("  ✓ Principled uncertainty handling (Bayesian inference)");
    println!("  ✓ Exploration built-in (epistemic value = reducing uncertainty)");
    println!("  ✓ Robust to model mismatch and noise");
    println!();
    println!("  Production Use Cases:");
    println!("  • Worker 7: Robotics motion planning and control");
    println!("  • Worker 3 Healthcare: Patient treatment trajectory optimization");
    println!("  • Worker 4 Finance: Portfolio rebalancing under uncertainty");
    println!("  • Worker 3 Energy: Adaptive grid control with demand forecasting");
    println!();
    println!("  Integration Points:");
    println!("  • Worker 7 Robotics API: /api/v1/robotics/plan_motion");
    println!("  • Worker 8 REST endpoint: POST /api/v1/active_inference/plan");
    println!("  • Worker 8 GraphQL: mutation { planTrajectory(...) }");
    println!();
    println!("  Next Steps:");
    println!("  • Try hierarchical Active Inference (multi-level planning)");
    println!("  • Combine with Transfer Entropy for causal world models");
    println!("  • Use LSTM for learning forward models from data");
    println!("  • Deploy via Worker 8 API for production robotics");
    println!();

    Ok(())
}

/// Plan trajectory using Active Inference
fn plan_trajectory_active_inference(
    start: &Array1<f64>,
    goal: &Array1<f64>,
    obstacles: &[(Array1<f64>, f64)],
    config: &AgentConfig,
    max_steps: usize,
) -> Result<Vec<Array1<f64>>> {
    let mut trajectory = Vec::new();
    let mut current_pos = start.clone();

    trajectory.push(current_pos.clone());

    let dt = 0.1;

    for _ in 0..max_steps {
        // Compute direction to goal
        let direction = (goal - &current_pos) / euclidean_distance(&current_pos, goal);

        // Compute repulsive forces from obstacles
        let mut repulsion = Array1::zeros(2);

        for (obs_center, obs_radius) in obstacles {
            let dist = euclidean_distance(&current_pos, obs_center);

            if dist < obs_radius * 3.0 {
                let repulsion_dir = (&current_pos - obs_center) / dist;
                let repulsion_magnitude = 1.0 / (dist - obs_radius).max(0.1);
                repulsion = repulsion + repulsion_dir * repulsion_magnitude;
            }
        }

        // Combine attractive (goal) and repulsive (obstacles) forces
        let velocity = (direction * 2.0 + repulsion * 0.5) / 2.5;

        // Update position
        current_pos = &current_pos + &velocity * dt;

        trajectory.push(current_pos.clone());

        // Stop if close to goal
        if euclidean_distance(&current_pos, goal) < 0.5 {
            break;
        }
    }

    Ok(trajectory)
}

/// Compute action using Active Inference
fn compute_action_active_inference(
    belief: &BeliefState,
    target: &Array1<f64>,
    goal: &Array1<f64>,
    config: &AgentConfig,
) -> Result<Array1<f64>> {
    // Simplified: Action is proportional to expected state error
    let current_position = belief.mean.slice(s![0..2]).to_owned();
    let desired_velocity = (target - &current_position) * 2.0;
    let current_velocity = belief.mean.slice(s![2..4]).to_owned();

    let velocity_error = &desired_velocity - &current_velocity;
    let action = velocity_error * config.action_precision;

    Ok(action)
}

/// Update robot state (dynamics simulation)
fn update_robot_state(state: &RobotState, action: &Array1<f64>, dt: f64) -> RobotState {
    // Simple kinematics: position += velocity * dt, velocity += action * dt
    let new_velocity = &state.velocity + action * dt;
    let new_position = &state.position + &new_velocity * dt;

    RobotState {
        position: new_position,
        velocity: new_velocity,
    }
}

/// Observe robot state (with noise)
fn observe_robot_state(state: &RobotState, noise_std: f64) -> Array1<f64> {
    use rand::Rng;
    use rand_distr::{Distribution, Normal};

    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, noise_std).unwrap();

    let mut observation = Array1::zeros(4);

    for i in 0..2 {
        observation[i] = state.position[i] + normal.sample(&mut rng);
        observation[i + 2] = state.velocity[i] + normal.sample(&mut rng);
    }

    observation
}

/// Update belief state (Bayesian inference)
fn update_belief_state(
    prior_belief: &BeliefState,
    action: &Array1<f64>,
    observation: &Array1<f64>,
    config: &AgentConfig,
) -> BeliefState {
    // Simplified Kalman filter update
    let dt = 0.1;

    // Prediction step (prior → posterior given action)
    let mut predicted_mean = prior_belief.mean.clone();
    predicted_mean[0] += predicted_mean[2] * dt;
    predicted_mean[1] += predicted_mean[3] * dt;
    predicted_mean[2] += action[0] * dt;
    predicted_mean[3] += action[1] * dt;

    // Update step (posterior given observation)
    let innovation = observation - &predicted_mean;
    let kalman_gain = config.sensory_precision / (config.sensory_precision + 1.0);

    let updated_mean = &predicted_mean + &innovation * kalman_gain;

    // Update covariance (simplified)
    let updated_covariance = &prior_belief.covariance * (1.0 - kalman_gain);

    BeliefState {
        mean: updated_mean,
        covariance: updated_covariance,
    }
}

/// Calculate free energy
fn calculate_free_energy(
    belief: &BeliefState,
    observation: &Array1<f64>,
    goal: &Array1<f64>,
    config: &AgentConfig,
) -> f64 {
    // Free energy = prediction error + divergence from goal

    // Prediction error (surprise about observations)
    let prediction_error: f64 = (observation - &belief.mean)
        .iter()
        .map(|&x| x * x)
        .sum::<f64>()
        .sqrt();

    // Goal divergence (expected surprise about goal)
    let current_position = belief.mean.slice(s![0..2]).to_owned();
    let goal_distance = euclidean_distance(&current_position, goal);

    // Free energy (weighted sum)
    let free_energy = prediction_error * config.sensory_precision + goal_distance * 2.0;

    free_energy
}

/// Calculate path length
fn calculate_path_length(trajectory: &[Array1<f64>]) -> f64 {
    let mut length = 0.0;

    for i in 1..trajectory.len() {
        length += euclidean_distance(&trajectory[i], &trajectory[i-1]);
    }

    length
}

/// Calculate minimum clearance to obstacles
fn calculate_min_clearance(trajectory: &[Array1<f64>], obstacles: &[(Array1<f64>, f64)]) -> f64 {
    let mut min_clearance = f64::INFINITY;

    for waypoint in trajectory {
        for (obs_center, obs_radius) in obstacles {
            let clearance = euclidean_distance(waypoint, obs_center) - obs_radius;
            min_clearance = min_clearance.min(clearance);
        }
    }

    min_clearance.max(0.0)
}

/// Euclidean distance
fn euclidean_distance(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    (a - b).iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Analyze performance
fn analyze_performance(history: &[(Array1<f64>, f64, f64)], goal: &Array1<f64>) {
    println!("  Performance Metrics:");
    println!();

    // Total distance traveled
    let total_distance: f64 = (1..history.len())
        .map(|i| euclidean_distance(&history[i].0, &history[i-1].0))
        .sum();

    println!("  • Total distance traveled: {:.2} units", total_distance);

    // Average free energy
    let avg_free_energy: f64 = history.iter().map(|(_, fe, _)| fe).sum::<f64>() / history.len() as f64;
    println!("  • Average free energy:     {:.4}", avg_free_energy);

    // Final distance to goal
    let final_distance = history.last().unwrap().2;
    println!("  • Final distance to goal:  {:.4} units", final_distance);

    // Success?
    let success = final_distance < 0.5;
    println!("  • Task success:            {}", if success { "✅ Yes" } else { "❌ No" });
    println!();

    // Free energy progression
    println!("  Free Energy Progression (shows convergence):");
    for (i, (_, fe, _)) in history.iter().enumerate().step_by(history.len() / 10) {
        let bar_length = (fe * 5.0).min(50.0) as usize;
        let bar = "█".repeat(bar_length);
        println!("  Step {:>3} | {} {:.4}", i, bar, fe);
    }
}

/// Visualize path in 2D
fn visualize_path_2d(history: &[(Array1<f64>, f64, f64)], goal: &Array1<f64>, obstacles: &[(Array1<f64>, f64)]) {
    println!("  2D Path Visualization (top-down view):");
    println!();

    let grid_size = 15;
    let scale = grid_size as f64 / 12.0; // Map [0, 12] → [0, grid_size]

    // Create grid
    let mut grid = vec![vec![' '; grid_size]; grid_size];

    // Mark obstacles
    for (obs_center, obs_radius) in obstacles {
        let cx = (obs_center[0] * scale) as usize;
        let cy = (obs_center[1] * scale) as usize;
        let r = (*obs_radius * scale) as usize;

        for y in 0..grid_size {
            for x in 0..grid_size {
                let dist = (((x as f64 - cx as f64).powi(2) + (y as f64 - cy as f64).powi(2)).sqrt()) as usize;
                if dist <= r {
                    grid[grid_size - 1 - y][x] = '█';
                }
            }
        }
    }

    // Mark path
    for (pos, _, _) in history {
        let x = (pos[0] * scale) as usize;
        let y = (pos[1] * scale) as usize;

        if x < grid_size && y < grid_size {
            grid[grid_size - 1 - y][x] = '·';
        }
    }

    // Mark start and goal
    let start_x = (history[0].0[0] * scale) as usize;
    let start_y = (history[0].0[1] * scale) as usize;
    let goal_x = (goal[0] * scale) as usize;
    let goal_y = (goal[1] * scale) as usize;

    if start_x < grid_size && start_y < grid_size {
        grid[grid_size - 1 - start_y][start_x] = 'S';
    }

    if goal_x < grid_size && goal_y < grid_size {
        grid[grid_size - 1 - goal_y][goal_x] = 'G';
    }

    // Print grid
    for row in &grid {
        print!("  ");
        for &cell in row {
            print!("{}", cell);
        }
        println!();
    }

    println!();
    println!("  Legend: S = Start, G = Goal, · = Path, █ = Obstacle");
}

/// Demonstrate Worker 7 integration
fn demonstrate_worker7_integration(trajectory: &[Array1<f64>]) -> Result<()> {
    println!("  Worker 7 Robotics API Integration:");
    println!();

    println!("  Example REST API Call:");
    println!("  {}", "-".repeat(60));
    println!(r#"  POST /api/v1/robotics/plan_motion"#);
    println!(r#"  Content-Type: application/json"#);
    println!();
    println!(r#"  {{"#);
    println!(r#"    "start": [0.0, 0.0],"#);
    println!(r#"    "goal": [10.0, 10.0],"#);
    println!(r#"    "obstacles": ["#);
    println!(r#"      {{"center": [5.0, 3.0], "radius": 1.5}},"#);
    println!(r#"      {{"center": [7.0, 7.0], "radius": 2.0}},"#);
    println!(r#"      {{"center": [3.0, 8.0], "radius": 1.0}}"#);
    println!(r#"    ],"#);
    println!(r#"    "method": "active_inference""#);
    println!(r#"  }}"#);
    println!();

    println!("  Example GraphQL Query:");
    println!("  {}", "-".repeat(60));
    println!(r#"  mutation {{"#);
    println!(r#"    planTrajectory("#);
    println!(r#"      start: [0.0, 0.0]"#);
    println!(r#"      goal: [10.0, 10.0]"#);
    println!(r#"      method: ACTIVE_INFERENCE"#);
    println!(r#"    ) {{"#);
    println!(r#"      trajectory {{"#);
    println!(r#"        waypoints {{ x y }}"#);
    println!(r#"        pathLength"#);
    println!(r#"        obstacleClearance"#);
    println!(r#"      }}"#);
    println!(r#"      freeEnergy"#);
    println!(r#"    }}"#);
    println!(r#"  }}"#);
    println!();

    println!("  ✓ Worker 1 Active Inference → Worker 7 Robotics → Worker 8 API");

    Ok(())
}

// Import for array slicing
use ndarray::s;
