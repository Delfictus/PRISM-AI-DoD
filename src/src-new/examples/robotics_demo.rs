//! Robotics Motion Planning Demo
//!
//! Demonstrates Worker 7's robotics module:
//! - Motion planning with Active Inference
//! - Obstacle avoidance using artificial potential fields
//! - Environment modeling
//! - ROS integration (optional)
//!
//! Run with: cargo run --example robotics_demo

use anyhow::Result;
use ndarray::{array, Array1};
use prism_ai::applications::{RoboticsController, RoboticsConfig};
use prism_ai::applications::robotics::{RobotState, Goal, Obstacle};
use prism_ai::active_inference::GenerativeModel;

fn main() -> Result<()> {
    println!("=== PRISM-AI Robotics Demo ===\n");

    // 1. Create robotics configuration
    let config = RoboticsConfig {
        max_velocity: 1.0,          // m/s
        max_acceleration: 0.5,      // m/s²
        planning_horizon: 5.0,      // seconds
        control_frequency: 10.0,    // Hz
        obstacle_radius: 0.5,       // meters
        goal_tolerance: 0.1,        // meters
        use_ros: false,             // Set true for ROS integration
    };

    println!("Configuration:");
    println!("  Max velocity: {} m/s", config.max_velocity);
    println!("  Planning horizon: {} seconds", config.planning_horizon);
    println!("  Obstacle avoidance radius: {} meters\n", config.obstacle_radius);

    // 2. Initialize robotics controller with Active Inference
    let mut controller = RoboticsController::new(config)?;
    println!("✓ Robotics controller initialized\n");

    // 3. Define robot initial state
    let robot_state = RobotState {
        position: array![0.0, 0.0, 0.0],     // Start at origin
        velocity: array![0.0, 0.0, 0.0],     // Initially stationary
        orientation: array![1.0, 0.0, 0.0, 0.0], // Quaternion (no rotation)
    };

    println!("Robot initial state:");
    println!("  Position: [{:.1}, {:.1}, {:.1}]",
        robot_state.position[0],
        robot_state.position[1],
        robot_state.position[2]
    );

    // 4. Define goal
    let goal = Goal {
        position: array![10.0, 10.0, 0.0],   // Target position
        tolerance: 0.1,                       // 10cm tolerance
    };

    println!("  Goal: [{:.1}, {:.1}, {:.1}]",
        goal.position[0],
        goal.position[1],
        goal.position[2]
    );

    // 5. Add obstacles to environment
    let obstacles = vec![
        Obstacle {
            position: array![5.0, 5.0, 0.0],
            radius: 1.0,
        },
        Obstacle {
            position: array![7.0, 8.0, 0.0],
            radius: 0.8,
        },
        Obstacle {
            position: array![3.0, 7.0, 0.0],
            radius: 1.2,
        },
    ];

    println!("\nEnvironment obstacles:");
    for (i, obs) in obstacles.iter().enumerate() {
        println!("  Obstacle {}: pos=[{:.1}, {:.1}, {:.1}], radius={:.1}m",
            i + 1,
            obs.position[0],
            obs.position[1],
            obs.position[2],
            obs.radius
        );
    }

    // 6. Update environment model
    controller.update_environment(&obstacles)?;
    println!("\n✓ Environment model updated with {} obstacles", obstacles.len());

    // 7. Plan motion using Active Inference
    println!("\n--- Planning Motion with Active Inference ---");
    let motion_plan = controller.plan_motion(&robot_state, &goal)?;

    println!("\nMotion plan generated:");
    println!("  Waypoints: {}", motion_plan.waypoints.len());
    println!("  Total distance: {:.2} meters", motion_plan.total_distance);
    println!("  Estimated time: {:.2} seconds", motion_plan.estimated_time);
    println!("  Free energy: {:.4}", motion_plan.free_energy);

    // 8. Display waypoints
    println!("\nPlanned waypoints:");
    for (i, waypoint) in motion_plan.waypoints.iter().take(5).enumerate() {
        println!("  Waypoint {}: [{:.2}, {:.2}, {:.2}]",
            i + 1,
            waypoint[0],
            waypoint[1],
            waypoint[2]
        );
    }
    if motion_plan.waypoints.len() > 5 {
        println!("  ... ({} more waypoints)", motion_plan.waypoints.len() - 5);
    }

    // 9. Simulate motion execution
    println!("\n--- Simulating Motion Execution ---");
    let mut current_state = robot_state.clone();
    let dt = 0.1; // 100ms timestep

    for (i, waypoint) in motion_plan.waypoints.iter().enumerate().step_by(10) {
        // Compute control action
        let control = controller.compute_control(&current_state, waypoint)?;

        // Update state (simplified dynamics)
        current_state.position = current_state.position.clone() + &control * dt;

        // Check distance to goal
        let distance_to_goal = (&goal.position - &current_state.position)
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();

        if i % 50 == 0 {
            println!("  Step {}: pos=[{:.2}, {:.2}, {:.2}], dist_to_goal={:.2}m",
                i,
                current_state.position[0],
                current_state.position[1],
                current_state.position[2],
                distance_to_goal
            );
        }

        if distance_to_goal < goal.tolerance {
            println!("\n✓ Goal reached at step {}!", i);
            break;
        }
    }

    // 10. Show final state
    let final_distance = (&goal.position - &current_state.position)
        .iter()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt();

    println!("\nFinal state:");
    println!("  Position: [{:.2}, {:.2}, {:.2}]",
        current_state.position[0],
        current_state.position[1],
        current_state.position[2]
    );
    println!("  Distance to goal: {:.2} meters", final_distance);

    if final_distance < goal.tolerance {
        println!("\n✓ SUCCESS: Robot reached goal!");
    } else {
        println!("\n⚠ Robot did not reach goal (within tolerance)");
    }

    // 11. Active Inference insights
    println!("\n--- Active Inference Analysis ---");
    println!("Free Energy Minimization:");
    println!("  The planner minimized variational free energy to find");
    println!("  the path that best balances:");
    println!("  - Goal achievement (minimize expected surprise)");
    println!("  - Obstacle avoidance (minimize collision risk)");
    println!("  - Smooth motion (minimize control effort)");
    println!("\nFinal free energy: {:.4}", motion_plan.free_energy);

    println!("\n=== Demo Complete ===");

    Ok(())
}
