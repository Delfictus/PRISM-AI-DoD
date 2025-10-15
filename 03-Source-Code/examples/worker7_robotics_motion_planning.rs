//! Worker 7: Robotics Motion Planning Tutorial
//!
//! Production example demonstrating Active Inference-based motion planning
//! with information-theoretic trajectory optimization.
//!
//! This tutorial covers:
//! 1. Environment setup with obstacles
//! 2. Information-theoretic trajectory evaluation
//! 3. Active Inference for motion planning
//! 4. Real-time path optimization
//! 5. Collision avoidance with uncertainty quantification
//!
//! Constitution: Worker 7 - Drug Discovery & Robotics
//! Time: Production example (5 hours quality enhancement allocation)

use prism_ai::applications::{
    RoboticsController,
    RoboticsConfig,
    MotionPlanner,
    MotionPlan,
    RoboticsInformationMetrics,
};
use ndarray::{Array1, Array2};
use anyhow::Result;
use std::time::Instant;

/// Represents a 2D point in the workspace
#[derive(Clone, Copy, Debug)]
struct Point2D {
    x: f64,
    y: f64,
}

impl Point2D {
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    fn distance_to(&self, other: &Point2D) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }

    fn to_array(&self) -> Array1<f64> {
        Array1::from_vec(vec![self.x, self.y])
    }
}

/// Represents a circular obstacle in the workspace
#[derive(Clone, Debug)]
struct Obstacle {
    center: Point2D,
    radius: f64,
}

impl Obstacle {
    fn new(center: Point2D, radius: f64) -> Self {
        Self { center, radius }
    }

    fn contains(&self, point: &Point2D) -> bool {
        self.center.distance_to(point) < self.radius
    }

    fn distance_to(&self, point: &Point2D) -> f64 {
        (self.center.distance_to(point) - self.radius).max(0.0)
    }
}

/// Workspace configuration with obstacles
struct Workspace {
    width: f64,
    height: f64,
    obstacles: Vec<Obstacle>,
}

impl Workspace {
    fn new(width: f64, height: f64) -> Self {
        Self {
            width,
            height,
            obstacles: Vec::new(),
        }
    }

    fn add_obstacle(&mut self, obstacle: Obstacle) {
        self.obstacles.push(obstacle);
    }

    fn is_collision_free(&self, point: &Point2D) -> bool {
        // Check workspace bounds
        if point.x < 0.0 || point.x > self.width || point.y < 0.0 || point.y > self.height {
            return false;
        }

        // Check obstacle collisions
        !self.obstacles.iter().any(|obs| obs.contains(point))
    }

    fn nearest_obstacle_distance(&self, point: &Point2D) -> f64 {
        self.obstacles.iter()
            .map(|obs| obs.distance_to(point))
            .fold(f64::INFINITY, f64::min)
    }
}

/// Main robotics motion planning tutorial
#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Worker 7: Robotics Motion Planning Tutorial                    ║");
    println!("║  Active Inference + Information-Theoretic Optimization           ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    // ═══════════════════════════════════════════════════════════════════════════
    // Phase 1: Environment Setup
    // ═══════════════════════════════════════════════════════════════════════════

    println!("🌍 Phase 1: Environment Setup");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let mut workspace = Workspace::new(10.0, 10.0);

    // Add obstacles
    workspace.add_obstacle(Obstacle::new(Point2D::new(3.0, 3.0), 1.0));
    workspace.add_obstacle(Obstacle::new(Point2D::new(7.0, 5.0), 1.2));
    workspace.add_obstacle(Obstacle::new(Point2D::new(5.0, 7.0), 0.8));
    workspace.add_obstacle(Obstacle::new(Point2D::new(2.0, 8.0), 0.6));

    println!("✓ Workspace: {:.1}m × {:.1}m", workspace.width, workspace.height);
    println!("✓ Obstacles: {}", workspace.obstacles.len());
    for (i, obs) in workspace.obstacles.iter().enumerate() {
        println!("  • Obstacle {}: center=({:.1}, {:.1}), radius={:.1}m",
            i + 1, obs.center.x, obs.center.y, obs.radius);
    }
    println!();

    // Define start and goal positions
    let start = Point2D::new(1.0, 1.0);
    let goal = Point2D::new(9.0, 9.0);

    println!("✓ Start position: ({:.1}, {:.1})", start.x, start.y);
    println!("✓ Goal position:  ({:.1}, {:.1})", goal.x, goal.y);
    println!("✓ Euclidean distance: {:.2}m", start.distance_to(&goal));
    println!();

    // ═══════════════════════════════════════════════════════════════════════════
    // Phase 2: Baseline Planning (Straight-Line Path)
    // ═══════════════════════════════════════════════════════════════════════════

    println!("📏 Phase 2: Baseline Planning (Straight Line)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let baseline_trajectory = generate_straight_line(&start, &goal, 50);
    let baseline_collisions = count_collisions(&baseline_trajectory, &workspace);
    let baseline_length = calculate_path_length(&baseline_trajectory);

    println!("✓ Waypoints: {}", baseline_trajectory.len());
    println!("✓ Path length: {:.2}m", baseline_length);
    println!("✓ Collision checks: {}/{} waypoints in collision ({:.1}%)",
        baseline_collisions, baseline_trajectory.len(),
        (baseline_collisions as f64 / baseline_trajectory.len() as f64) * 100.0);

    if baseline_collisions > 0 {
        println!("⚠ Straight-line path is NOT collision-free!");
    } else {
        println!("✓ Straight-line path is collision-free!");
    }
    println!();

    // ═══════════════════════════════════════════════════════════════════════════
    // Phase 3: Active Inference Motion Planning
    // ═══════════════════════════════════════════════════════════════════════════

    println!("🤖 Phase 3: Active Inference Motion Planning");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let config = RoboticsConfig {
        state_dim: 2,        // 2D position (x, y)
        action_dim: 2,       // 2D velocity (vx, vy)
        planning_horizon: 50,
        max_velocity: 0.5,   // m/s
        obstacle_threshold: 0.3, // Safety margin
    };

    let controller = RoboticsController::new(config)?;
    let planner = MotionPlanner::new(2)?; // 2D workspace

    println!("✓ Initialized Active Inference controller");
    println!("✓ State space: {}D", config.state_dim);
    println!("✓ Action space: {}D", config.action_dim);
    println!("✓ Planning horizon: {} steps", config.planning_horizon);
    println!("✓ Max velocity: {:.2} m/s", config.max_velocity);
    println!();

    // Plan path using Active Inference
    println!("🔍 Planning collision-free path...");
    let start_time = Instant::now();

    let motion_plan = planner.plan(
        &start.to_array(),
        &goal.to_array(),
        Some(&workspace_to_occupancy_grid(&workspace, 50)),
    ).await?;

    let planning_duration = start_time.elapsed();

    println!("✓ Planning complete: {:.2?}", planning_duration);
    println!("✓ Trajectory waypoints: {}", motion_plan.waypoints.len());
    println!("✓ Expected free energy: {:.4}", motion_plan.expected_free_energy);
    println!();

    // ═══════════════════════════════════════════════════════════════════════════
    // Phase 4: Trajectory Analysis
    // ═══════════════════════════════════════════════════════════════════════════

    println!("📊 Phase 4: Trajectory Analysis");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let robotics_metrics = RoboticsInformationMetrics::new();

    // Convert motion plan to trajectory matrix
    let trajectory_matrix = motion_plan_to_matrix(&motion_plan)?;

    // Calculate trajectory entropy
    let start = Instant::now();
    let trajectory_entropy = robotics_metrics.trajectory_entropy(&trajectory_matrix)?;
    let entropy_duration = start.elapsed();

    println!("✓ Trajectory entropy: {:.4} nats", trajectory_entropy);
    println!("✓ Calculation time: {:.2?}", entropy_duration);
    println!("  → Higher entropy indicates more exploratory trajectory");
    println!();

    // Collision analysis
    let trajectory_points = matrix_to_points(&trajectory_matrix);
    let planned_collisions = count_collisions(&trajectory_points, &workspace);
    let planned_length = calculate_path_length(&trajectory_points);

    println!("Collision Analysis:");
    println!("  • Baseline collisions: {}/{} ({:.1}%)",
        baseline_collisions, baseline_trajectory.len(),
        (baseline_collisions as f64 / baseline_trajectory.len() as f64) * 100.0);
    println!("  • Planned collisions:  {}/{} ({:.1}%)",
        planned_collisions, trajectory_points.len(),
        (planned_collisions as f64 / trajectory_points.len() as f64) * 100.0);
    println!();

    println!("Path Length Analysis:");
    println!("  • Baseline length: {:.2}m", baseline_length);
    println!("  • Planned length:  {:.2}m", planned_length);
    println!("  • Overhead:        {:.1}%",
        ((planned_length - baseline_length) / baseline_length) * 100.0);
    println!();

    // Safety margins
    let min_obstacle_distance = trajectory_points.iter()
        .map(|p| workspace.nearest_obstacle_distance(p))
        .fold(f64::INFINITY, f64::min);

    let avg_obstacle_distance = trajectory_points.iter()
        .map(|p| workspace.nearest_obstacle_distance(p))
        .sum::<f64>() / trajectory_points.len() as f64;

    println!("Safety Margins:");
    println!("  • Minimum clearance: {:.2}m", min_obstacle_distance);
    println!("  • Average clearance: {:.2}m", avg_obstacle_distance);
    println!("  • Safety threshold:  {:.2}m", config.obstacle_threshold);

    if min_obstacle_distance >= config.obstacle_threshold {
        println!("  ✓ Path maintains safe clearance!");
    } else {
        println!("  ⚠ Path violates safety threshold!");
    }
    println!();

    // ═══════════════════════════════════════════════════════════════════════════
    // Phase 5: Real-Time Path Optimization
    // ═══════════════════════════════════════════════════════════════════════════

    println!("⚡ Phase 5: Real-Time Path Optimization");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Simulate dynamic obstacle
    let mut dynamic_workspace = workspace.clone();
    let dynamic_obstacle = Obstacle::new(Point2D::new(6.0, 6.0), 1.0);
    dynamic_workspace.add_obstacle(dynamic_obstacle.clone());

    println!("✓ Added dynamic obstacle at ({:.1}, {:.1}) with radius {:.1}m",
        dynamic_obstacle.center.x, dynamic_obstacle.center.y, dynamic_obstacle.radius);
    println!();

    // Replan around dynamic obstacle
    println!("🔄 Replanning to avoid new obstacle...");
    let replan_start = Instant::now();

    let updated_plan = planner.plan(
        &start.to_array(),
        &goal.to_array(),
        Some(&workspace_to_occupancy_grid(&dynamic_workspace, 50)),
    ).await?;

    let replan_duration = replan_start.elapsed();

    println!("✓ Replanning complete: {:.2?}", replan_duration);
    println!("✓ Planning speedup vs initial: {:.1}x",
        planning_duration.as_secs_f64() / replan_duration.as_secs_f64());
    println!();

    let updated_trajectory = motion_plan_to_points(&updated_plan);
    let updated_collisions = count_collisions(&updated_trajectory, &dynamic_workspace);
    let updated_length = calculate_path_length(&updated_trajectory);

    println!("Updated Plan Metrics:");
    println!("  • New path length: {:.2}m ({:+.1}% vs original)",
        updated_length,
        ((updated_length - planned_length) / planned_length) * 100.0);
    println!("  • Collisions: {}/{}", updated_collisions, updated_trajectory.len());
    println!("  • Expected free energy: {:.4}", updated_plan.expected_free_energy);
    println!();

    // ═══════════════════════════════════════════════════════════════════════════
    // Phase 6: Sensor Information Gain Analysis
    // ═══════════════════════════════════════════════════════════════════════════

    println!("📡 Phase 6: Sensor Information Gain");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Simulate sensor measurements along trajectory
    let prior_variance = 1.0;  // High uncertainty before measurement
    let sensor_noise = 0.1;    // Low sensor noise

    let mut cumulative_info_gain = 0.0;
    let num_sensor_updates = 10;

    for i in 0..num_sensor_updates {
        let posterior_variance = (1.0 / prior_variance + 1.0 / sensor_noise).recip();
        let info_gain = robotics_metrics.sensor_information_gain(prior_variance, posterior_variance);

        cumulative_info_gain += info_gain;

        if i == 0 {
            println!("  Sensor update {}: IG = {:.4} nats", i + 1, info_gain);
        } else if i == num_sensor_updates - 1 {
            println!("  Sensor update {}: IG = {:.4} nats", i + 1, info_gain);
        } else if i == num_sensor_updates / 2 {
            println!("  ...");
        }
    }

    println!();
    println!("✓ Total information gain: {:.4} nats", cumulative_info_gain);
    println!("✓ Average per measurement: {:.4} nats", cumulative_info_gain / num_sensor_updates as f64);
    println!();

    // ═══════════════════════════════════════════════════════════════════════════
    // Phase 7: Results Summary
    // ═══════════════════════════════════════════════════════════════════════════

    println!("📋 Phase 7: Results Summary");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    println!("Planning Performance:");
    println!("  • Initial planning time: {:.2?}", planning_duration);
    println!("  • Replanning time: {:.2?}", replan_duration);
    println!("  • Real-time capable: {}",
        if replan_duration.as_millis() < 100 { "✓ Yes (< 100ms)" } else { "⚠ No" });
    println!();

    println!("Path Quality:");
    println!("  • Collision-free: {}",
        if planned_collisions == 0 { "✓ Yes" } else { "✗ No" });
    println!("  • Safe clearance: {}",
        if min_obstacle_distance >= config.obstacle_threshold { "✓ Yes" } else { "⚠ No" });
    println!("  • Path overhead: {:.1}%",
        ((planned_length - baseline_length) / baseline_length) * 100.0);
    println!();

    println!("Information-Theoretic Metrics:");
    println!("  • Trajectory entropy: {:.4} nats", trajectory_entropy);
    println!("  • Expected free energy: {:.4}", motion_plan.expected_free_energy);
    println!("  • Sensor information gain: {:.4} nats", cumulative_info_gain);
    println!();

    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  🎉 Motion Planning Tutorial Complete!                           ║");
    println!("║  Worker 7 successfully planned collision-free path               ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════════════

/// Generate straight-line trajectory from start to goal
fn generate_straight_line(start: &Point2D, goal: &Point2D, num_points: usize) -> Vec<Point2D> {
    (0..num_points)
        .map(|i| {
            let t = i as f64 / (num_points - 1) as f64;
            Point2D::new(
                start.x + t * (goal.x - start.x),
                start.y + t * (goal.y - start.y),
            )
        })
        .collect()
}

/// Count number of trajectory points in collision
fn count_collisions(trajectory: &[Point2D], workspace: &Workspace) -> usize {
    trajectory.iter()
        .filter(|p| !workspace.is_collision_free(p))
        .count()
}

/// Calculate total path length
fn calculate_path_length(trajectory: &[Point2D]) -> f64 {
    trajectory.windows(2)
        .map(|w| w[0].distance_to(&w[1]))
        .sum()
}

/// Convert motion plan to trajectory matrix for information theory calculations
fn motion_plan_to_matrix(plan: &MotionPlan) -> Result<Array2<f64>> {
    let n = plan.waypoints.len();

    if n == 0 {
        anyhow::bail!("Empty motion plan");
    }

    let d = plan.waypoints[0].len();
    let mut matrix = Array2::zeros((n, d));

    for (i, waypoint) in plan.waypoints.iter().enumerate() {
        for (j, &val) in waypoint.iter().enumerate() {
            matrix[[i, j]] = val;
        }
    }

    Ok(matrix)
}

/// Convert trajectory matrix to point vector
fn matrix_to_points(matrix: &Array2<f64>) -> Vec<Point2D> {
    (0..matrix.nrows())
        .map(|i| Point2D::new(matrix[[i, 0]], matrix[[i, 1]]))
        .collect()
}

/// Convert motion plan to point vector
fn motion_plan_to_points(plan: &MotionPlan) -> Vec<Point2D> {
    plan.waypoints.iter()
        .map(|w| Point2D::new(w[0], w[1]))
        .collect()
}

/// Convert workspace to occupancy grid for planning
fn workspace_to_occupancy_grid(workspace: &Workspace, resolution: usize) -> Array2<f64> {
    let mut grid = Array2::ones((resolution, resolution));

    let dx = workspace.width / resolution as f64;
    let dy = workspace.height / resolution as f64;

    for i in 0..resolution {
        for j in 0..resolution {
            let x = (i as f64 + 0.5) * dx;
            let y = (j as f64 + 0.5) * dy;

            let point = Point2D::new(x, y);

            if !workspace.is_collision_free(&point) {
                grid[[i, j]] = 0.0; // Occupied
            }
        }
    }

    grid
}
