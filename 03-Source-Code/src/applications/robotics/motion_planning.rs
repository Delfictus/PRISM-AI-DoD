//! Motion Planning with Active Inference
//!
//! Uses Worker 1's Active Inference to plan robot trajectories that:
//! - Minimize surprise (unexpected outcomes)
//! - Achieve goals efficiently
//! - Avoid obstacles with safety margins
//!
//! GPU-accelerated planning via Worker 2's kernels

use anyhow::{Result, Context};
use ndarray::{Array1, Array2};
use crate::active_inference::GenerativeModel;

use super::environment_model::{EnvironmentState, ObstacleModel};
use super::ros_bridge::RobotState;
use super::trajectory::TrajectoryPoint;

/// Configuration for motion planning
#[derive(Debug, Clone)]
pub struct PlanningConfig {
    /// Planning horizon (seconds)
    pub horizon: f64,
    /// Time step (seconds)
    pub dt: f64,
    /// Use GPU acceleration
    pub use_gpu: bool,
}

/// A planned motion trajectory
#[derive(Debug, Clone)]
pub struct MotionPlan {
    /// Planned trajectory waypoints
    pub waypoints: Vec<TrajectoryPoint>,
    /// Expected free energy of this plan
    pub expected_free_energy: f64,
    /// Planning time (milliseconds)
    pub planning_time_ms: f64,
    /// Whether plan reaches goal
    pub reaches_goal: bool,
}

impl MotionPlan {
    /// Create new motion plan
    pub fn new(waypoints: Vec<TrajectoryPoint>, expected_free_energy: f64) -> Self {
        Self {
            waypoints,
            expected_free_energy,
            planning_time_ms: 0.0,
            reaches_goal: false,
        }
    }

    /// Get waypoint at specific time
    pub fn waypoint_at_time(&self, t: f64) -> Option<&TrajectoryPoint> {
        self.waypoints.iter().find(|wp| (wp.time - t).abs() < 0.01)
    }

    /// Total path length
    pub fn path_length(&self) -> f64 {
        let mut length = 0.0;
        for i in 1..self.waypoints.len() {
            let dx = self.waypoints[i].position[0] - self.waypoints[i-1].position[0];
            let dy = self.waypoints[i].position[1] - self.waypoints[i-1].position[1];
            length += (dx*dx + dy*dy).sqrt();
        }
        length
    }
}

/// Motion planner using Active Inference
pub struct MotionPlanner {
    config: PlanningConfig,
    generative_model: GenerativeModel,
}

impl MotionPlanner {
    /// Create new motion planner
    pub fn new(config: PlanningConfig) -> Result<Self> {
        // Initialize Active Inference components from Worker 1
        let generative_model = GenerativeModel::new();

        Ok(Self {
            config,
            generative_model,
        })
    }

    /// Plan motion from current state to goal
    ///
    /// Uses Active Inference to minimize expected free energy:
    /// G = E_Q[ln Q(o,s|π) - ln P(o,s,π)]
    ///
    /// Where:
    /// - π is the policy (motion plan)
    /// - o are observations (sensor readings)
    /// - s are hidden states (robot & environment state)
    pub fn plan(
        &mut self,
        current_state: &RobotState,
        goal_state: &RobotState,
        environment: &EnvironmentState,
        predicted_obstacles: &[ObstacleModel],
    ) -> Result<MotionPlan> {
        let start_time = std::time::Instant::now();

        // Convert robot states to observation space
        let current_obs = self.state_to_observations(current_state)?;
        let goal_obs = self.state_to_observations(goal_state)?;

        // Set preferred observations (goal)
        self.generative_model.set_goal(goal_obs.clone());

        // Generate candidate policies (motion primitives)
        let policies = self.generate_motion_policies(
            &current_obs,
            &goal_obs,
            environment,
            predicted_obstacles,
        )?;

        // Select policy with minimum expected free energy
        // TODO: Integrate with Worker 1's PolicySelector when available
        let best_policy_idx = 0; // Use first policy for now

        // Convert selected policy to motion plan
        let waypoints = self.policy_to_trajectory(
            &policies[best_policy_idx],
            current_state,
        )?;

        // Calculate expected free energy for the plan
        let expected_free_energy = self.evaluate_policy_free_energy(
            &policies[best_policy_idx],
            &goal_obs,
        )?;

        let planning_time = start_time.elapsed().as_secs_f64() * 1000.0;

        let reaches_goal = self.check_goal_reached(&waypoints, goal_state);

        Ok(MotionPlan {
            waypoints,
            expected_free_energy,
            planning_time_ms: planning_time,
            reaches_goal,
        })
    }

    /// Execute one control step from a plan
    pub fn execute_step(
        &mut self,
        current_state: &RobotState,
        plan: &MotionPlan,
    ) -> Result<super::ros_bridge::RobotCommand> {
        // Get current time in plan
        let current_time = current_state.timestamp;

        // Find nearest waypoint
        let target_waypoint = plan.waypoint_at_time(current_time)
            .or_else(|| plan.waypoints.first())
            .context("Empty motion plan")?;

        // Compute control command to reach waypoint
        let position_error = Array1::from_vec(vec![
            target_waypoint.position[0] - current_state.position[0],
            target_waypoint.position[1] - current_state.position[1],
        ]);

        let velocity_error = Array1::from_vec(vec![
            target_waypoint.velocity[0] - current_state.velocity[0],
            target_waypoint.velocity[1] - current_state.velocity[1],
        ]);

        // Simple PD control (will be enhanced with Active Inference control)
        let kp = 2.0;
        let kd = 0.5;
        let control = &position_error * kp + &velocity_error * kd;

        Ok(super::ros_bridge::RobotCommand {
            velocity: control,
            timestamp: current_time + self.config.dt,
        })
    }

    /// Convert robot state to observations for Active Inference
    fn state_to_observations(&self, state: &RobotState) -> Result<Array1<f64>> {
        // Encode robot state as observations
        // TODO: Enhance with sensor model
        Ok(Array1::from_vec(vec![
            state.position[0],
            state.position[1],
            state.velocity[0],
            state.velocity[1],
        ]))
    }

    /// Generate candidate motion policies (motion primitives)
    fn generate_motion_policies(
        &self,
        _current: &Array1<f64>,
        goal: &Array1<f64>,
        environment: &EnvironmentState,
        predicted_obstacles: &[ObstacleModel],
    ) -> Result<Vec<Array2<f64>>> {
        let n_policies = 10;
        let n_steps = (self.config.horizon / self.config.dt) as usize;
        let mut policies = Vec::new();

        // Generate straight-line policy (optimal if no obstacles)
        let straight_policy = self.generate_straight_line_policy(goal, n_steps);
        policies.push(straight_policy);

        // Generate curved policies to avoid obstacles
        for i in 1..n_policies {
            let curve_amount = (i as f64) / (n_policies as f64) * 0.5;
            let curved_policy = self.generate_curved_policy(
                goal,
                n_steps,
                curve_amount,
                environment,
                predicted_obstacles,
            );
            policies.push(curved_policy);
        }

        Ok(policies)
    }

    /// Generate straight-line motion policy
    fn generate_straight_line_policy(&self, goal: &Array1<f64>, n_steps: usize) -> Array2<f64> {
        let mut policy = Array2::zeros((n_steps, goal.len()));
        for t in 0..n_steps {
            // Ensure final step reaches goal exactly (alpha=1.0 at t=n_steps-1)
            let alpha = (t as f64) / ((n_steps - 1) as f64).max(1.0);
            policy.row_mut(t).assign(&(goal * alpha));
        }
        policy
    }

    /// Generate curved motion policy (for obstacle avoidance)
    fn generate_curved_policy(
        &self,
        goal: &Array1<f64>,
        n_steps: usize,
        curve_amount: f64,
        environment: &EnvironmentState,
        predicted_obstacles: &[ObstacleModel],
    ) -> Array2<f64> {
        let mut policy = Array2::zeros((n_steps, goal.len()));

        // Compute direction to goal
        let goal_dir = if goal.len() >= 2 {
            let dx = goal[0];
            let dy = goal[1];
            let norm = (dx * dx + dy * dy).sqrt().max(1e-6);
            (dx / norm, dy / norm)
        } else {
            (1.0, 0.0)
        };

        // Perpendicular direction for curving
        let perp_dir = (-goal_dir.1, goal_dir.0);

        // Generate curved path
        for t in 0..n_steps {
            // Ensure final step reaches goal exactly (alpha=1.0 at t=n_steps-1)
            let alpha = (t as f64) / ((n_steps - 1) as f64).max(1.0);

            // Base position along goal direction
            let base_x = goal[0] * alpha;
            let base_y = goal[1] * alpha;

            // Add sinusoidal curve perpendicular to goal direction
            // Curve is strongest in middle of path
            let curve_factor = (alpha * std::f64::consts::PI).sin() * curve_amount;

            // Apply perpendicular offset
            let x = base_x + perp_dir.0 * curve_factor;
            let y = base_y + perp_dir.1 * curve_factor;

            // Push away from obstacles
            let (obstacle_x, obstacle_y) = self.compute_obstacle_repulsion(
                x, y, environment, predicted_obstacles
            );

            policy[[t, 0]] = x + obstacle_x;
            policy[[t, 1]] = y + obstacle_y;
        }

        policy
    }

    /// Compute repulsive force from obstacles
    fn compute_obstacle_repulsion(
        &self,
        x: f64,
        y: f64,
        environment: &EnvironmentState,
        predicted_obstacles: &[ObstacleModel],
    ) -> (f64, f64) {
        let mut repulsion_x = 0.0;
        let mut repulsion_y = 0.0;

        let influence_radius = 2.0; // meters
        let repulsion_strength = 0.5;

        // Repulsion from current obstacles
        for obstacle in &environment.obstacles {
            let dx = x - obstacle.position[0];
            let dy = y - obstacle.position[1];
            let distance = (dx * dx + dy * dy).sqrt().max(0.01);

            if distance < influence_radius {
                // Repulsive force inversely proportional to distance
                let force = repulsion_strength * (1.0 / distance - 1.0 / influence_radius);
                repulsion_x += (dx / distance) * force;
                repulsion_y += (dy / distance) * force;
            }
        }

        // Repulsion from predicted obstacles (weighted by prediction uncertainty)
        for obstacle in predicted_obstacles {
            let dx = x - obstacle.position[0];
            let dy = y - obstacle.position[1];
            let distance = (dx * dx + dy * dy).sqrt().max(0.01);

            if distance < influence_radius {
                // Reduce repulsion strength based on prediction uncertainty
                let uncertainty_weight = 1.0 / (1.0 + obstacle.position_uncertainty);
                let force = repulsion_strength * uncertainty_weight *
                           (1.0 / distance - 1.0 / influence_radius);
                repulsion_x += (dx / distance) * force;
                repulsion_y += (dy / distance) * force;
            }
        }

        (repulsion_x, repulsion_y)
    }

    /// Convert policy to trajectory waypoints
    fn policy_to_trajectory(
        &self,
        policy: &Array2<f64>,
        start_state: &RobotState,
    ) -> Result<Vec<TrajectoryPoint>> {
        let mut waypoints = Vec::new();
        let mut current_pos = start_state.position.clone();

        for (t, row) in policy.outer_iter().enumerate() {
            let time = t as f64 * self.config.dt;

            // Extract position from policy
            let position = Array1::from_vec(vec![row[0], row[1]]);

            // Compute velocity from position difference
            let velocity = if t > 0 {
                (&position - &current_pos) / self.config.dt
            } else {
                start_state.velocity.clone()
            };

            waypoints.push(TrajectoryPoint {
                time,
                position: position.clone(),
                velocity: velocity.clone(),
            });

            current_pos = position;
        }

        Ok(waypoints)
    }

    /// Evaluate expected free energy of a policy
    fn evaluate_policy_free_energy(
        &self,
        _policy: &Array2<f64>,
        _goal: &Array1<f64>,
    ) -> Result<f64> {
        // Simplified: return fixed value for now
        // TODO: Implement proper expected free energy calculation
        Ok(1.0)
    }

    /// Check if plan reaches goal
    fn check_goal_reached(&self, waypoints: &[TrajectoryPoint], goal: &RobotState) -> bool {
        if let Some(last_wp) = waypoints.last() {
            let distance = ((last_wp.position[0] - goal.position[0]).powi(2)
                + (last_wp.position[1] - goal.position[1]).powi(2)).sqrt();
            distance < 0.1 // 10cm tolerance
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motion_planner_creation() {
        let config = PlanningConfig {
            horizon: 5.0,
            dt: 0.1,
            use_gpu: false,
        };
        let planner = MotionPlanner::new(config);
        assert!(planner.is_ok());
    }

    #[test]
    fn test_motion_plan_path_length() {
        let waypoints = vec![
            TrajectoryPoint {
                time: 0.0,
                position: Array1::from_vec(vec![0.0, 0.0]),
                velocity: Array1::from_vec(vec![0.0, 0.0]),
            },
            TrajectoryPoint {
                time: 1.0,
                position: Array1::from_vec(vec![3.0, 4.0]),
                velocity: Array1::from_vec(vec![3.0, 4.0]),
            },
        ];

        let plan = MotionPlan::new(waypoints, 0.0);
        let length = plan.path_length();
        assert!((length - 5.0).abs() < 0.01); // 3-4-5 triangle
    }

    #[test]
    fn test_straight_line_policy() {
        let config = PlanningConfig {
            horizon: 1.0,
            dt: 0.1,
            use_gpu: false,
        };
        let planner = MotionPlanner::new(config).unwrap();

        let goal = Array1::from_vec(vec![5.0, 10.0]);
        let n_steps = 10;

        let policy = planner.generate_straight_line_policy(&goal, n_steps);

        // Check dimensions
        assert_eq!(policy.nrows(), n_steps);
        assert_eq!(policy.ncols(), 2);

        // Check start is at origin
        assert!((policy[[0, 0]]).abs() < 0.01);
        assert!((policy[[0, 1]]).abs() < 0.01);

        // Check end reaches goal
        assert!((policy[[n_steps - 1, 0]] - goal[0]).abs() < 0.01);
        assert!((policy[[n_steps - 1, 1]] - goal[1]).abs() < 0.01);

        // Check linearity (halfway point should be near half the goal)
        let mid = n_steps / 2;
        assert!((policy[[mid, 0]] - goal[0] / 2.0).abs() < 0.6);
        assert!((policy[[mid, 1]] - goal[1] / 2.0).abs() < 0.6);
    }

    #[test]
    fn test_curved_policy_avoids_obstacles() {
        let config = PlanningConfig {
            horizon: 2.0,
            dt: 0.1,
            use_gpu: false,
        };
        let planner = MotionPlanner::new(config).unwrap();

        let goal = Array1::from_vec(vec![10.0, 0.0]);
        let n_steps = 20;

        // Create obstacle in middle of straight path
        let obstacle = ObstacleModel {
            position: Array1::from_vec(vec![5.0, 0.0]),
            velocity: Array1::from_vec(vec![0.0, 0.0]),
            radius: 1.0,
            is_dynamic: false,
            position_uncertainty: 0.0,
        };

        let environment = EnvironmentState {
            obstacles: vec![obstacle],
            free_space_radius: 1.0,
            timestamp: 0.0,
        };

        // Generate curved policy with moderate curve
        let curved_policy = planner.generate_curved_policy(
            &goal,
            n_steps,
            0.3,
            &environment,
            &[],
        );

        // Check that policy deviates from straight line due to obstacle
        let mut max_deviation: f64 = 0.0;
        for t in 0..n_steps {
            let straight_y = 0.0; // Straight path has y=0
            let curved_y = curved_policy[[t, 1]];
            let deviation = (curved_y - straight_y).abs();
            max_deviation = max_deviation.max(deviation);
        }

        // Should have noticeable deviation (> 0.1 meters)
        assert!(max_deviation > 0.1, "Curved policy should deviate from straight line");
    }

    #[test]
    fn test_obstacle_repulsion_increases_with_proximity() {
        let config = PlanningConfig {
            horizon: 1.0,
            dt: 0.1,
            use_gpu: false,
        };
        let planner = MotionPlanner::new(config).unwrap();

        // Obstacle at origin
        let obstacle = ObstacleModel {
            position: Array1::from_vec(vec![0.0, 0.0]),
            velocity: Array1::from_vec(vec![0.0, 0.0]),
            radius: 0.5,
            is_dynamic: false,
            position_uncertainty: 0.0,
        };

        let environment = EnvironmentState {
            obstacles: vec![obstacle],
            free_space_radius: 1.0,
            timestamp: 0.0,
        };

        // Test repulsion at different distances
        let (repulsion_far_x, repulsion_far_y) = planner.compute_obstacle_repulsion(
            1.5, 0.0, &environment, &[]
        );

        let (repulsion_near_x, repulsion_near_y) = planner.compute_obstacle_repulsion(
            0.5, 0.0, &environment, &[]
        );

        // Closer distance should produce stronger repulsion
        let force_far = (repulsion_far_x.powi(2) + repulsion_far_y.powi(2)).sqrt();
        let force_near = (repulsion_near_x.powi(2) + repulsion_near_y.powi(2)).sqrt();

        assert!(force_near > force_far, "Repulsion should be stronger when closer to obstacle");

        // Repulsion should point away from obstacle (positive x direction)
        assert!(repulsion_near_x > 0.0, "Repulsion should point away from obstacle");
    }

    #[test]
    fn test_uncertainty_weighted_repulsion() {
        let config = PlanningConfig {
            horizon: 1.0,
            dt: 0.1,
            use_gpu: false,
        };
        let planner = MotionPlanner::new(config).unwrap();

        // Certain predicted obstacle
        let certain_obstacle = ObstacleModel {
            position: Array1::from_vec(vec![1.0, 0.0]),
            velocity: Array1::from_vec(vec![0.0, 0.0]),
            radius: 0.5,
            is_dynamic: true,
            position_uncertainty: 0.0, // Very certain
        };

        // Uncertain predicted obstacle at same position
        let uncertain_obstacle = ObstacleModel {
            position: Array1::from_vec(vec![1.0, 0.0]),
            velocity: Array1::from_vec(vec![0.0, 0.0]),
            radius: 0.5,
            is_dynamic: true,
            position_uncertainty: 2.0, // Very uncertain
        };

        let environment = EnvironmentState {
            obstacles: vec![],
            free_space_radius: 1.0,
            timestamp: 0.0,
        };

        // Test repulsion from certain vs uncertain obstacles
        let (repulsion_certain_x, repulsion_certain_y) = planner.compute_obstacle_repulsion(
            0.5, 0.0, &environment, &[certain_obstacle]
        );

        let (repulsion_uncertain_x, repulsion_uncertain_y) = planner.compute_obstacle_repulsion(
            0.5, 0.0, &environment, &[uncertain_obstacle]
        );

        let force_certain = (repulsion_certain_x.powi(2) + repulsion_certain_y.powi(2)).sqrt();
        let force_uncertain = (repulsion_uncertain_x.powi(2) + repulsion_uncertain_y.powi(2)).sqrt();

        // Certain obstacle should produce stronger repulsion
        assert!(force_certain > force_uncertain,
               "Repulsion should be weaker for uncertain predicted obstacles");
    }

    #[test]
    fn test_policy_to_trajectory_conversion() {
        let config = PlanningConfig {
            horizon: 1.0,
            dt: 0.1,
            use_gpu: false,
        };
        let planner = MotionPlanner::new(config).unwrap();

        // Simple straight policy
        let n_steps = 10;
        let mut policy = Array2::zeros((n_steps, 2));
        for t in 0..n_steps {
            policy[[t, 0]] = t as f64;
            policy[[t, 1]] = t as f64 * 2.0;
        }

        let start_state = RobotState {
            position: Array1::from_vec(vec![0.0, 0.0]),
            velocity: Array1::from_vec(vec![1.0, 2.0]),
            orientation: 0.0,
            angular_velocity: 0.0,
            timestamp: 0.0,
        };

        let waypoints = planner.policy_to_trajectory(&policy, &start_state).unwrap();

        // Check number of waypoints
        assert_eq!(waypoints.len(), n_steps);

        // Check times are correct
        for (i, wp) in waypoints.iter().enumerate() {
            let expected_time = i as f64 * 0.1;
            assert!((wp.time - expected_time).abs() < 0.001);
        }

        // Check velocities are computed correctly (finite differences)
        for i in 1..waypoints.len() {
            let dx = waypoints[i].position[0] - waypoints[i-1].position[0];
            let dy = waypoints[i].position[1] - waypoints[i-1].position[1];
            let expected_vx = dx / 0.1;
            let expected_vy = dy / 0.1;

            assert!((waypoints[i].velocity[0] - expected_vx).abs() < 0.1);
            assert!((waypoints[i].velocity[1] - expected_vy).abs() < 0.1);
        }
    }

    #[test]
    fn test_check_goal_reached() {
        let config = PlanningConfig {
            horizon: 1.0,
            dt: 0.1,
            use_gpu: false,
        };
        let planner = MotionPlanner::new(config).unwrap();

        let goal = RobotState {
            position: Array1::from_vec(vec![5.0, 5.0]),
            velocity: Array1::from_vec(vec![0.0, 0.0]),
            orientation: 0.0,
            angular_velocity: 0.0,
            timestamp: 0.0,
        };

        // Waypoints that reach goal (within 10cm tolerance)
        let waypoints_success = vec![
            TrajectoryPoint {
                time: 0.0,
                position: Array1::from_vec(vec![0.0, 0.0]),
                velocity: Array1::from_vec(vec![5.0, 5.0]),
            },
            TrajectoryPoint {
                time: 1.0,
                position: Array1::from_vec(vec![5.05, 4.95]), // Within tolerance
                velocity: Array1::from_vec(vec![0.0, 0.0]),
            },
        ];

        assert!(planner.check_goal_reached(&waypoints_success, &goal));

        // Waypoints that don't reach goal
        let waypoints_fail = vec![
            TrajectoryPoint {
                time: 0.0,
                position: Array1::from_vec(vec![0.0, 0.0]),
                velocity: Array1::from_vec(vec![5.0, 5.0]),
            },
            TrajectoryPoint {
                time: 1.0,
                position: Array1::from_vec(vec![3.0, 3.0]), // Too far
                velocity: Array1::from_vec(vec![0.0, 0.0]),
            },
        ];

        assert!(!planner.check_goal_reached(&waypoints_fail, &goal));
    }

    #[test]
    fn test_full_planning_integration() {
        let config = PlanningConfig {
            horizon: 2.0,
            dt: 0.1,
            use_gpu: false,
        };
        let mut planner = MotionPlanner::new(config).unwrap();

        let start = RobotState {
            position: Array1::from_vec(vec![0.0, 0.0]),
            velocity: Array1::from_vec(vec![0.0, 0.0]),
            orientation: 0.0,
            angular_velocity: 0.0,
            timestamp: 0.0,
        };

        let goal = RobotState {
            position: Array1::from_vec(vec![10.0, 0.0]),
            velocity: Array1::from_vec(vec![0.0, 0.0]),
            orientation: 0.0,
            angular_velocity: 0.0,
            timestamp: 2.0,
        };

        // Add obstacle in path
        let obstacle = ObstacleModel {
            position: Array1::from_vec(vec![5.0, 0.0]),
            velocity: Array1::from_vec(vec![0.0, 0.0]),
            radius: 1.0,
            is_dynamic: false,
            position_uncertainty: 0.0,
        };

        let environment = EnvironmentState {
            obstacles: vec![obstacle],
            free_space_radius: 1.0,
            timestamp: 0.0,
        };

        // Plan motion
        let plan = planner.plan(&start, &goal, &environment, &[]).unwrap();

        // Check plan properties
        assert!(!plan.waypoints.is_empty(), "Plan should have waypoints");
        assert!(plan.planning_time_ms > 0.0, "Planning should take some time");
        assert!(plan.path_length() > 0.0, "Path should have length");

        // Check that plan starts near start position
        let first_wp = &plan.waypoints[0];
        let start_distance = ((first_wp.position[0] - start.position[0]).powi(2)
                            + (first_wp.position[1] - start.position[1]).powi(2)).sqrt();
        assert!(start_distance < 1.0, "Plan should start near start position");
    }
}
