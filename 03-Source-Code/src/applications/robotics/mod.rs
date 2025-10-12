//! Robotics Applications Module - Worker 7
//!
//! Motion planning and control using Active Inference for robotic systems.
//! Integrates with Worker 1's Active Inference core and Worker 2's GPU kernels.
//!
//! Constitution: Worker 7 - Robotics & Scientific Discovery
//! - Motion planning with Active Inference
//! - Environment dynamics prediction
//! - Trajectory forecasting
//! - ROS integration

pub mod motion_planning;
pub mod environment_model;
pub mod trajectory;
pub mod ros_bridge;

pub use motion_planning::{MotionPlanner, PlanningConfig, MotionPlan};
pub use environment_model::{EnvironmentModel, EnvironmentState, ObstacleModel};
pub use trajectory::{TrajectoryPredictor, Trajectory, TrajectoryPoint};
pub use ros_bridge::{RosInterface, RobotState, RobotCommand};

/// Robotics module configuration
#[derive(Debug, Clone)]
pub struct RoboticsConfig {
    /// Planning horizon (seconds)
    pub planning_horizon: f64,
    /// Control frequency (Hz)
    pub control_frequency: f64,
    /// Use GPU acceleration for planning
    pub use_gpu: bool,
    /// Enable trajectory forecasting
    pub enable_forecasting: bool,
    /// Maximum planning time (milliseconds)
    pub max_planning_time_ms: u64,
}

impl Default for RoboticsConfig {
    fn default() -> Self {
        Self {
            planning_horizon: 5.0,
            control_frequency: 50.0,
            use_gpu: true,
            enable_forecasting: true,
            max_planning_time_ms: 100,
        }
    }
}

/// Unified robotics controller integrating all Worker 7 components
pub struct RoboticsController {
    config: RoboticsConfig,
    motion_planner: MotionPlanner,
    environment_model: EnvironmentModel,
    trajectory_predictor: Option<TrajectoryPredictor>,
}

impl RoboticsController {
    /// Create new robotics controller
    pub fn new(config: RoboticsConfig) -> anyhow::Result<Self> {
        let motion_planner = MotionPlanner::new(PlanningConfig {
            horizon: config.planning_horizon,
            dt: 1.0 / config.control_frequency,
            use_gpu: config.use_gpu,
        })?;

        let environment_model = EnvironmentModel::new();

        let trajectory_predictor = if config.enable_forecasting {
            Some(TrajectoryPredictor::new()?)
        } else {
            None
        };

        Ok(Self {
            config,
            motion_planner,
            environment_model,
            trajectory_predictor,
        })
    }

    /// Plan motion from current state to goal
    pub fn plan_motion(
        &mut self,
        current_state: &RobotState,
        goal_state: &RobotState,
    ) -> anyhow::Result<MotionPlan> {
        // Update environment model
        let env_state = self.environment_model.observe(current_state)?;

        // Predict future environment dynamics if forecasting enabled
        let predicted_obstacles = if let Some(predictor) = &self.trajectory_predictor {
            predictor.predict_obstacle_trajectories(
                &env_state.obstacles,
                self.config.planning_horizon,
            )?
        } else {
            Vec::new()
        };

        // Plan motion using Active Inference
        self.motion_planner.plan(
            current_state,
            goal_state,
            &env_state,
            &predicted_obstacles,
        )
    }

    /// Execute one control step
    pub fn control_step(
        &mut self,
        current_state: &RobotState,
        plan: &MotionPlan,
    ) -> anyhow::Result<RobotCommand> {
        self.motion_planner.execute_step(current_state, plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_robotics_config_default() {
        let config = RoboticsConfig::default();
        assert_eq!(config.planning_horizon, 5.0);
        assert_eq!(config.control_frequency, 50.0);
        assert!(config.use_gpu);
        assert!(config.enable_forecasting);
    }

    #[test]
    fn test_robotics_controller_creation() {
        let config = RoboticsConfig::default();
        let controller = RoboticsController::new(config);
        assert!(controller.is_ok());
    }
}
