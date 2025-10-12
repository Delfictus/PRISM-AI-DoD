//! Environment Model for Robotics
//!
//! Represents the robot's understanding of its environment including:
//! - Obstacles (static and dynamic)
//! - Workspace boundaries
//! - Sensor uncertainty

use anyhow::Result;
use ndarray::Array1;

use super::ros_bridge::RobotState;

/// Model of the robot's environment
#[derive(Debug, Clone)]
pub struct EnvironmentModel {
    /// Workspace boundaries [xmin, xmax, ymin, ymax]
    pub boundaries: [f64; 4],
    /// Static obstacles
    pub static_obstacles: Vec<ObstacleModel>,
    /// Sensor noise standard deviation
    pub sensor_noise_std: f64,
}

impl EnvironmentModel {
    /// Create new environment model with default boundaries
    pub fn new() -> Self {
        Self {
            boundaries: [-10.0, 10.0, -10.0, 10.0],
            static_obstacles: Vec::new(),
            sensor_noise_std: 0.01,
        }
    }

    /// Observe environment from current robot state
    pub fn observe(&self, robot_state: &RobotState) -> Result<EnvironmentState> {
        // Detect nearby obstacles (simplified sensor model)
        let nearby_obstacles = self.detect_nearby_obstacles(&robot_state.position);

        Ok(EnvironmentState {
            obstacles: nearby_obstacles,
            free_space_radius: self.compute_free_space_radius(&robot_state.position),
            timestamp: robot_state.timestamp,
        })
    }

    /// Add static obstacle to environment
    pub fn add_obstacle(&mut self, obstacle: ObstacleModel) {
        self.static_obstacles.push(obstacle);
    }

    /// Detect obstacles near a position
    fn detect_nearby_obstacles(&self, position: &Array1<f64>) -> Vec<ObstacleModel> {
        let detection_radius = 5.0; // meters

        self.static_obstacles
            .iter()
            .filter(|obs| {
                let dx = obs.position[0] - position[0];
                let dy = obs.position[1] - position[1];
                (dx * dx + dy * dy).sqrt() < detection_radius
            })
            .cloned()
            .collect()
    }

    /// Compute radius of free space around position
    fn compute_free_space_radius(&self, position: &Array1<f64>) -> f64 {
        self.static_obstacles
            .iter()
            .map(|obs| {
                let dx = obs.position[0] - position[0];
                let dy = obs.position[1] - position[1];
                (dx * dx + dy * dy).sqrt() - obs.radius
            })
            .fold(f64::INFINITY, f64::min)
            .max(0.0)
    }

    /// Check if position is collision-free
    pub fn is_collision_free(&self, position: &Array1<f64>, robot_radius: f64) -> bool {
        // Check workspace boundaries
        if position[0] < self.boundaries[0] + robot_radius
            || position[0] > self.boundaries[1] - robot_radius
            || position[1] < self.boundaries[2] + robot_radius
            || position[1] > self.boundaries[3] - robot_radius
        {
            return false;
        }

        // Check obstacles
        for obstacle in &self.static_obstacles {
            let dx = obstacle.position[0] - position[0];
            let dy = obstacle.position[1] - position[1];
            let distance = (dx * dx + dy * dy).sqrt();
            if distance < (obstacle.radius + robot_radius) {
                return false;
            }
        }

        true
    }
}

impl Default for EnvironmentModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Current state of the environment
#[derive(Debug, Clone)]
pub struct EnvironmentState {
    /// Detected obstacles
    pub obstacles: Vec<ObstacleModel>,
    /// Radius of free space around robot
    pub free_space_radius: f64,
    /// Timestamp of observation
    pub timestamp: f64,
}

/// Model of an obstacle
#[derive(Debug, Clone)]
pub struct ObstacleModel {
    /// Position [x, y] (meters)
    pub position: Array1<f64>,
    /// Velocity [vx, vy] (meters/second)
    pub velocity: Array1<f64>,
    /// Obstacle radius (meters)
    pub radius: f64,
    /// Is this obstacle dynamic (moving)?
    pub is_dynamic: bool,
    /// Uncertainty in position (standard deviation)
    pub position_uncertainty: f64,
}

impl ObstacleModel {
    /// Create new static obstacle
    pub fn new_static(position: Array1<f64>, radius: f64) -> Self {
        Self {
            position,
            velocity: Array1::zeros(2),
            radius,
            is_dynamic: false,
            position_uncertainty: 0.01,
        }
    }

    /// Create new dynamic obstacle
    pub fn new_dynamic(
        position: Array1<f64>,
        velocity: Array1<f64>,
        radius: f64,
    ) -> Self {
        Self {
            position,
            velocity,
            radius,
            is_dynamic: true,
            position_uncertainty: 0.05,
        }
    }

    /// Predict obstacle position after time dt
    pub fn predict_position(&self, dt: f64) -> Array1<f64> {
        if self.is_dynamic {
            &self.position + &self.velocity * dt
        } else {
            self.position.clone()
        }
    }

    /// Check if point is inside obstacle (with margin)
    pub fn contains_point(&self, point: &Array1<f64>, margin: f64) -> bool {
        let dx = point[0] - self.position[0];
        let dy = point[1] - self.position[1];
        (dx * dx + dy * dy).sqrt() < (self.radius + margin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_environment_model_creation() {
        let env = EnvironmentModel::new();
        assert_eq!(env.boundaries, [-10.0, 10.0, -10.0, 10.0]);
        assert_eq!(env.static_obstacles.len(), 0);
    }

    #[test]
    fn test_add_obstacle() {
        let mut env = EnvironmentModel::new();
        let obstacle = ObstacleModel::new_static(
            Array1::from_vec(vec![1.0, 1.0]),
            0.5,
        );
        env.add_obstacle(obstacle);
        assert_eq!(env.static_obstacles.len(), 1);
    }

    #[test]
    fn test_collision_detection() {
        let mut env = EnvironmentModel::new();
        env.add_obstacle(ObstacleModel::new_static(
            Array1::from_vec(vec![1.0, 1.0]),
            0.5,
        ));

        let robot_radius = 0.3;

        // Point far from obstacle - should be free
        let free_point = Array1::from_vec(vec![5.0, 5.0]);
        assert!(env.is_collision_free(&free_point, robot_radius));

        // Point inside obstacle - should collide
        let collision_point = Array1::from_vec(vec![1.0, 1.0]);
        assert!(!env.is_collision_free(&collision_point, robot_radius));
    }

    #[test]
    fn test_obstacle_prediction() {
        let obstacle = ObstacleModel::new_dynamic(
            Array1::from_vec(vec![0.0, 0.0]),
            Array1::from_vec(vec![1.0, 2.0]),
            0.5,
        );

        let predicted = obstacle.predict_position(2.0);
        assert_eq!(predicted[0], 2.0);
        assert_eq!(predicted[1], 4.0);
    }

    #[test]
    fn test_static_obstacle_prediction() {
        let obstacle = ObstacleModel::new_static(
            Array1::from_vec(vec![1.0, 1.0]),
            0.5,
        );

        let predicted = obstacle.predict_position(5.0);
        assert_eq!(predicted[0], 1.0);
        assert_eq!(predicted[1], 1.0);
    }
}
