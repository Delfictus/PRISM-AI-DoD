//! Trajectory Prediction and Forecasting
//!
//! Time series forecasting for robotics applications:
//! - Predict obstacle trajectories
//! - Forecast environment dynamics
//! - Multi-agent motion prediction
//!
//! Integration point with Worker 1's time series module (when available)

use anyhow::Result;
use ndarray::Array1;

use super::environment_model::ObstacleModel;

/// A point on a trajectory
#[derive(Debug, Clone)]
pub struct TrajectoryPoint {
    /// Time (seconds)
    pub time: f64,
    /// Position [x, y] (meters)
    pub position: Array1<f64>,
    /// Velocity [vx, vy] (meters/second)
    pub velocity: Array1<f64>,
}

/// A complete trajectory
#[derive(Debug, Clone)]
pub struct Trajectory {
    /// Trajectory waypoints
    pub points: Vec<TrajectoryPoint>,
    /// Uncertainty bounds (standard deviation at each point)
    pub uncertainty: Vec<f64>,
}

impl Trajectory {
    /// Create new trajectory
    pub fn new(points: Vec<TrajectoryPoint>) -> Self {
        let n = points.len();
        Self {
            points,
            uncertainty: vec![0.0; n],
        }
    }

    /// Create trajectory with uncertainty quantification
    pub fn with_uncertainty(points: Vec<TrajectoryPoint>, uncertainty: Vec<f64>) -> Self {
        assert_eq!(points.len(), uncertainty.len());
        Self {
            points,
            uncertainty,
        }
    }

    /// Get position at specific time (linear interpolation)
    pub fn position_at(&self, t: f64) -> Option<Array1<f64>> {
        if self.points.is_empty() {
            return None;
        }

        // Find surrounding points
        let mut before = None;
        let mut after = None;

        for (i, point) in self.points.iter().enumerate() {
            if point.time <= t {
                before = Some(i);
            }
            if point.time >= t && after.is_none() {
                after = Some(i);
            }
        }

        match (before, after) {
            (Some(b), Some(a)) if b == a => Some(self.points[b].position.clone()),
            (Some(b), Some(a)) => {
                let t0 = self.points[b].time;
                let t1 = self.points[a].time;
                let alpha = (t - t0) / (t1 - t0);

                let p0 = &self.points[b].position;
                let p1 = &self.points[a].position;

                Some(p0 * (1.0 - alpha) + p1 * alpha)
            }
            (Some(b), None) => Some(self.points[b].position.clone()),
            (None, Some(a)) => Some(self.points[a].position.clone()),
            (None, None) => None,
        }
    }

    /// Total trajectory duration
    pub fn duration(&self) -> f64 {
        if self.points.is_empty() {
            0.0
        } else {
            self.points.last().unwrap().time - self.points.first().unwrap().time
        }
    }
}

/// Trajectory predictor using time series forecasting
///
/// NOTE: This is a placeholder implementation. When Worker 1 completes
/// the time series module, this will be enhanced with ARIMA/LSTM forecasting.
pub struct TrajectoryPredictor {
    /// Prediction horizon (seconds)
    horizon: f64,
    /// Time step for predictions (seconds)
    dt: f64,
}

impl TrajectoryPredictor {
    /// Create new trajectory predictor
    pub fn new() -> Result<Self> {
        Ok(Self {
            horizon: 5.0,
            dt: 0.1,
        })
    }

    /// Predict future trajectory of an obstacle
    ///
    /// Currently uses constant velocity model.
    /// TODO: Replace with Worker 1's time series forecasting when available:
    /// - ARIMA for short-term prediction
    /// - LSTM for complex motion patterns
    /// - Uncertainty quantification
    pub fn predict_obstacle_trajectory(
        &self,
        obstacle: &ObstacleModel,
    ) -> Result<Trajectory> {
        let n_steps = (self.horizon / self.dt) as usize;
        let mut points = Vec::with_capacity(n_steps);
        let mut uncertainty = Vec::with_capacity(n_steps);

        for i in 0..n_steps {
            let t = i as f64 * self.dt;
            let position = obstacle.predict_position(t);
            let velocity = obstacle.velocity.clone();

            points.push(TrajectoryPoint {
                time: t,
                position,
                velocity,
            });

            // Uncertainty grows with prediction horizon
            let uncertainty_t = obstacle.position_uncertainty * (1.0 + t * 0.5);
            uncertainty.push(uncertainty_t);
        }

        Ok(Trajectory::with_uncertainty(points, uncertainty))
    }

    /// Predict trajectories for multiple obstacles
    pub fn predict_obstacle_trajectories(
        &self,
        obstacles: &[ObstacleModel],
        horizon: f64,
    ) -> Result<Vec<ObstacleModel>> {
        // Temporarily override horizon
        let original_horizon = self.horizon;
        let mut predictor = Self {
            horizon,
            dt: self.dt,
        };

        let mut predicted = Vec::new();

        for obstacle in obstacles {
            let trajectory = predictor.predict_obstacle_trajectory(obstacle)?;

            // Get position at end of horizon
            if let Some(final_point) = trajectory.points.last() {
                let mut predicted_obstacle = obstacle.clone();
                predicted_obstacle.position = final_point.position.clone();
                predicted_obstacle.position_uncertainty =
                    *trajectory.uncertainty.last().unwrap_or(&0.0);
                predicted.push(predicted_obstacle);
            }
        }

        // Restore original horizon
        predictor.horizon = original_horizon;

        Ok(predicted)
    }

    /// Forecast environment dynamics
    ///
    /// Placeholder for time series integration.
    /// TODO: Integrate with Worker 1's time series module:
    /// ```rust
    /// use crate::time_series::lstm_forecaster::LSTMForecaster;
    ///
    /// let forecaster = LSTMForecaster::new()?;
    /// let forecast = forecaster.forecast(
    ///     historical_positions,
    ///     horizon_seconds=5.0
    /// )?;
    /// ```
    pub fn forecast_environment_dynamics(
        &self,
        _historical_data: &[EnvironmentSnapshot],
        _horizon: f64,
    ) -> Result<Vec<EnvironmentSnapshot>> {
        // Placeholder implementation
        Ok(Vec::new())
    }

    /// Multi-agent trajectory forecasting
    ///
    /// Predicts coordinated motion of multiple agents.
    /// TODO: Implement with Worker 1's time series and Worker 2's GPU kernels
    pub fn forecast_multi_agent(
        &self,
        _agents: &[AgentState],
        _historical_interactions: &[InteractionHistory],
        _horizon: f64,
    ) -> Result<Vec<Trajectory>> {
        // Placeholder for multi-agent forecasting
        Ok(Vec::new())
    }
}

/// Snapshot of environment state for time series
#[derive(Debug, Clone)]
pub struct EnvironmentSnapshot {
    pub timestamp: f64,
    pub obstacles: Vec<ObstacleModel>,
}

/// Agent state for multi-agent prediction
#[derive(Debug, Clone)]
pub struct AgentState {
    pub id: usize,
    pub position: Array1<f64>,
    pub velocity: Array1<f64>,
}

/// Historical interaction data between agents
#[derive(Debug, Clone)]
pub struct InteractionHistory {
    pub agent_a: usize,
    pub agent_b: usize,
    pub timestamps: Vec<f64>,
    pub distances: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trajectory_creation() {
        let points = vec![
            TrajectoryPoint {
                time: 0.0,
                position: Array1::from_vec(vec![0.0, 0.0]),
                velocity: Array1::from_vec(vec![1.0, 0.0]),
            },
            TrajectoryPoint {
                time: 1.0,
                position: Array1::from_vec(vec![1.0, 0.0]),
                velocity: Array1::from_vec(vec![1.0, 0.0]),
            },
        ];

        let traj = Trajectory::new(points);
        assert_eq!(traj.points.len(), 2);
        assert_eq!(traj.duration(), 1.0);
    }

    #[test]
    fn test_trajectory_interpolation() {
        let points = vec![
            TrajectoryPoint {
                time: 0.0,
                position: Array1::from_vec(vec![0.0, 0.0]),
                velocity: Array1::from_vec(vec![2.0, 0.0]),
            },
            TrajectoryPoint {
                time: 2.0,
                position: Array1::from_vec(vec![4.0, 0.0]),
                velocity: Array1::from_vec(vec![2.0, 0.0]),
            },
        ];

        let traj = Trajectory::new(points);

        // Test interpolation at midpoint
        let pos = traj.position_at(1.0).unwrap();
        assert!((pos[0] - 2.0).abs() < 0.001);
        assert!((pos[1] - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_trajectory_predictor() {
        let predictor = TrajectoryPredictor::new().unwrap();

        let obstacle = ObstacleModel::new_dynamic(
            Array1::from_vec(vec![0.0, 0.0]),
            Array1::from_vec(vec![1.0, 0.5]),
            0.5,
        );

        let predicted = predictor.predict_obstacle_trajectory(&obstacle).unwrap();
        assert!(!predicted.points.is_empty());

        // Check final position approximates constant velocity
        if let Some(final_point) = predicted.points.last() {
            let expected_x = obstacle.velocity[0] * final_point.time;
            let expected_y = obstacle.velocity[1] * final_point.time;

            assert!((final_point.position[0] - expected_x).abs() < 0.01);
            assert!((final_point.position[1] - expected_y).abs() < 0.01);
        }
    }

    #[test]
    fn test_uncertainty_growth() {
        let predictor = TrajectoryPredictor::new().unwrap();

        let obstacle = ObstacleModel::new_dynamic(
            Array1::from_vec(vec![0.0, 0.0]),
            Array1::from_vec(vec![1.0, 0.0]),
            0.5,
        );

        let predicted = predictor.predict_obstacle_trajectory(&obstacle).unwrap();

        // Uncertainty should grow with time
        assert!(predicted.uncertainty.first().unwrap() < predicted.uncertainty.last().unwrap());
    }
}
