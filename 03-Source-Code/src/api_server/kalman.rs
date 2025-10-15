//! Kalman filtering for multi-sensor data fusion
//!
//! Implements extended Kalman filter (EKF) for optimal state estimation
//! from noisy sensor measurements with uncertainty quantification.

use serde::{Deserialize, Serialize};

/// 6D state vector: [x, y, z, vx, vy, vz]
type StateVector = [f64; 6];

/// 6x6 covariance matrix (stored as flattened array)
type CovarianceMatrix = [[f64; 6]; 6];

/// 3D measurement vector: [x, y, z]
type MeasurementVector = [f64; 3];

/// Kalman filter for tracking objects in 3D space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KalmanFilter {
    /// Current state estimate [x, y, z, vx, vy, vz]
    pub state: StateVector,

    /// State covariance matrix (uncertainty in estimate)
    pub covariance: CovarianceMatrix,

    /// Process noise covariance Q
    process_noise: CovarianceMatrix,

    /// Measurement noise covariance R (3x3)
    measurement_noise: [[f64; 3]; 3],

    /// Time of last update
    last_update_time: f64,
}

impl KalmanFilter {
    /// Create new Kalman filter with initial state and uncertainties
    pub fn new(
        initial_position: [f64; 3],
        initial_velocity: [f64; 3],
        position_uncertainty: f64,
        velocity_uncertainty: f64,
        measurement_noise: f64,
    ) -> Self {
        let state = [
            initial_position[0],
            initial_position[1],
            initial_position[2],
            initial_velocity[0],
            initial_velocity[1],
            initial_velocity[2],
        ];

        // Initial covariance matrix
        let mut covariance = [[0.0; 6]; 6];
        for i in 0..3 {
            covariance[i][i] = position_uncertainty * position_uncertainty;
            covariance[i + 3][i + 3] = velocity_uncertainty * velocity_uncertainty;
        }

        // Process noise (models uncertainty in dynamics)
        let mut process_noise = [[0.0; 6]; 6];
        let q_pos = 0.1;
        let q_vel = 1.0;
        for i in 0..3 {
            process_noise[i][i] = q_pos * q_pos;
            process_noise[i + 3][i + 3] = q_vel * q_vel;
        }

        // Measurement noise covariance (3x3 for position measurements)
        let mut meas_noise = [[0.0; 3]; 3];
        for i in 0..3 {
            meas_noise[i][i] = measurement_noise * measurement_noise;
        }

        Self {
            state,
            covariance,
            process_noise,
            measurement_noise: meas_noise,
            last_update_time: 0.0,
        }
    }

    /// Prediction step: propagate state and covariance forward in time
    ///
    /// Uses constant velocity model: x(t+dt) = x(t) + vx(t)*dt
    pub fn predict(&mut self, dt: f64) {
        // State transition matrix F (constant velocity model)
        let f = self.state_transition_matrix(dt);

        // Predict state: x = F * x
        let new_state = self.matrix_vector_multiply_6(&f, &self.state);
        self.state = new_state;

        // Predict covariance: P = F * P * F^T + Q
        let f_p = self.matrix_multiply_6(&f, &self.covariance);
        let f_p_ft = self.matrix_multiply_transpose_6(&f_p, &f);
        self.covariance = self.matrix_add_6(&f_p_ft, &self.process_noise);

        self.last_update_time += dt;
    }

    /// Update step: incorporate measurement to refine estimate
    ///
    /// Uses Kalman gain to optimally blend prediction with measurement
    pub fn update(&mut self, measurement: MeasurementVector) {
        // Measurement matrix H (we only measure position, not velocity)
        let h = self.measurement_matrix();

        // Innovation: y = z - H * x
        let h_x = self.measurement_prediction(&h);
        let innovation = [
            measurement[0] - h_x[0],
            measurement[1] - h_x[1],
            measurement[2] - h_x[2],
        ];

        // Innovation covariance: S = H * P * H^T + R
        let innovation_cov = self.innovation_covariance(&h);

        // Kalman gain: K = P * H^T * S^(-1)
        let kalman_gain = self.compute_kalman_gain(&h, &innovation_cov);

        // Update state: x = x + K * y
        for i in 0..6 {
            for j in 0..3 {
                self.state[i] += kalman_gain[i][j] * innovation[j];
            }
        }

        // Update covariance: P = (I - K * H) * P
        let identity = self.identity_6();
        let k_h = self.kalman_times_measurement(&kalman_gain, &h);
        let i_minus_kh = self.matrix_subtract_6(&identity, &k_h);
        self.covariance = self.matrix_multiply_6(&i_minus_kh, &self.covariance);
    }

    /// Get current position estimate
    pub fn position(&self) -> [f64; 3] {
        [self.state[0], self.state[1], self.state[2]]
    }

    /// Get current velocity estimate
    pub fn velocity(&self) -> [f64; 3] {
        [self.state[3], self.state[4], self.state[5]]
    }

    /// Get position uncertainty (standard deviation)
    pub fn position_uncertainty(&self) -> [f64; 3] {
        [
            self.covariance[0][0].sqrt(),
            self.covariance[1][1].sqrt(),
            self.covariance[2][2].sqrt(),
        ]
    }

    /// Get velocity uncertainty (standard deviation)
    pub fn velocity_uncertainty(&self) -> [f64; 3] {
        [
            self.covariance[3][3].sqrt(),
            self.covariance[4][4].sqrt(),
            self.covariance[5][5].sqrt(),
        ]
    }

    // Matrix operations

    fn state_transition_matrix(&self, dt: f64) -> CovarianceMatrix {
        let mut f = [[0.0; 6]; 6];

        // Identity for position and velocity
        for i in 0..6 {
            f[i][i] = 1.0;
        }

        // Position update from velocity: x += vx * dt
        f[0][3] = dt;
        f[1][4] = dt;
        f[2][5] = dt;

        f
    }

    fn measurement_matrix(&self) -> [[f64; 6]; 3] {
        let mut h = [[0.0; 6]; 3];
        // We measure position only
        h[0][0] = 1.0;
        h[1][1] = 1.0;
        h[2][2] = 1.0;
        h
    }

    fn measurement_prediction(&self, h: &[[f64; 6]; 3]) -> [f64; 3] {
        let mut result = [0.0; 3];
        for i in 0..3 {
            for j in 0..6 {
                result[i] += h[i][j] * self.state[j];
            }
        }
        result
    }

    fn innovation_covariance(&self, h: &[[f64; 6]; 3]) -> [[f64; 3]; 3] {
        // S = H * P * H^T + R

        // H * P
        let mut h_p = [[0.0; 6]; 3];
        for i in 0..3 {
            for j in 0..6 {
                for k in 0..6 {
                    h_p[i][j] += h[i][k] * self.covariance[k][j];
                }
            }
        }

        // (H * P) * H^T
        let mut h_p_ht = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..6 {
                    h_p_ht[i][j] += h_p[i][k] * h[j][k];
                }
            }
        }

        // Add R
        for i in 0..3 {
            for j in 0..3 {
                h_p_ht[i][j] += self.measurement_noise[i][j];
            }
        }

        h_p_ht
    }

    fn compute_kalman_gain(&self, h: &[[f64; 6]; 3], s: &[[f64; 3]; 3]) -> [[f64; 3]; 6] {
        // K = P * H^T * S^(-1)

        // P * H^T
        let mut p_ht = [[0.0; 3]; 6];
        for i in 0..6 {
            for j in 0..3 {
                for k in 0..6 {
                    p_ht[i][j] += self.covariance[i][k] * h[j][k];
                }
            }
        }

        // S^(-1) (3x3 matrix inversion)
        let s_inv = self.invert_3x3(s);

        // (P * H^T) * S^(-1)
        let mut k = [[0.0; 3]; 6];
        for i in 0..6 {
            for j in 0..3 {
                for m in 0..3 {
                    k[i][j] += p_ht[i][m] * s_inv[m][j];
                }
            }
        }

        k
    }

    fn invert_3x3(&self, matrix: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
        let m = matrix;

        // Calculate determinant
        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

        if det.abs() < 1e-10 {
            // Singular matrix, return identity
            return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        }

        let inv_det = 1.0 / det;

        // Calculate inverse using cofactor method
        [
            [
                inv_det * (m[1][1] * m[2][2] - m[1][2] * m[2][1]),
                inv_det * (m[0][2] * m[2][1] - m[0][1] * m[2][2]),
                inv_det * (m[0][1] * m[1][2] - m[0][2] * m[1][1]),
            ],
            [
                inv_det * (m[1][2] * m[2][0] - m[1][0] * m[2][2]),
                inv_det * (m[0][0] * m[2][2] - m[0][2] * m[2][0]),
                inv_det * (m[0][2] * m[1][0] - m[0][0] * m[1][2]),
            ],
            [
                inv_det * (m[1][0] * m[2][1] - m[1][1] * m[2][0]),
                inv_det * (m[0][1] * m[2][0] - m[0][0] * m[2][1]),
                inv_det * (m[0][0] * m[1][1] - m[0][1] * m[1][0]),
            ],
        ]
    }

    fn matrix_vector_multiply_6(&self, matrix: &CovarianceMatrix, vector: &StateVector) -> StateVector {
        let mut result = [0.0; 6];
        for i in 0..6 {
            for j in 0..6 {
                result[i] += matrix[i][j] * vector[j];
            }
        }
        result
    }

    fn matrix_multiply_6(&self, a: &CovarianceMatrix, b: &CovarianceMatrix) -> CovarianceMatrix {
        let mut result = [[0.0; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                for k in 0..6 {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        result
    }

    fn matrix_multiply_transpose_6(&self, a: &CovarianceMatrix, b: &CovarianceMatrix) -> CovarianceMatrix {
        let mut result = [[0.0; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                for k in 0..6 {
                    result[i][j] += a[i][k] * b[j][k]; // Note: b[j][k] not b[k][j]
                }
            }
        }
        result
    }

    fn matrix_add_6(&self, a: &CovarianceMatrix, b: &CovarianceMatrix) -> CovarianceMatrix {
        let mut result = [[0.0; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                result[i][j] = a[i][j] + b[i][j];
            }
        }
        result
    }

    fn matrix_subtract_6(&self, a: &CovarianceMatrix, b: &CovarianceMatrix) -> CovarianceMatrix {
        let mut result = [[0.0; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                result[i][j] = a[i][j] - b[i][j];
            }
        }
        result
    }

    fn identity_6(&self) -> CovarianceMatrix {
        let mut identity = [[0.0; 6]; 6];
        for i in 0..6 {
            identity[i][i] = 1.0;
        }
        identity
    }

    fn kalman_times_measurement(&self, k: &[[f64; 3]; 6], h: &[[f64; 6]; 3]) -> CovarianceMatrix {
        let mut result = [[0.0; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                for m in 0..3 {
                    result[i][j] += k[i][m] * h[m][j];
                }
            }
        }
        result
    }
}

/// Multi-sensor fusion using Kalman filtering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiSensorFusion {
    /// Kalman filter for each track
    filters: Vec<(String, KalmanFilter)>,
}

impl MultiSensorFusion {
    pub fn new() -> Self {
        Self {
            filters: Vec::new(),
        }
    }

    /// Fuse measurements from multiple sensors for a single track
    pub fn fuse_measurements(
        &mut self,
        track_id: &str,
        measurements: &[(MeasurementVector, f64)], // (measurement, timestamp)
    ) -> FusionResult {
        // Find existing filter or create new one
        let filter_idx = self.filters.iter().position(|(id, _)| id == track_id);

        let mut filter = if let Some(idx) = filter_idx {
            self.filters[idx].1.clone()
        } else {
            // Initialize with first measurement
            if measurements.is_empty() {
                return FusionResult::default();
            }

            let first_meas = measurements[0].0;
            KalmanFilter::new(
                first_meas,
                [0.0, 0.0, 0.0],
                10.0,  // Initial position uncertainty
                5.0,   // Initial velocity uncertainty
                1.0,   // Measurement noise
            )
        };

        // Process all measurements in time order
        let mut last_time = filter.last_update_time;
        for (measurement, timestamp) in measurements {
            let dt = timestamp - last_time;
            if dt > 0.0 {
                filter.predict(dt);
            }
            filter.update(*measurement);
            last_time = *timestamp;
        }

        // Update or insert filter
        if let Some(idx) = filter_idx {
            self.filters[idx].1 = filter.clone();
        } else {
            self.filters.push((track_id.to_string(), filter.clone()));
        }

        FusionResult {
            position: filter.position(),
            velocity: filter.velocity(),
            position_uncertainty: filter.position_uncertainty(),
            velocity_uncertainty: filter.velocity_uncertainty(),
            num_sensors_fused: measurements.len(),
        }
    }

    /// Get all active tracks
    pub fn active_tracks(&self) -> Vec<String> {
        self.filters.iter().map(|(id, _)| id.clone()).collect()
    }
}

impl Default for MultiSensorFusion {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of sensor fusion
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FusionResult {
    /// Fused position estimate [x, y, z]
    pub position: [f64; 3],

    /// Fused velocity estimate [vx, vy, vz]
    pub velocity: [f64; 3],

    /// Position uncertainty (1-sigma) [σx, σy, σz]
    pub position_uncertainty: [f64; 3],

    /// Velocity uncertainty (1-sigma) [σvx, σvy, σvz]
    pub velocity_uncertainty: [f64; 3],

    /// Number of sensors fused
    pub num_sensors_fused: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kalman_predict() {
        let mut kf = KalmanFilter::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            1.0,
            0.1,
            0.5,
        );

        kf.predict(1.0);

        // After 1 second with vx=1, position should be [1, 0, 0]
        let pos = kf.position();
        assert!((pos[0] - 1.0).abs() < 1e-10);
        assert!(pos[1].abs() < 1e-10);
        assert!(pos[2].abs() < 1e-10);
    }

    #[test]
    fn test_kalman_update() {
        let mut kf = KalmanFilter::new(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            1.0,
            1.0,
            0.1,
        );

        // Provide measurement at [1, 0, 0]
        kf.update([1.0, 0.0, 0.0]);

        // Position should move toward measurement
        let pos = kf.position();
        assert!(pos[0] > 0.0 && pos[0] <= 1.0);
    }

    #[test]
    fn test_multi_sensor_fusion() {
        let mut fusion = MultiSensorFusion::new();

        let measurements = vec![
            ([1.0, 0.0, 0.0], 0.0),
            ([2.1, 0.1, 0.0], 1.0),
            ([3.0, -0.1, 0.0], 2.0),
        ];

        let result = fusion.fuse_measurements("track-001", &measurements);

        assert_eq!(result.num_sensors_fused, 3);
        // Should have positive velocity in x direction
        assert!(result.velocity[0] > 0.0);
    }
}
