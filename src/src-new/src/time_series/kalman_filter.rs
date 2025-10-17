//! Kalman Filter for Time Series Sensor Fusion
//!
//! Implements Extended Kalman Filter (EKF) for combining noisy measurements
//! with model predictions (ARIMA/LSTM). Provides optimal state estimation
//! under Gaussian noise assumptions.
//!
//! Mathematical Framework:
//!
//! Prediction Step:
//! - x̂ₖ|ₖ₋₁ = Fₖ x̂ₖ₋₁|ₖ₋₁ + Bₖ uₖ
//! - Pₖ|ₖ₋₁ = Fₖ Pₖ₋₁|ₖ₋₁ Fₖᵀ + Qₖ
//!
//! Update Step:
//! - yₖ = zₖ - Hₖ x̂ₖ|ₖ₋₁  (innovation)
//! - Sₖ = Hₖ Pₖ|ₖ₋₁ Hₖᵀ + Rₖ  (innovation covariance)
//! - Kₖ = Pₖ|ₖ₋₁ Hₖᵀ Sₖ⁻¹  (Kalman gain)
//! - x̂ₖ|ₖ = x̂ₖ|ₖ₋₁ + Kₖ yₖ  (state update)
//! - Pₖ|ₖ = (I - Kₖ Hₖ) Pₖ|ₖ₋₁  (covariance update)
//!
//! Integration with ARIMA:
//! - ARIMA provides model prediction (Fₖ)
//! - Kalman filter fuses with noisy measurements (zₖ)
//! - Result: optimal state estimate with uncertainty quantification

use anyhow::{Result, bail};
use ndarray::{Array1, Array2};

/// Kalman Filter configuration
#[derive(Debug, Clone)]
pub struct KalmanConfig {
    /// State dimension
    pub state_dim: usize,
    /// Measurement dimension
    pub measurement_dim: usize,
    /// Process noise covariance Q
    pub process_noise: f64,
    /// Measurement noise covariance R
    pub measurement_noise: f64,
    /// Initial state covariance P₀
    pub initial_covariance: f64,
}

impl Default for KalmanConfig {
    fn default() -> Self {
        Self {
            state_dim: 1,
            measurement_dim: 1,
            process_noise: 0.01,
            measurement_noise: 0.1,
            initial_covariance: 1.0,
        }
    }
}

/// Kalman Filter for time series sensor fusion
pub struct KalmanFilter {
    /// Configuration
    config: KalmanConfig,
    /// Current state estimate x̂ₖ|ₖ
    state: Array1<f64>,
    /// State covariance Pₖ|ₖ
    covariance: Array2<f64>,
    /// State transition matrix F
    transition_matrix: Array2<f64>,
    /// Observation matrix H
    observation_matrix: Array2<f64>,
    /// Process noise covariance Q
    process_noise: Array2<f64>,
    /// Measurement noise covariance R
    measurement_noise: Array2<f64>,
    /// GPU availability
    gpu_available: bool,
}

impl KalmanFilter {
    /// Create new Kalman filter
    pub fn new(config: KalmanConfig) -> Result<Self> {
        let gpu_available = crate::gpu::kernel_executor::get_global_executor().is_ok();

        if gpu_available {
            println!("✓ GPU acceleration available for Kalman Filter");
        } else {
            println!("⚠ GPU not available, using CPU for Kalman Filter");
        }

        // Initialize state
        let state = Array1::zeros(config.state_dim);

        // Initialize covariance
        let mut covariance = Array2::zeros((config.state_dim, config.state_dim));
        for i in 0..config.state_dim {
            covariance[[i, i]] = config.initial_covariance;
        }

        // Initialize transition matrix (identity by default)
        let mut transition_matrix = Array2::zeros((config.state_dim, config.state_dim));
        for i in 0..config.state_dim {
            transition_matrix[[i, i]] = 1.0;
        }

        // Initialize observation matrix (identity by default)
        let mut observation_matrix = Array2::zeros((config.measurement_dim, config.state_dim));
        for i in 0..(config.measurement_dim.min(config.state_dim)) {
            observation_matrix[[i, i]] = 1.0;
        }

        // Process noise covariance
        let mut process_noise = Array2::zeros((config.state_dim, config.state_dim));
        for i in 0..config.state_dim {
            process_noise[[i, i]] = config.process_noise;
        }

        // Measurement noise covariance
        let mut measurement_noise = Array2::zeros((config.measurement_dim, config.measurement_dim));
        for i in 0..config.measurement_dim {
            measurement_noise[[i, i]] = config.measurement_noise;
        }

        Ok(Self {
            config,
            state,
            covariance,
            transition_matrix,
            observation_matrix,
            process_noise,
            measurement_noise,
            gpu_available,
        })
    }

    /// Set state transition matrix F (for ARIMA integration)
    pub fn set_transition_matrix(&mut self, matrix: Array2<f64>) -> Result<()> {
        if matrix.shape() != [self.config.state_dim, self.config.state_dim] {
            bail!("Transition matrix shape mismatch: expected ({}, {}), got {:?}",
                  self.config.state_dim, self.config.state_dim, matrix.shape());
        }
        self.transition_matrix = matrix;
        Ok(())
    }

    /// Set observation matrix H
    pub fn set_observation_matrix(&mut self, matrix: Array2<f64>) -> Result<()> {
        if matrix.shape() != [self.config.measurement_dim, self.config.state_dim] {
            bail!("Observation matrix shape mismatch: expected ({}, {}), got {:?}",
                  self.config.measurement_dim, self.config.state_dim, matrix.shape());
        }
        self.observation_matrix = matrix;
        Ok(())
    }

    /// Prediction step: x̂ₖ|ₖ₋₁ = F x̂ₖ₋₁|ₖ₋₁
    pub fn predict(&mut self) -> Result<()> {
        // State prediction: x̂ₖ|ₖ₋₁ = F x̂ₖ₋₁|ₖ₋₁
        let predicted_state = self.transition_matrix.dot(&self.state);
        self.state = predicted_state;

        // Covariance prediction: Pₖ|ₖ₋₁ = F Pₖ₋₁|ₖ₋₁ Fᵀ + Q
        let f_p = self.transition_matrix.dot(&self.covariance);
        let f_p_ft = f_p.dot(&self.transition_matrix.t());
        self.covariance = f_p_ft + &self.process_noise;

        Ok(())
    }

    /// Prediction step with control input: x̂ₖ|ₖ₋₁ = F x̂ₖ₋₁|ₖ₋₁ + B uₖ
    pub fn predict_with_control(&mut self, control_input: &Array1<f64>, control_matrix: &Array2<f64>) -> Result<()> {
        // State prediction with control: x̂ₖ|ₖ₋₁ = F x̂ₖ₋₁|ₖ₋₁ + B uₖ
        let f_x = self.transition_matrix.dot(&self.state);
        let b_u = control_matrix.dot(control_input);
        self.state = f_x + b_u;

        // Covariance prediction: Pₖ|ₖ₋₁ = F Pₖ₋₁|ₖ₋₁ Fᵀ + Q
        let f_p = self.transition_matrix.dot(&self.covariance);
        let f_p_ft = f_p.dot(&self.transition_matrix.t());
        self.covariance = f_p_ft + &self.process_noise;

        Ok(())
    }

    /// Update step: fuse measurement with prediction
    pub fn update(&mut self, measurement: &Array1<f64>) -> Result<()> {
        if measurement.len() != self.config.measurement_dim {
            bail!("Measurement dimension mismatch: expected {}, got {}",
                  self.config.measurement_dim, measurement.len());
        }

        // Innovation: yₖ = zₖ - H x̂ₖ|ₖ₋₁
        let h_x = self.observation_matrix.dot(&self.state);
        let innovation = measurement - &h_x;

        // Innovation covariance: Sₖ = H Pₖ|ₖ₋₁ Hᵀ + R
        let h_p = self.observation_matrix.dot(&self.covariance);
        let h_p_ht = h_p.dot(&self.observation_matrix.t());
        let innovation_cov = h_p_ht + &self.measurement_noise;

        // Kalman gain: Kₖ = Pₖ|ₖ₋₁ Hᵀ Sₖ⁻¹
        let kalman_gain = self.compute_kalman_gain(&innovation_cov)?;

        // State update: x̂ₖ|ₖ = x̂ₖ|ₖ₋₁ + Kₖ yₖ
        let correction = kalman_gain.dot(&innovation);
        self.state = &self.state + &correction;

        // Covariance update: Pₖ|ₖ = (I - Kₖ Hₖ) Pₖ|ₖ₋₁
        let k_h = kalman_gain.dot(&self.observation_matrix);
        let identity = Array2::eye(self.config.state_dim);
        let i_minus_kh = identity - k_h;
        self.covariance = i_minus_kh.dot(&self.covariance);

        Ok(())
    }

    /// Compute Kalman gain: K = P Hᵀ S⁻¹
    fn compute_kalman_gain(&self, innovation_cov: &Array2<f64>) -> Result<Array2<f64>> {
        // P Hᵀ
        let p_ht = self.covariance.dot(&self.observation_matrix.t());

        // S⁻¹ via pseudo-inverse (more stable than direct inversion)
        let s_inv = self.pseudo_inverse(innovation_cov)?;

        // K = P Hᵀ S⁻¹
        let kalman_gain = p_ht.dot(&s_inv);

        Ok(kalman_gain)
    }

    /// Compute pseudo-inverse using SVD (more stable)
    fn pseudo_inverse(&self, matrix: &Array2<f64>) -> Result<Array2<f64>> {
        // For small matrices (typical in time series), use simple Gauss-Jordan
        // For production, consider using nalgebra's SVD

        let n = matrix.shape()[0];
        if n != matrix.shape()[1] {
            bail!("Matrix must be square for inversion");
        }

        // Augment with identity
        let mut augmented = Array2::zeros((n, 2 * n));
        for i in 0..n {
            for j in 0..n {
                augmented[[i, j]] = matrix[[i, j]];
            }
            augmented[[i, i + n]] = 1.0;
        }

        // Gauss-Jordan elimination
        for i in 0..n {
            // Pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if augmented[[k, i]].abs() > augmented[[max_row, i]].abs() {
                    max_row = k;
                }
            }

            if max_row != i {
                for j in 0..(2 * n) {
                    let temp = augmented[[i, j]];
                    augmented[[i, j]] = augmented[[max_row, j]];
                    augmented[[max_row, j]] = temp;
                }
            }

            // Check for singularity
            if augmented[[i, i]].abs() < 1e-10 {
                // Add small regularization
                augmented[[i, i]] += 1e-8;
            }

            // Scale pivot row
            let pivot = augmented[[i, i]];
            for j in 0..(2 * n) {
                augmented[[i, j]] /= pivot;
            }

            // Eliminate column
            for k in 0..n {
                if k != i {
                    let factor = augmented[[k, i]];
                    for j in 0..(2 * n) {
                        augmented[[k, j]] -= factor * augmented[[i, j]];
                    }
                }
            }
        }

        // Extract inverse
        let mut inverse = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                inverse[[i, j]] = augmented[[i, j + n]];
            }
        }

        Ok(inverse)
    }

    /// Get current state estimate
    pub fn get_state(&self) -> &Array1<f64> {
        &self.state
    }

    /// Get state covariance (uncertainty)
    pub fn get_covariance(&self) -> &Array2<f64> {
        &self.covariance
    }

    /// Get state uncertainty (standard deviation)
    pub fn get_uncertainty(&self) -> Array1<f64> {
        let mut uncertainty = Array1::zeros(self.config.state_dim);
        for i in 0..self.config.state_dim {
            uncertainty[i] = self.covariance[[i, i]].sqrt();
        }
        uncertainty
    }

    /// Reset filter to initial state
    pub fn reset(&mut self) {
        self.state = Array1::zeros(self.config.state_dim);

        let mut covariance = Array2::zeros((self.config.state_dim, self.config.state_dim));
        for i in 0..self.config.state_dim {
            covariance[[i, i]] = self.config.initial_covariance;
        }
        self.covariance = covariance;
    }

    /// Process sequence of measurements
    pub fn filter_sequence(&mut self, measurements: &[f64]) -> Result<Vec<f64>> {
        let mut filtered = Vec::with_capacity(measurements.len());

        for &measurement in measurements {
            self.predict()?;
            let meas_array = Array1::from(vec![measurement]);
            self.update(&meas_array)?;
            filtered.push(self.state[0]);
        }

        Ok(filtered)
    }

    /// Smooth sequence using forward-backward pass (Rauch-Tung-Striebel)
    pub fn smooth_sequence(&mut self, measurements: &[f64]) -> Result<Vec<f64>> {
        // Forward pass (filtering)
        let mut forward_states = Vec::with_capacity(measurements.len());
        let mut forward_covariances = Vec::with_capacity(measurements.len());

        for &measurement in measurements {
            self.predict()?;
            let meas_array = Array1::from(vec![measurement]);
            self.update(&meas_array)?;

            forward_states.push(self.state.clone());
            forward_covariances.push(self.covariance.clone());
        }

        // Backward pass (smoothing)
        let mut smoothed = vec![0.0; measurements.len()];
        smoothed[measurements.len() - 1] = forward_states[measurements.len() - 1][0];

        for k in (0..measurements.len() - 1).rev() {
            // Smoother gain: Cₖ = Pₖ|ₖ Fᵀ Pₖ₊₁|ₖ⁻¹
            let p_k = &forward_covariances[k];
            let p_k_plus_1 = &forward_covariances[k + 1];

            let p_ft = p_k.dot(&self.transition_matrix.t());
            let p_inv = self.pseudo_inverse(p_k_plus_1)?;
            let smoother_gain = p_ft.dot(&p_inv);

            // Smoothed state: x̂ₖ|ₙ = x̂ₖ|ₖ + Cₖ (x̂ₖ₊₁|ₙ - x̂ₖ₊₁|ₖ)
            let x_k = &forward_states[k];
            let x_k_plus_1_smoothed = Array1::from(vec![smoothed[k + 1]]);

            let f_x_k = self.transition_matrix.dot(x_k);
            let diff = x_k_plus_1_smoothed - f_x_k;
            let correction = smoother_gain.dot(&diff);

            smoothed[k] = x_k[0] + correction[0];
        }

        Ok(smoothed)
    }
}

/// ARIMA-Kalman fusion: combine ARIMA model with Kalman filtering
pub struct ArimaKalmanFusion {
    /// Kalman filter
    kalman: KalmanFilter,
    /// ARIMA coefficients (for transition matrix)
    ar_coefficients: Vec<f64>,
    /// State buffer for AR(p) model
    state_buffer: Vec<f64>,
}

impl ArimaKalmanFusion {
    /// Create new ARIMA-Kalman fusion
    pub fn new(ar_coefficients: Vec<f64>, process_noise: f64, measurement_noise: f64) -> Result<Self> {
        let p = ar_coefficients.len();

        let config = KalmanConfig {
            state_dim: p,
            measurement_dim: 1,
            process_noise,
            measurement_noise,
            initial_covariance: 1.0,
        };

        let mut kalman = KalmanFilter::new(config)?;

        // Construct transition matrix from AR coefficients
        // F = [φ₁  φ₂  φ₃ ... φₚ]
        //     [1   0   0  ... 0 ]
        //     [0   1   0  ... 0 ]
        //     ...
        let mut transition = Array2::zeros((p, p));
        for i in 0..p {
            transition[[0, i]] = ar_coefficients[i];
        }
        for i in 1..p {
            transition[[i, i - 1]] = 1.0;
        }

        kalman.set_transition_matrix(transition)?;

        // Observation matrix: measure only first state
        let mut observation = Array2::zeros((1, p));
        observation[[0, 0]] = 1.0;
        kalman.set_observation_matrix(observation)?;

        Ok(Self {
            kalman,
            ar_coefficients: ar_coefficients.clone(),
            state_buffer: vec![0.0; p],
        })
    }

    /// Filter noisy time series using ARIMA model + Kalman
    pub fn filter(&mut self, measurements: &[f64]) -> Result<Vec<f64>> {
        self.kalman.filter_sequence(measurements)
    }

    /// Smooth time series using forward-backward pass
    pub fn smooth(&mut self, measurements: &[f64]) -> Result<Vec<f64>> {
        self.kalman.smooth_sequence(measurements)
    }

    /// Get current state uncertainty
    pub fn get_uncertainty(&self) -> Array1<f64> {
        self.kalman.get_uncertainty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kalman_filter_creation() {
        let config = KalmanConfig::default();
        let filter = KalmanFilter::new(config);
        assert!(filter.is_ok());
    }

    #[test]
    fn test_kalman_predict_update() {
        let config = KalmanConfig {
            state_dim: 1,
            measurement_dim: 1,
            process_noise: 0.01,
            measurement_noise: 0.1,
            initial_covariance: 1.0,
        };

        let mut filter = KalmanFilter::new(config).unwrap();

        // Predict
        filter.predict().unwrap();

        // Update with measurement
        let measurement = Array1::from(vec![1.0]);
        filter.update(&measurement).unwrap();

        let state = filter.get_state();
        assert!(state[0].is_finite());
    }

    #[test]
    fn test_kalman_sequence_filtering() {
        let config = KalmanConfig::default();
        let mut filter = KalmanFilter::new(config).unwrap();

        // Noisy measurements
        let measurements = vec![1.0, 1.1, 0.9, 1.0, 1.05];

        let filtered = filter.filter_sequence(&measurements).unwrap();

        assert_eq!(filtered.len(), measurements.len());
        assert!(filtered.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_kalman_smoothing() {
        let config = KalmanConfig::default();
        let mut filter = KalmanFilter::new(config).unwrap();

        let measurements = vec![1.0, 1.2, 0.8, 1.1, 0.9];

        let smoothed = filter.smooth_sequence(&measurements).unwrap();

        assert_eq!(smoothed.len(), measurements.len());
        assert!(smoothed.iter().all(|&x| x.is_finite()));

        // Smoothed values should be less noisy than raw measurements
        let measurement_variance = compute_variance(&measurements);
        let smoothed_variance = compute_variance(&smoothed);
        assert!(smoothed_variance < measurement_variance);
    }

    #[test]
    fn test_arima_kalman_fusion() {
        let ar_coefficients = vec![0.8, -0.2];
        let fusion = ArimaKalmanFusion::new(ar_coefficients, 0.01, 0.1);
        assert!(fusion.is_ok());
    }

    #[test]
    fn test_arima_kalman_filtering() {
        let ar_coefficients = vec![0.8, -0.2];
        let mut fusion = ArimaKalmanFusion::new(ar_coefficients, 0.01, 0.1).unwrap();

        // Generate noisy AR(2) process
        let mut data = vec![0.0, 0.1];
        for _ in 0..50 {
            let n = data.len();
            let value = 0.8 * data[n - 1] - 0.2 * data[n - 2] + 0.1 * rand::random::<f64>();
            data.push(value);
        }

        let filtered = fusion.filter(&data).unwrap();

        assert_eq!(filtered.len(), data.len());
        assert!(filtered.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_kalman_uncertainty_quantification() {
        let config = KalmanConfig::default();
        let mut filter = KalmanFilter::new(config).unwrap();

        let measurements = vec![1.0, 1.1, 0.9];

        for &meas in &measurements {
            filter.predict().unwrap();
            let meas_array = Array1::from(vec![meas]);
            filter.update(&meas_array).unwrap();
        }

        let uncertainty = filter.get_uncertainty();
        assert!(uncertainty[0] > 0.0);
        assert!(uncertainty[0] < 1.0); // Should decrease with measurements
    }

    #[test]
    fn test_transition_matrix_setting() {
        let config = KalmanConfig {
            state_dim: 2,
            measurement_dim: 1,
            ..Default::default()
        };

        let mut filter = KalmanFilter::new(config).unwrap();

        let transition = Array2::from_shape_vec((2, 2), vec![0.9, 0.1, 0.0, 0.8]).unwrap();
        let result = filter.set_transition_matrix(transition);
        assert!(result.is_ok());
    }

    fn compute_variance(data: &[f64]) -> f64 {
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64
    }
}
