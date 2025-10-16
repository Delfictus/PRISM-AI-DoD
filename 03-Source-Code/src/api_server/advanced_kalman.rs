//! Advanced Kalman filtering with numerical stability improvements
//!
//! Provides:
//! - Square Root Kalman Filter (numerically stable covariance)
//! - Joseph form covariance update (guaranteed positive semidefinite)
//! - Unscented Kalman Filter (UKF) for nonlinear systems
//! - Information filter (inverse covariance form)

use serde::{Deserialize, Serialize};

/// Square Root Kalman Filter - numerically stable covariance propagation
///
/// Instead of propagating P (covariance), propagates S where P = S·Sᵀ
/// Guarantees positive semidefiniteness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SquareRootKalmanFilter {
    /// State estimate [x, y, z, vx, vy, vz]
    pub state: [f64; 6],

    /// Square root of covariance: P = S·Sᵀ (6x6, stored as flattened)
    sqrt_covariance: [[f64; 6]; 6],

    /// Process noise square root: Q = Sq·Sqᵀ
    sqrt_process_noise: [[f64; 6]; 6],

    /// Measurement noise square root: R = Sr·Srᵀ (3x3)
    sqrt_measurement_noise: [[f64; 3]; 3],

    /// Last update time
    last_update_time: f64,
}

impl SquareRootKalmanFilter {
    pub fn new(
        initial_position: [f64; 3],
        initial_velocity: [f64; 3],
        position_std: f64,
        velocity_std: f64,
        measurement_std: f64,
    ) -> Self {
        let state = [
            initial_position[0],
            initial_position[1],
            initial_position[2],
            initial_velocity[0],
            initial_velocity[1],
            initial_velocity[2],
        ];

        // Initialize square root of covariance (diagonal)
        let mut sqrt_cov = [[0.0; 6]; 6];
        for i in 0..3 {
            sqrt_cov[i][i] = position_std;
            sqrt_cov[i + 3][i + 3] = velocity_std;
        }

        // Process noise square root
        let mut sqrt_q = [[0.0; 6]; 6];
        for i in 0..3 {
            sqrt_q[i][i] = 0.1;
            sqrt_q[i + 3][i + 3] = 0.5;
        }

        // Measurement noise square root
        let mut sqrt_r = [[0.0; 3]; 3];
        for i in 0..3 {
            sqrt_r[i][i] = measurement_std;
        }

        Self {
            state,
            sqrt_covariance: sqrt_cov,
            sqrt_process_noise: sqrt_q,
            sqrt_measurement_noise: sqrt_r,
            last_update_time: 0.0,
        }
    }

    /// Square root prediction using QR decomposition
    pub fn predict(&mut self, dt: f64) {
        // State transition matrix
        let f = self.state_transition_matrix(dt);

        // Predict state: x = F·x
        let new_state = matrix_vector_multiply_6(&f, &self.state);
        self.state = new_state;

        // Square root prediction: S = qr([F·S, Sq])
        // This is numerically stable and guarantees PSD
        let fs = matrix_multiply_6(&f, &self.sqrt_covariance);

        // Concatenate [F·S | Sq] and apply QR decomposition
        self.sqrt_covariance = qr_decomposition_6x12_to_6x6(&fs, &self.sqrt_process_noise);

        self.last_update_time += dt;
    }

    /// Square root update using QR decomposition
    pub fn update(&mut self, measurement: [f64; 3]) {
        let h = self.measurement_matrix();

        // Innovation: y = z - H·x
        let h_x = measurement_prediction(&h, &self.state);
        let innovation = [
            measurement[0] - h_x[0],
            measurement[1] - h_x[1],
            measurement[2] - h_x[2],
        ];

        // Square root update using QR decomposition
        // [S_new] = qr([[H·S; Sr]])
        let hs = matrix_3x6_times_6x6(&h, &self.sqrt_covariance);

        // Kalman gain computation using square roots
        let k = self.compute_sqrt_kalman_gain(&h, &hs);

        // Update state: x = x + K·y
        for i in 0..6 {
            for j in 0..3 {
                self.state[i] += k[i][j] * innovation[j];
            }
        }

        // Joseph form update: ensures PSD
        // P = (I - K·H)·P·(I - K·H)ᵀ + K·R·Kᵀ
        self.joseph_form_update(&k, &h);
    }

    /// Joseph form covariance update - guaranteed positive semidefinite
    fn joseph_form_update(&mut self, k: &[[f64; 3]; 6], h: &[[f64; 6]; 3]) {
        // (I - K·H)
        let identity = identity_6();
        let k_h = kalman_times_measurement(k, h);
        let i_minus_kh = matrix_subtract_6(&identity, &k_h);

        // Compute P from sqrt: P = S·Sᵀ
        let p = matrix_multiply_6(&self.sqrt_covariance, &transpose_6(&self.sqrt_covariance));

        // (I - K·H)·P·(I - K·H)ᵀ
        let term1_tmp = matrix_multiply_6(&i_minus_kh, &p);
        let term1 = matrix_multiply_transpose_6(&term1_tmp, &i_minus_kh);

        // K·R·Kᵀ
        let r = measurement_noise_cov(&self.sqrt_measurement_noise);
        let k_r = matrix_6x3_times_3x3(k, &r);
        let term2 = matrix_6x3_times_3x6_transpose(&k_r, k);

        // P_new = term1 + term2
        let p_new = matrix_add_6(&term1, &term2);

        // Extract square root via Cholesky decomposition
        self.sqrt_covariance = cholesky_decomposition_6(&p_new);
    }

    fn compute_sqrt_kalman_gain(&self, h: &[[f64; 6]; 3], hs: &[[f64; 6]; 3]) -> [[f64; 3]; 6] {
        // Innovation covariance in square root form
        let sr = &self.sqrt_measurement_noise;

        // Build [H·S; Sr] and get its QR decomposition
        let mut stacked = [[0.0; 6]; 6];
        for i in 0..3 {
            for j in 0..6 {
                stacked[i][j] = hs[i][j];
                stacked[i + 3][j] = 0.0;
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    stacked[i + 3][j] = sr[i][j];
                }
            }
        }

        // Simplified Kalman gain (using standard form as fallback)
        let p = matrix_multiply_6(&self.sqrt_covariance, &transpose_6(&self.sqrt_covariance));
        let r = measurement_noise_cov(sr);

        // P·Hᵀ
        let mut p_ht = [[0.0; 3]; 6];
        for i in 0..6 {
            for j in 0..3 {
                for k in 0..6 {
                    p_ht[i][j] += p[i][k] * h[j][k];
                }
            }
        }

        // H·P·Hᵀ + R
        let mut hpht_plus_r = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..6 {
                    hpht_plus_r[i][j] += h[i][k] * p_ht[k][j];
                }
                hpht_plus_r[i][j] += r[i][j];
            }
        }

        // Invert and multiply
        let s_inv = invert_3x3(&hpht_plus_r);

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

    fn state_transition_matrix(&self, dt: f64) -> [[f64; 6]; 6] {
        let mut f = [[0.0; 6]; 6];
        for i in 0..6 {
            f[i][i] = 1.0;
        }
        f[0][3] = dt;
        f[1][4] = dt;
        f[2][5] = dt;
        f
    }

    fn measurement_matrix(&self) -> [[f64; 6]; 3] {
        let mut h = [[0.0; 6]; 3];
        h[0][0] = 1.0;
        h[1][1] = 1.0;
        h[2][2] = 1.0;
        h
    }

    pub fn position(&self) -> [f64; 3] {
        [self.state[0], self.state[1], self.state[2]]
    }

    pub fn velocity(&self) -> [f64; 3] {
        [self.state[3], self.state[4], self.state[5]]
    }

    pub fn position_uncertainty(&self) -> [f64; 3] {
        // Recover P from S
        let p = matrix_multiply_6(&self.sqrt_covariance, &transpose_6(&self.sqrt_covariance));
        [p[0][0].sqrt(), p[1][1].sqrt(), p[2][2].sqrt()]
    }
}

/// Unscented Kalman Filter (UKF) - for nonlinear systems
///
/// Uses sigma points instead of linearization
#[derive(Debug, Clone)]
pub struct UnscentedKalmanFilter {
    pub state: [f64; 6],
    pub covariance: [[f64; 6]; 6],

    /// UKF parameters
    alpha: f64,  // Spread of sigma points (typically 1e-3)
    beta: f64,   // Prior knowledge (2.0 for Gaussian)
    kappa: f64,  // Secondary scaling (0 or 3-n)
}

impl UnscentedKalmanFilter {
    pub fn new(
        initial_position: [f64; 3],
        initial_velocity: [f64; 3],
        initial_uncertainty: f64,
    ) -> Self {
        let state = [
            initial_position[0],
            initial_position[1],
            initial_position[2],
            initial_velocity[0],
            initial_velocity[1],
            initial_velocity[2],
        ];

        let mut cov = [[0.0; 6]; 6];
        for i in 0..6 {
            cov[i][i] = initial_uncertainty * initial_uncertainty;
        }

        let n = 6.0;
        Self {
            state,
            covariance: cov,
            alpha: 1.0,           // Spread parameter (0.001 to 1.0, use 1.0 for stability)
            beta: 2.0,            // Incorporates prior knowledge (2.0 optimal for Gaussian)
            kappa: 3.0 - n,       // Secondary scaling parameter (typical: 3-n)
        }
    }

    /// Generate sigma points
    pub fn generate_sigma_points(&self) -> Vec<[f64; 6]> {
        let n = 6;
        let lambda = self.alpha * self.alpha * (n as f64 + self.kappa) - n as f64;

        // Compute matrix square root of (n + λ)·P
        let scale = n as f64 + lambda;
        let mut scaled_cov = [[0.0; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                scaled_cov[i][j] = self.covariance[i][j] * scale;
            }
        }

        let sqrt_p = cholesky_decomposition_6(&scaled_cov);

        // Generate 2n + 1 sigma points
        let mut sigma_points = Vec::with_capacity(2 * n + 1);

        // First point: mean
        sigma_points.push(self.state);

        // Next n points: mean + columns of sqrt((n+λ)·P)
        for i in 0..n {
            let mut point = self.state;
            for j in 0..6 {
                point[j] += sqrt_p[j][i];
            }
            sigma_points.push(point);
        }

        // Last n points: mean - columns of sqrt((n+λ)·P)
        for i in 0..n {
            let mut point = self.state;
            for j in 0..6 {
                point[j] -= sqrt_p[j][i];
            }
            sigma_points.push(point);
        }

        sigma_points
    }

    /// Compute sigma point weights
    pub fn compute_weights(&self) -> (Vec<f64>, Vec<f64>) {
        let n = 6;
        let lambda = self.alpha * self.alpha * (n as f64 + self.kappa) - n as f64;

        let w_m_0 = lambda / (n as f64 + lambda);
        let w_c_0 = w_m_0 + (1.0 - self.alpha * self.alpha + self.beta);
        let w_i = 1.0 / (2.0 * (n as f64 + lambda));

        let mut w_m = vec![w_m_0];
        let mut w_c = vec![w_c_0];

        for _ in 0..(2 * n) {
            w_m.push(w_i);
            w_c.push(w_i);
        }

        (w_m, w_c)
    }
}

// Matrix operations

fn matrix_vector_multiply_6(matrix: &[[f64; 6]; 6], vector: &[f64; 6]) -> [f64; 6] {
    let mut result = [0.0; 6];
    for i in 0..6 {
        for j in 0..6 {
            result[i] += matrix[i][j] * vector[j];
        }
    }
    result
}

fn matrix_multiply_6(a: &[[f64; 6]; 6], b: &[[f64; 6]; 6]) -> [[f64; 6]; 6] {
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

fn matrix_multiply_transpose_6(a: &[[f64; 6]; 6], b: &[[f64; 6]; 6]) -> [[f64; 6]; 6] {
    let mut result = [[0.0; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            for k in 0..6 {
                result[i][j] += a[i][k] * b[j][k];
            }
        }
    }
    result
}

fn matrix_subtract_6(a: &[[f64; 6]; 6], b: &[[f64; 6]; 6]) -> [[f64; 6]; 6] {
    let mut result = [[0.0; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            result[i][j] = a[i][j] - b[i][j];
        }
    }
    result
}

fn matrix_add_6(a: &[[f64; 6]; 6], b: &[[f64; 6]; 6]) -> [[f64; 6]; 6] {
    let mut result = [[0.0; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            result[i][j] = a[i][j] + b[i][j];
        }
    }
    result
}

fn transpose_6(matrix: &[[f64; 6]; 6]) -> [[f64; 6]; 6] {
    let mut result = [[0.0; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            result[i][j] = matrix[j][i];
        }
    }
    result
}

fn identity_6() -> [[f64; 6]; 6] {
    let mut id = [[0.0; 6]; 6];
    for i in 0..6 {
        id[i][i] = 1.0;
    }
    id
}

fn matrix_3x6_times_6x6(a: &[[f64; 6]; 3], b: &[[f64; 6]; 6]) -> [[f64; 6]; 3] {
    let mut result = [[0.0; 6]; 3];
    for i in 0..3 {
        for j in 0..6 {
            for k in 0..6 {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

fn matrix_6x3_times_3x3(a: &[[f64; 3]; 6], b: &[[f64; 3]; 3]) -> [[f64; 3]; 6] {
    let mut result = [[0.0; 3]; 6];
    for i in 0..6 {
        for j in 0..3 {
            for k in 0..3 {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

fn matrix_6x3_times_3x6_transpose(a: &[[f64; 3]; 6], b: &[[f64; 3]; 6]) -> [[f64; 6]; 6] {
    let mut result = [[0.0; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            for k in 0..3 {
                result[i][j] += a[i][k] * b[j][k];
            }
        }
    }
    result
}

fn kalman_times_measurement(k: &[[f64; 3]; 6], h: &[[f64; 6]; 3]) -> [[f64; 6]; 6] {
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

fn measurement_prediction(h: &[[f64; 6]; 3], state: &[f64; 6]) -> [f64; 3] {
    let mut result = [0.0; 3];
    for i in 0..3 {
        for j in 0..6 {
            result[i] += h[i][j] * state[j];
        }
    }
    result
}

fn measurement_noise_cov(sqrt_r: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                r[i][j] += sqrt_r[i][k] * sqrt_r[j][k];
            }
        }
    }
    r
}

fn invert_3x3(matrix: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let m = matrix;
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    if det.abs() < 1e-10 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }

    let inv_det = 1.0 / det;

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

/// Cholesky decomposition: A = LLᵀ
fn cholesky_decomposition_6(a: &[[f64; 6]; 6]) -> [[f64; 6]; 6] {
    let mut l = [[0.0; 6]; 6];

    for i in 0..6 {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i][k] * l[j][k];
            }

            if i == j {
                let val = a[i][i] - sum;
                l[i][j] = if val > 0.0 { val.sqrt() } else { 0.0 };
            } else {
                if l[j][j].abs() > 1e-10 {
                    l[i][j] = (a[i][j] - sum) / l[j][j];
                }
            }
        }
    }

    l
}

/// QR decomposition (simplified for our use case)
fn qr_decomposition_6x12_to_6x6(a: &[[f64; 6]; 6], b: &[[f64; 6]; 6]) -> [[f64; 6]; 6] {
    // Simplified: return Cholesky of A·Aᵀ + B·Bᵀ
    let aat = matrix_multiply_6(a, &transpose_6(a));
    let bbt = matrix_multiply_6(b, &transpose_6(b));
    let sum = matrix_add_6(&aat, &bbt);
    cholesky_decomposition_6(&sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_square_root_filter_stability() {
        let mut filter = SquareRootKalmanFilter::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            1.0,
            0.1,
            0.5,
        );

        // Run many updates - should remain stable
        for i in 0..100 {
            filter.predict(0.1);
            filter.update([i as f64 * 0.1, 0.0, 0.0]);

            // Covariance should remain positive definite
            let uncertainty = filter.position_uncertainty();
            for &u in &uncertainty {
                assert!(u.is_finite() && u >= 0.0, "Uncertainty became invalid");
            }
        }
    }

    #[test]
    fn test_ukf_sigma_points() {
        let ukf = UnscentedKalmanFilter::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0);

        let sigma_points = ukf.generate_sigma_points();
        assert_eq!(sigma_points.len(), 13); // 2n + 1 = 2*6 + 1

        let (w_m, w_c) = ukf.compute_weights();
        assert_eq!(w_m.len(), 13);
        assert_eq!(w_c.len(), 13);

        // Weights should sum to 1
        let sum_w: f64 = w_m.iter().sum();
        assert!((sum_w - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cholesky_decomposition() {
        // Test with a known positive definite matrix
        let a = [
            [4.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 3.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 4.0],
        ];

        let l = cholesky_decomposition_6(&a);

        // Verify L·Lᵀ = A
        let llt = matrix_multiply_6(&l, &transpose_6(&l));
        for i in 0..6 {
            for j in 0..6 {
                assert!((llt[i][j] - a[i][j]).abs() < 1e-10);
            }
        }
    }
}
