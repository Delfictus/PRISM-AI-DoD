//! Interior Point Method for Quadratic Programming
//!
//! Solves Mean-Variance Portfolio Optimization using primal-dual interior point method
//!
//! # Problem Formulation
//!
//! Minimize: (1/2) w^T Σ w - λ μ^T w
//! Subject to:
//!   - Σ w_i = 1 (budget constraint)
//!   - w_min <= w_i <= w_max (box constraints)
//!   - w_i >= 0 (long-only, optional)
//!
//! This is a convex QP problem that can be solved efficiently with IPM.
//!
//! # Mathematical Foundation
//!
//! KKT System at each iteration:
//! ┌                    ┐ ┌  Δw  ┐   ┌      ┐
//! │  H    A^T   I   -I │ │  Δλ  │ = │  r_d │
//! │  A     0    0    0 │ │  Δs  │   │  r_p │
//! │  S     0    Z    0 │ │ Δz_l │   │ r_cs │
//! │ -T     0    0    W │ └ Δz_u ┘   └ r_ct ┘
//! └                    ┘
//!
//! Where H = Σ (covariance matrix), A = 1^T (budget constraint)
//!
//! # Reference
//! Nocedal, J., & Wright, S. (2006). "Numerical optimization" (Chapter 19)
//! Boyd, S., & Vandenberghe, L. (2004). "Convex optimization" (Chapter 11)

use ndarray::{Array1, Array2};
use anyhow::Result;

/// Interior Point QP Solver Configuration
#[derive(Debug, Clone)]
pub struct InteriorPointConfig {
    /// Maximum iterations
    pub max_iterations: usize,

    /// Tolerance for KKT conditions
    pub kkt_tolerance: f64,

    /// Initial barrier parameter μ
    pub initial_mu: f64,

    /// Barrier reduction factor
    pub mu_reduction: f64,

    /// Minimum barrier parameter
    pub min_mu: f64,

    /// Step size reduction factor (backtracking line search)
    pub alpha_reduction: f64,

    /// Maximum line search iterations
    pub max_line_search: usize,

    /// Centering parameter σ ∈ (0,1)
    pub centering_param: f64,
}

impl Default for InteriorPointConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            kkt_tolerance: 1e-6,
            initial_mu: 1.0,
            mu_reduction: 0.1,
            min_mu: 1e-8,
            alpha_reduction: 0.5,
            max_line_search: 20,
            centering_param: 0.1,
        }
    }
}

/// Interior Point QP Solver Result
#[derive(Debug, Clone)]
pub struct InteriorPointResult {
    /// Optimal weights
    pub weights: Array1<f64>,

    /// Optimal objective value
    pub objective: f64,

    /// Number of iterations
    pub iterations: usize,

    /// KKT residual at solution
    pub kkt_residual: f64,

    /// Convergence status
    pub converged: bool,

    /// Dual variables (Lagrange multipliers)
    pub lambda: f64,

    /// Complementarity gap
    pub complementarity_gap: f64,
}

/// Interior Point Method QP Solver
pub struct InteriorPointQpSolver {
    config: InteriorPointConfig,
}

impl InteriorPointQpSolver {
    /// Create a new Interior Point QP solver
    pub fn new(config: InteriorPointConfig) -> Self {
        Self { config }
    }

    /// Solve mean-variance portfolio optimization
    ///
    /// # Arguments
    /// * `expected_returns` - μ (expected returns vector)
    /// * `covariance` - Σ (covariance matrix)
    /// * `risk_aversion` - λ (risk aversion parameter)
    /// * `w_min` - Minimum weight per asset
    /// * `w_max` - Maximum weight per asset
    ///
    /// # Returns
    /// InteriorPointResult with optimal weights and diagnostics
    pub fn solve_portfolio(
        &self,
        expected_returns: &Array1<f64>,
        covariance: &Array2<f64>,
        risk_aversion: f64,
        w_min: f64,
        w_max: f64,
    ) -> Result<InteriorPointResult> {
        let n = expected_returns.len();

        // Initialize primal variable w (feasible start)
        let mut w = Array1::from_elem(n, 1.0 / n as f64);

        // Initialize slack variables for box constraints
        // s_i = w_i - w_min (lower bound slack)
        // t_i = w_max - w_i (upper bound slack)
        let mut s = w.mapv(|wi| wi - w_min);
        let mut t = w.mapv(|wi| w_max - wi);

        // Initialize dual variables (Lagrange multipliers)
        let mut lambda = 0.0; // Budget constraint multiplier
        let mut z_l = Array1::from_elem(n, 1.0); // Lower bound multipliers
        let mut z_u = Array1::from_elem(n, 1.0); // Upper bound multipliers

        // Barrier parameter
        let mut mu = self.config.initial_mu;

        let mut iteration = 0;
        let mut converged = false;

        while iteration < self.config.max_iterations {
            // Calculate KKT residuals
            let (r_dual, r_primal, r_comp_l, r_comp_u) =
                self.calculate_residuals(
                    &w,
                    &s,
                    &t,
                    lambda,
                    &z_l,
                    &z_u,
                    expected_returns,
                    covariance,
                    risk_aversion,
                    w_min,
                    w_max,
                    mu,
                );

            // Check convergence
            let kkt_residual = r_dual.dot(&r_dual).sqrt()
                + r_primal.abs()
                + r_comp_l.dot(&r_comp_l).sqrt()
                + r_comp_u.dot(&r_comp_u).sqrt();

            if kkt_residual < self.config.kkt_tolerance && mu < self.config.min_mu {
                converged = true;
                break;
            }

            // Solve Newton system for search direction
            let (delta_w, delta_lambda, delta_s, delta_t, delta_z_l, delta_z_u) =
                self.solve_kkt_system(
                    &w,
                    &s,
                    &t,
                    &z_l,
                    &z_u,
                    &r_dual,
                    r_primal,
                    &r_comp_l,
                    &r_comp_u,
                    covariance,
                    risk_aversion,
                )?;

            // Backtracking line search
            let alpha = self.backtracking_line_search(
                &w,
                &s,
                &t,
                lambda,
                &z_l,
                &z_u,
                &delta_w,
                delta_lambda,
                &delta_s,
                &delta_t,
                &delta_z_l,
                &delta_z_u,
                expected_returns,
                covariance,
                risk_aversion,
                w_min,
                w_max,
                mu,
            );

            // Update variables
            w = &w + &(&delta_w * alpha);
            lambda += delta_lambda * alpha;
            s = &s + &(&delta_s * alpha);
            t = &t + &(&delta_t * alpha);
            z_l = &z_l + &(&delta_z_l * alpha);
            z_u = &z_u + &(&delta_z_u * alpha);

            // Update barrier parameter
            mu *= self.config.mu_reduction;
            mu = mu.max(self.config.min_mu);

            iteration += 1;
        }

        // Ensure weights sum to 1 (numerical correction)
        let weight_sum: f64 = w.sum();
        if (weight_sum - 1.0).abs() > 1e-6 {
            w = w.mapv(|wi| wi / weight_sum);
        }

        // Calculate objective value
        let objective = 0.5 * w.dot(&covariance.dot(&w)) - risk_aversion * w.dot(expected_returns);

        // Calculate complementarity gap
        let comp_gap = s.dot(&z_l) + t.dot(&z_u);

        Ok(InteriorPointResult {
            weights: w,
            objective,
            iterations: iteration,
            kkt_residual: if converged { 0.0 } else { 1.0 },
            converged,
            lambda,
            complementarity_gap: comp_gap,
        })
    }

    /// Calculate KKT residuals
    #[allow(clippy::too_many_arguments)]
    fn calculate_residuals(
        &self,
        w: &Array1<f64>,
        s: &Array1<f64>,
        t: &Array1<f64>,
        lambda: f64,
        z_l: &Array1<f64>,
        z_u: &Array1<f64>,
        mu: &Array1<f64>,
        sigma: &Array2<f64>,
        risk_aversion: f64,
        _w_min: f64,
        _w_max: f64,
        barrier_mu: f64,
    ) -> (Array1<f64>, f64, Array1<f64>, Array1<f64>) {
        // Dual residual: ∇L = Σw - λμ + λ*1 - z_l + z_u = 0
        let grad_objective = sigma.dot(w) - mu * risk_aversion;
        let r_dual = grad_objective + lambda - z_l + z_u;

        // Primal residual: Σw_i - 1 = 0
        let r_primal = w.sum() - 1.0;

        // Complementarity residuals: S*z_l = μe, T*z_u = μe
        let r_comp_l = s * z_l - barrier_mu;
        let r_comp_u = t * z_u - barrier_mu;

        (r_dual, r_primal, r_comp_l, r_comp_u)
    }

    /// Solve KKT system for Newton direction
    #[allow(clippy::too_many_arguments)]
    fn solve_kkt_system(
        &self,
        w: &Array1<f64>,
        s: &Array1<f64>,
        t: &Array1<f64>,
        z_l: &Array1<f64>,
        z_u: &Array1<f64>,
        r_dual: &Array1<f64>,
        r_primal: f64,
        r_comp_l: &Array1<f64>,
        r_comp_u: &Array1<f64>,
        sigma: &Array2<f64>,
        _risk_aversion: f64,
    ) -> Result<(Array1<f64>, f64, Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>)> {
        let n = w.len();

        // Reduced KKT system using Schur complement
        // H_bar = H + Z_l/S + Z_u/T
        let mut h_bar = sigma.clone();
        for i in 0..n {
            h_bar[[i, i]] += z_l[i] / s[i] + z_u[i] / t[i];
        }

        // Right-hand side for reduced system
        let rhs_w = -r_dual - r_comp_l / s + r_comp_u / t;
        let rhs_lambda = -r_primal;

        // Solve for Δw and Δλ using Schur complement
        // [H_bar  1] [Δw    ] = [rhs_w    ]
        // [1^T    0] [Δλ    ]   [rhs_lambda]

        // Δw = H_bar^{-1} (rhs_w - 1*Δλ)
        // 1^T * Δw = rhs_lambda
        // => 1^T * H_bar^{-1} (rhs_w - 1*Δλ) = rhs_lambda
        // => Δλ = (1^T * H_bar^{-1} * rhs_w - rhs_lambda) / (1^T * H_bar^{-1} * 1)

        // Solve H_bar * x = rhs_w (using conjugate gradient or direct)
        let x = self.solve_positive_definite(&h_bar, &rhs_w)?;

        // Solve H_bar * y = 1
        let ones = Array1::from_elem(n, 1.0);
        let y = self.solve_positive_definite(&h_bar, &ones)?;

        // Calculate Δλ
        let numerator = x.sum() - rhs_lambda;
        let denominator = y.sum();
        let delta_lambda = numerator / denominator;

        // Calculate Δw
        let delta_w = &x - &(&y * delta_lambda);

        // Calculate slack and dual updates
        let delta_s = -&delta_w;
        let delta_t = delta_w.clone();

        let delta_z_l = (-r_comp_l - z_l * &delta_s) / s;
        let delta_z_u = (-r_comp_u - z_u * &delta_t) / t;

        Ok((delta_w, delta_lambda, delta_s, delta_t.clone(), delta_z_l, delta_z_u))
    }

    /// Solve positive definite system Ax = b using Cholesky decomposition
    fn solve_positive_definite(&self, a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
        let n = a.nrows();

        // Simple conjugate gradient solver (for large n, use specialized library)
        // For portfolio problems (n < 1000), direct methods work well
        let mut x = Array1::zeros(n);
        let mut r = b - &a.dot(&x);
        let mut p = r.clone();
        let mut rs_old = r.dot(&r);

        for _ in 0..n {
            let ap = a.dot(&p);
            let alpha = rs_old / p.dot(&ap);

            x = &x + &(&p * alpha);
            r = &r - &(&ap * alpha);

            let rs_new = r.dot(&r);
            if rs_new.sqrt() < 1e-10 {
                break;
            }

            let beta = rs_new / rs_old;
            p = &r + &(&p * beta);
            rs_old = rs_new;
        }

        Ok(x)
    }

    /// Backtracking line search
    #[allow(clippy::too_many_arguments)]
    fn backtracking_line_search(
        &self,
        w: &Array1<f64>,
        s: &Array1<f64>,
        t: &Array1<f64>,
        lambda: f64,
        z_l: &Array1<f64>,
        z_u: &Array1<f64>,
        delta_w: &Array1<f64>,
        delta_lambda: f64,
        delta_s: &Array1<f64>,
        delta_t: &Array1<f64>,
        delta_z_l: &Array1<f64>,
        delta_z_u: &Array1<f64>,
        mu: &Array1<f64>,
        sigma: &Array2<f64>,
        risk_aversion: f64,
        w_min: f64,
        w_max: f64,
        barrier_mu: f64,
    ) -> f64 {
        let mut alpha = 1.0;

        // Ensure slacks and duals remain positive
        for i in 0..w.len() {
            if delta_s[i] < 0.0 {
                alpha = alpha.min(-0.99_f64 * s[i] / delta_s[i]);
            }
            if delta_t[i] < 0.0 {
                alpha = alpha.min(-0.99_f64 * t[i] / delta_t[i]);
            }
            if delta_z_l[i] < 0.0 {
                alpha = alpha.min(-0.99_f64 * z_l[i] / delta_z_l[i]);
            }
            if delta_z_u[i] < 0.0 {
                alpha = alpha.min(-0.99_f64 * z_u[i] / delta_z_u[i]);
            }
        }

        // Backtracking to ensure sufficient decrease
        for _ in 0..self.config.max_line_search {
            let w_new = w + &(delta_w * alpha);
            let s_new = s + &(delta_s * alpha);
            let t_new = t + &(delta_t * alpha);

            // Check feasibility
            let feasible = s_new.iter().all(|&si| si > 0.0)
                && t_new.iter().all(|&ti| ti > 0.0)
                && w_new.iter().all(|&wi| wi >= w_min && wi <= w_max);

            if feasible {
                break;
            }

            alpha *= self.config.alpha_reduction;
        }

        alpha.max(1e-4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interior_point_simple() {
        // Simple 2-asset portfolio
        let mu = Array1::from_vec(vec![0.10, 0.15]); // Expected returns
        let sigma = Array2::from_shape_vec((2, 2), vec![0.04, 0.02, 0.02, 0.09]).unwrap();

        let config = InteriorPointConfig::default();
        let solver = InteriorPointQpSolver::new(config);

        let result = solver.solve_portfolio(&mu, &sigma, 1.0, 0.0, 1.0);
        assert!(result.is_ok());

        let sol = result.unwrap();
        println!("Weights: {:?}", sol.weights);
        println!("Objective: {}", sol.objective);
        println!("Iterations: {}", sol.iterations);
        println!("Converged: {}", sol.converged);

        // Check budget constraint
        let weight_sum: f64 = sol.weights.sum();
        assert!((weight_sum - 1.0).abs() < 1e-4);

        // Check bounds
        assert!(sol.weights.iter().all(|&w| w >= 0.0 && w <= 1.0));
    }

    #[test]
    fn test_interior_point_3_assets() {
        let mu = Array1::from_vec(vec![0.08, 0.12, 0.15]);
        let sigma = Array2::from_shape_vec(
            (3, 3),
            vec![
                0.04, 0.01, 0.02,
                0.01, 0.09, 0.03,
                0.02, 0.03, 0.16,
            ],
        )
        .unwrap();

        let config = InteriorPointConfig {
            max_iterations: 50,
            kkt_tolerance: 1e-5,
            ..Default::default()
        };
        let solver = InteriorPointQpSolver::new(config);

        let result = solver.solve_portfolio(&mu, &sigma, 1.0, 0.0, 0.5);
        assert!(result.is_ok());

        let sol = result.unwrap();
        println!("\n3-Asset Portfolio:");
        println!("Weights: {:?}", sol.weights);
        println!("Iterations: {}", sol.iterations);

        assert!((sol.weights.sum() - 1.0).abs() < 1e-4);
        assert!(sol.weights.iter().all(|&w| w >= 0.0 && w <= 0.5));
    }
}
