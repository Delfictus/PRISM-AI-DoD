//! Adaptive Temperature Control for Thermodynamic Schedules
//!
//! Implements sophisticated adaptive temperature control mechanisms to
//! automatically tune temperature parameters based on acceptance rates
//! and system dynamics.
//!
//! Worker 5 Enhancement: Week 3, Task 3.3
//!
//! ## Theory
//!
//! In thermodynamic optimization, the temperature parameter controls the
//! exploration/exploitation trade-off. Optimal temperatures:
//! - Too high → random search (poor convergence)
//! - Too low → greedy search (local optima)
//! - Just right → efficient global optimization
//!
//! Adaptive control automatically adjusts temperature to maintain target
//! acceptance rates, which correlate with optimal exploration.

use anyhow::{Context, Result};
use std::collections::VecDeque;

/// Target acceptance rate for optimal exploration
///
/// Classic result from simulated annealing theory: ~0.44 acceptance rate
/// provides good balance between exploration and exploitation.
pub const OPTIMAL_ACCEPTANCE_RATE: f64 = 0.44;

/// Acceptance rate monitoring window
///
/// Tracks recent acceptance decisions to compute current acceptance rate
#[derive(Debug, Clone)]
pub struct AcceptanceMonitor {
    /// Window of recent acceptance decisions (true = accepted, false = rejected)
    history: VecDeque<bool>,
    /// Maximum window size
    max_window: usize,
    /// Current acceptance rate (cached)
    current_rate: f64,
}

impl AcceptanceMonitor {
    /// Create new acceptance monitor
    ///
    /// # Arguments
    /// * `window_size` - Number of recent decisions to track (default: 100)
    pub fn new(window_size: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(window_size),
            max_window: window_size,
            current_rate: 0.0,
        }
    }

    /// Record an acceptance decision
    ///
    /// # Arguments
    /// * `accepted` - Whether the move was accepted
    pub fn record(&mut self, accepted: bool) {
        // Add new decision
        self.history.push_back(accepted);

        // Remove oldest if window full
        if self.history.len() > self.max_window {
            self.history.pop_front();
        }

        // Update cached acceptance rate
        self.update_rate();
    }

    /// Update cached acceptance rate from history
    fn update_rate(&mut self) {
        if self.history.is_empty() {
            self.current_rate = 0.0;
        } else {
            let accepted_count = self.history.iter().filter(|&&x| x).count();
            self.current_rate = accepted_count as f64 / self.history.len() as f64;
        }
    }

    /// Get current acceptance rate
    pub fn acceptance_rate(&self) -> f64 {
        self.current_rate
    }

    /// Get number of samples in history
    pub fn sample_count(&self) -> usize {
        self.history.len()
    }

    /// Check if monitor has enough samples for reliable rate estimate
    pub fn is_ready(&self) -> bool {
        self.history.len() >= self.max_window / 2
    }

    /// Reset the monitor
    pub fn reset(&mut self) {
        self.history.clear();
        self.current_rate = 0.0;
    }
}

/// PID controller for temperature adjustment
///
/// Proportional-Integral-Derivative controller that adjusts temperature
/// to maintain target acceptance rate.
///
/// ## PID Control Theory
///
/// - **P (Proportional)**: Corrects based on current error
/// - **I (Integral)**: Corrects based on accumulated past error
/// - **D (Derivative)**: Predicts future error based on rate of change
///
/// Output: `u(t) = Kp*e(t) + Ki*∫e(τ)dτ + Kd*de(t)/dt`
#[derive(Debug, Clone)]
pub struct PIDController {
    /// Proportional gain (Kp)
    kp: f64,
    /// Integral gain (Ki)
    ki: f64,
    /// Derivative gain (Kd)
    kd: f64,

    /// Target setpoint (desired acceptance rate)
    target: f64,

    /// Accumulated error (integral term)
    integral: f64,
    /// Previous error (for derivative term)
    prev_error: f64,

    /// Output limits
    output_min: f64,
    output_max: f64,

    /// Anti-windup: max integral accumulation
    integral_max: f64,
}

impl PIDController {
    /// Create new PID controller with default tuning
    ///
    /// # Arguments
    /// * `target` - Target acceptance rate (typically 0.44)
    /// * `output_range` - (min, max) temperature adjustment multiplier
    pub fn new(target: f64, output_range: (f64, f64)) -> Self {
        Self::with_gains(target, 0.1, 0.01, 0.05, output_range)
    }

    /// Create PID controller with custom gains
    ///
    /// # Arguments
    /// * `target` - Target acceptance rate
    /// * `kp` - Proportional gain (typical: 0.1-1.0)
    /// * `ki` - Integral gain (typical: 0.01-0.1)
    /// * `kd` - Derivative gain (typical: 0.01-0.1)
    /// * `output_range` - (min, max) temperature adjustment multiplier
    pub fn with_gains(
        target: f64,
        kp: f64,
        ki: f64,
        kd: f64,
        output_range: (f64, f64),
    ) -> Self {
        Self {
            kp,
            ki,
            kd,
            target,
            integral: 0.0,
            prev_error: 0.0,
            output_min: output_range.0,
            output_max: output_range.1,
            integral_max: 10.0, // Anti-windup threshold
        }
    }

    /// Compute control output based on current measurement
    ///
    /// # Arguments
    /// * `current` - Current acceptance rate measurement
    ///
    /// # Returns
    /// Temperature adjustment multiplier (apply as: T_new = T_old * output)
    pub fn update(&mut self, current: f64) -> f64 {
        // Compute error
        let error = self.target - current;

        // Proportional term
        let p_term = self.kp * error;

        // Integral term with anti-windup
        self.integral += error;
        self.integral = self.integral.clamp(-self.integral_max, self.integral_max);
        let i_term = self.ki * self.integral;

        // Derivative term
        let d_term = self.kd * (error - self.prev_error);
        self.prev_error = error;

        // Compute output
        let output = p_term + i_term + d_term;

        // Convert to temperature adjustment multiplier
        // Positive output → increase temperature (lower acceptance rate)
        // Negative output → decrease temperature (higher acceptance rate)
        let adjustment = 1.0 + output;

        // Clamp to output range
        adjustment.clamp(self.output_min, self.output_max)
    }

    /// Reset controller state
    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.prev_error = 0.0;
    }

    /// Get current integral value (for diagnostics)
    pub fn integral_value(&self) -> f64 {
        self.integral
    }
}

/// Adaptive temperature controller combining monitoring and PID control
///
/// High-level interface that:
/// 1. Monitors acceptance rate
/// 2. Uses PID control to adjust temperature
/// 3. Provides adaptive cooling schedule
#[derive(Debug, Clone)]
pub struct AdaptiveTemperatureController {
    /// Acceptance rate monitor
    monitor: AcceptanceMonitor,
    /// PID controller
    pid: PIDController,

    /// Current temperature
    temperature: f64,
    /// Minimum allowed temperature
    min_temperature: f64,
    /// Maximum allowed temperature
    max_temperature: f64,

    /// Number of updates performed
    update_count: usize,
    /// Temperature history (for analysis)
    temp_history: VecDeque<f64>,
    /// Max history length
    max_history: usize,
}

impl AdaptiveTemperatureController {
    /// Create new adaptive temperature controller
    ///
    /// # Arguments
    /// * `initial_temp` - Starting temperature
    /// * `temp_range` - (min, max) allowed temperatures
    /// * `target_acceptance` - Target acceptance rate (default: 0.44)
    /// * `window_size` - Acceptance rate window size (default: 100)
    pub fn new(
        initial_temp: f64,
        temp_range: (f64, f64),
        target_acceptance: Option<f64>,
        window_size: Option<usize>,
    ) -> Self {
        let target = target_acceptance.unwrap_or(OPTIMAL_ACCEPTANCE_RATE);
        let window = window_size.unwrap_or(100);

        Self {
            monitor: AcceptanceMonitor::new(window),
            pid: PIDController::new(target, (0.9, 1.1)), // ±10% adjustment per step
            temperature: initial_temp,
            min_temperature: temp_range.0,
            max_temperature: temp_range.1,
            update_count: 0,
            temp_history: VecDeque::with_capacity(1000),
            max_history: 1000,
        }
    }

    /// Record an acceptance decision and update temperature
    ///
    /// # Arguments
    /// * `accepted` - Whether the move was accepted
    ///
    /// # Returns
    /// New temperature value
    pub fn record_and_update(&mut self, accepted: bool) -> f64 {
        // Record acceptance
        self.monitor.record(accepted);

        // Only update temperature if we have enough samples
        if self.monitor.is_ready() {
            let current_rate = self.monitor.acceptance_rate();
            let adjustment = self.pid.update(current_rate);

            // Apply adjustment
            self.temperature *= adjustment;

            // Clamp to allowed range
            self.temperature = self.temperature.clamp(
                self.min_temperature,
                self.max_temperature,
            );
        }

        self.update_count += 1;

        // Record temperature in history
        self.temp_history.push_back(self.temperature);
        if self.temp_history.len() > self.max_history {
            self.temp_history.pop_front();
        }

        self.temperature
    }

    /// Get current temperature
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Get current acceptance rate
    pub fn acceptance_rate(&self) -> f64 {
        self.monitor.acceptance_rate()
    }

    /// Check if controller is ready (has enough samples)
    pub fn is_ready(&self) -> bool {
        self.monitor.is_ready()
    }

    /// Get statistics about controller performance
    pub fn statistics(&self) -> AdaptiveControllerStats {
        AdaptiveControllerStats {
            update_count: self.update_count,
            current_temperature: self.temperature,
            current_acceptance_rate: self.monitor.acceptance_rate(),
            avg_temperature: self.average_temperature(),
            temperature_std: self.temperature_std(),
            is_converged: self.is_converged(),
        }
    }

    /// Calculate average temperature over history
    fn average_temperature(&self) -> f64 {
        if self.temp_history.is_empty() {
            self.temperature
        } else {
            self.temp_history.iter().sum::<f64>() / self.temp_history.len() as f64
        }
    }

    /// Calculate standard deviation of temperature
    fn temperature_std(&self) -> f64 {
        if self.temp_history.len() < 2 {
            return 0.0;
        }

        let mean = self.average_temperature();
        let variance = self.temp_history.iter()
            .map(|&t| (t - mean).powi(2))
            .sum::<f64>() / self.temp_history.len() as f64;

        variance.sqrt()
    }

    /// Check if temperature has converged (stable)
    fn is_converged(&self) -> bool {
        if self.temp_history.len() < 50 {
            return false;
        }

        // Check if recent temperature changes are small
        let recent: Vec<f64> = self.temp_history.iter()
            .rev()
            .take(50)
            .copied()
            .collect();

        let mean = recent.iter().sum::<f64>() / recent.len() as f64;
        let max_deviation = recent.iter()
            .map(|&t| (t - mean).abs() / mean)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        max_deviation < 0.05 // Less than 5% variation
    }

    /// Reset controller to initial state
    pub fn reset(&mut self, new_initial_temp: f64) {
        self.temperature = new_initial_temp;
        self.monitor.reset();
        self.pid.reset();
        self.update_count = 0;
        self.temp_history.clear();
    }

    /// Get temperature history for analysis
    pub fn temperature_history(&self) -> Vec<f64> {
        self.temp_history.iter().copied().collect()
    }
}

/// Statistics about adaptive controller performance
#[derive(Debug, Clone)]
pub struct AdaptiveControllerStats {
    pub update_count: usize,
    pub current_temperature: f64,
    pub current_acceptance_rate: f64,
    pub avg_temperature: f64,
    pub temperature_std: f64,
    pub is_converged: bool,
}

/// Adaptive cooling schedule using acceptance rate feedback
///
/// Combines traditional cooling with adaptive temperature control
#[derive(Debug, Clone)]
pub struct AdaptiveCoolingSchedule {
    /// Adaptive controller
    controller: AdaptiveTemperatureController,
    /// Base cooling rate (applied in addition to adaptive control)
    base_cooling_rate: f64,
    /// Iteration counter
    iteration: usize,
}

impl AdaptiveCoolingSchedule {
    /// Create new adaptive cooling schedule
    ///
    /// # Arguments
    /// * `initial_temp` - Starting temperature
    /// * `temp_range` - (min, max) allowed temperatures
    /// * `base_cooling_rate` - Base cooling rate (e.g., 0.995 for 0.5% per iteration)
    pub fn new(
        initial_temp: f64,
        temp_range: (f64, f64),
        base_cooling_rate: f64,
    ) -> Self {
        Self {
            controller: AdaptiveTemperatureController::new(
                initial_temp,
                temp_range,
                Some(OPTIMAL_ACCEPTANCE_RATE),
                Some(100),
            ),
            base_cooling_rate,
            iteration: 0,
        }
    }

    /// Update schedule with acceptance decision
    ///
    /// Combines base cooling with adaptive adjustment
    ///
    /// # Arguments
    /// * `accepted` - Whether move was accepted
    ///
    /// # Returns
    /// New temperature
    pub fn update(&mut self, accepted: bool) -> f64 {
        // Apply adaptive control
        let adaptive_temp = self.controller.record_and_update(accepted);

        // Apply base cooling
        let cooled_temp = adaptive_temp * self.base_cooling_rate;

        // Update controller's temperature
        self.controller.temperature = cooled_temp;

        self.iteration += 1;
        cooled_temp
    }

    /// Get current temperature
    pub fn temperature(&self) -> f64 {
        self.controller.temperature()
    }

    /// Get current acceptance rate
    pub fn acceptance_rate(&self) -> f64 {
        self.controller.acceptance_rate()
    }

    /// Get controller statistics
    pub fn statistics(&self) -> AdaptiveControllerStats {
        self.controller.statistics()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_acceptance_monitor() {
        let mut monitor = AcceptanceMonitor::new(10);

        // Record 5 acceptances, 5 rejections
        for i in 0..10 {
            monitor.record(i < 5);
        }

        assert_eq!(monitor.sample_count(), 10);
        assert!((monitor.acceptance_rate() - 0.5).abs() < 1e-6);
        assert!(monitor.is_ready());
    }

    #[test]
    fn test_acceptance_monitor_window() {
        let mut monitor = AcceptanceMonitor::new(5);

        // Record 10 values, only last 5 should be kept
        for i in 0..10 {
            monitor.record(i >= 5); // First 5 rejected, last 5 accepted
        }

        assert_eq!(monitor.sample_count(), 5);
        assert!((monitor.acceptance_rate() - 1.0).abs() < 1e-6); // All recent accepted
    }

    #[test]
    fn test_pid_controller_proportional() {
        let mut pid = PIDController::new(0.44, (0.5, 1.5));

        // Current rate too low (0.2), need to increase temperature
        let adjustment = pid.update(0.2);
        assert!(adjustment > 1.0); // Should increase temperature

        // Current rate too high (0.7), need to decrease temperature
        pid.reset();
        let adjustment = pid.update(0.7);
        assert!(adjustment < 1.0); // Should decrease temperature
    }

    #[test]
    fn test_pid_controller_at_target() {
        let mut pid = PIDController::new(0.44, (0.5, 1.5));

        // At target, should have minimal adjustment
        let adjustment = pid.update(0.44);
        assert!((adjustment - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_adaptive_controller() {
        let mut controller = AdaptiveTemperatureController::new(
            10.0,        // initial temp
            (0.1, 100.0), // range
            Some(0.5),   // target 50% acceptance
            Some(10),    // small window for testing
        );

        // Simulate low acceptance rate (rejecting most moves)
        for _ in 0..20 {
            controller.record_and_update(false);
        }

        // Temperature should increase to improve acceptance
        assert!(controller.temperature() > 10.0);

        // Reset and simulate high acceptance rate
        controller.reset(10.0);
        for _ in 0..20 {
            controller.record_and_update(true);
        }

        // Temperature should decrease to reduce acceptance
        assert!(controller.temperature() < 10.0);
    }

    #[test]
    fn test_adaptive_controller_bounds() {
        let mut controller = AdaptiveTemperatureController::new(
            10.0,
            (5.0, 15.0), // Narrow range
            Some(0.5),
            Some(10),
        );

        // Extreme low acceptance - should hit upper bound
        for _ in 0..100 {
            controller.record_and_update(false);
        }
        assert!(controller.temperature() <= 15.0);

        // Extreme high acceptance - should hit lower bound
        controller.reset(10.0);
        for _ in 0..100 {
            controller.record_and_update(true);
        }
        assert!(controller.temperature() >= 5.0);
    }

    #[test]
    fn test_adaptive_cooling_schedule() {
        let mut schedule = AdaptiveCoolingSchedule::new(
            10.0,  // initial temp
            (0.1, 100.0),
            0.99,  // 1% cooling per iteration
        );

        let initial = schedule.temperature();

        // Run for some iterations with balanced acceptance
        for i in 0..50 {
            schedule.update(i % 2 == 0); // 50% acceptance
        }

        // Temperature should decrease due to base cooling
        assert!(schedule.temperature() < initial);

        // Acceptance rate should be tracked
        let rate = schedule.acceptance_rate();
        assert!(rate > 0.4 && rate < 0.6); // Should be near 50%
    }

    #[test]
    fn test_convergence_detection() {
        let mut controller = AdaptiveTemperatureController::new(
            10.0,
            (1.0, 100.0),
            Some(0.44),
            Some(20),
        );

        // Should not be converged initially
        assert!(!controller.statistics().is_converged);

        // Simulate stable acceptance rate at target
        for _ in 0..100 {
            // Alternate to maintain ~44% acceptance
            controller.record_and_update(true);
            controller.record_and_update(false);
            controller.record_and_update(false);
        }

        // Should eventually converge
        let stats = controller.statistics();
        assert!(stats.is_converged || stats.temperature_std < 0.5);
    }
}
