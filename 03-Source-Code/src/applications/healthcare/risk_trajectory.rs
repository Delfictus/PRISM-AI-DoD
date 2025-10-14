//! Patient Risk Trajectory Forecasting
//!
//! Predicts future patient risk scores using time series analysis:
//! - Historical risk score tracking
//! - ARIMA-based trajectory forecasting
//! - Early deterioration detection
//! - Treatment impact assessment

use anyhow::{Result, Context};
use super::risk_predictor::{RiskAssessment, MedicalHistory, HealthcareRiskPredictor, RiskCategory};
use crate::time_series::{TimeSeriesForecaster, ArimaConfig};

/// Historical risk data point
#[derive(Debug, Clone)]
pub struct RiskTimePoint {
    /// Time offset (hours from baseline)
    pub time_hours: f64,
    /// Mortality risk at this time
    pub mortality_risk: f64,
    /// Sepsis risk at this time
    pub sepsis_risk: f64,
    /// Severity score at this time
    pub severity_score: f64,
    /// Overall risk category
    pub risk_category: RiskCategory,
}

/// Forecasted risk trajectory
#[derive(Debug, Clone)]
pub struct RiskTrajectory {
    /// Historical risk points
    pub historical: Vec<RiskTimePoint>,
    /// Forecasted mortality risk (hourly for next 24-48 hours)
    pub forecasted_mortality: Vec<f64>,
    /// Forecasted sepsis risk
    pub forecasted_sepsis: Vec<f64>,
    /// Forecasted severity scores
    pub forecasted_severity: Vec<f64>,
    /// Forecast horizon (hours)
    pub horizon_hours: usize,
    /// Uncertainty intervals (95% CI)
    pub mortality_lower: Vec<f64>,
    pub mortality_upper: Vec<f64>,
    /// Early warning alerts
    pub warnings: Vec<String>,
}

/// Trajectory forecasting configuration
#[derive(Debug, Clone)]
pub struct TrajectoryConfig {
    /// Forecast horizon in hours (default: 24)
    pub horizon_hours: usize,
    /// ARIMA model configuration
    pub arima_config: ArimaConfig,
    /// Deterioration alert threshold (risk increase rate)
    pub deterioration_threshold: f64,
    /// Use uncertainty quantification
    pub use_uncertainty: bool,
}

impl Default for TrajectoryConfig {
    fn default() -> Self {
        Self {
            horizon_hours: 24,  // 24-hour forecast
            arima_config: ArimaConfig {
                p: 2,  // Autoregressive order
                d: 0,  // No differencing (risk scores are stationary)
                q: 1,  // Moving average order
                include_constant: true,
            },
            deterioration_threshold: 0.1,  // 10% risk increase triggers warning
            use_uncertainty: true,
        }
    }
}

/// Risk trajectory forecaster
pub struct RiskTrajectoryForecaster {
    /// Time series forecaster
    forecaster: TimeSeriesForecaster,
    /// Configuration
    config: TrajectoryConfig,
}

impl RiskTrajectoryForecaster {
    /// Create new trajectory forecaster
    pub fn new(config: TrajectoryConfig) -> Self {
        Self {
            forecaster: TimeSeriesForecaster::new(),
            config,
        }
    }

    /// Forecast risk trajectory from historical assessments
    pub fn forecast_trajectory(
        &mut self,
        historical_risk: &[RiskTimePoint],
    ) -> Result<RiskTrajectory> {
        if historical_risk.len() < 3 {
            anyhow::bail!("Need at least 3 historical risk points for forecasting");
        }

        // Extract time series
        let mortality_series: Vec<f64> = historical_risk.iter()
            .map(|r| r.mortality_risk)
            .collect();

        let sepsis_series: Vec<f64> = historical_risk.iter()
            .map(|r| r.sepsis_risk)
            .collect();

        let severity_series: Vec<f64> = historical_risk.iter()
            .map(|r| r.severity_score)
            .collect();

        // Forecast mortality risk
        let forecasted_mortality = match self.forecaster.fit_arima(&mortality_series, self.config.arima_config.clone()) {
            Ok(_) => {
                self.forecaster.forecast_arima(self.config.horizon_hours)
                    .context("Failed to forecast mortality risk")?
            },
            Err(_) => {
                // Fallback: Use simple linear extrapolation if ARIMA fails
                let trend = self.estimate_linear_trend(&mortality_series);
                self.simple_linear_forecast(&mortality_series, self.config.horizon_hours, trend)
            }
        };

        // Compute uncertainty if enabled
        let (mortality_lower, mortality_upper) = if self.config.use_uncertainty {
            self.compute_uncertainty_bounds(&forecasted_mortality, &mortality_series)?
        } else {
            (vec![0.0; forecasted_mortality.len()], vec![1.0; forecasted_mortality.len()])
        };

        // Forecast sepsis risk
        let forecasted_sepsis = match self.forecaster.fit_arima(&sepsis_series, self.config.arima_config.clone()) {
            Ok(_) => self.forecaster.forecast_arima(self.config.horizon_hours)?,
            Err(_) => {
                let trend = self.estimate_linear_trend(&sepsis_series);
                self.simple_linear_forecast(&sepsis_series, self.config.horizon_hours, trend)
            }
        };

        // Forecast severity score
        let forecasted_severity = match self.forecaster.fit_arima(&severity_series, self.config.arima_config.clone()) {
            Ok(_) => self.forecaster.forecast_arima(self.config.horizon_hours)?,
            Err(_) => {
                let trend = self.estimate_linear_trend(&severity_series);
                self.simple_linear_forecast(&severity_series, self.config.horizon_hours, trend)
            }
        };

        // Generate early warning alerts
        let warnings = self.generate_warnings(
            historical_risk,
            &forecasted_mortality,
            &forecasted_sepsis,
            &forecasted_severity,
        )?;

        Ok(RiskTrajectory {
            historical: historical_risk.to_vec(),
            forecasted_mortality: forecasted_mortality.clone(),
            forecasted_sepsis,
            forecasted_severity,
            horizon_hours: self.config.horizon_hours,
            mortality_lower,
            mortality_upper,
            warnings,
        })
    }

    /// Estimate linear trend from historical data
    fn estimate_linear_trend(&self, data: &[f64]) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }

        // Simple linear regression: y = mx + b
        let n = data.len() as f64;
        let x_mean = (n - 1.0) / 2.0;  // Indices 0..n-1, mean is (n-1)/2
        let y_mean = data.iter().sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (i, &y) in data.iter().enumerate() {
            let x = i as f64;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean).powi(2);
        }

        if denominator.abs() < 1e-10 {
            0.0  // No trend if denominator is zero
        } else {
            numerator / denominator  // Slope (trend)
        }
    }

    /// Simple linear forecast (fallback when ARIMA fails)
    fn simple_linear_forecast(&self, historical: &[f64], horizon: usize, trend: f64) -> Vec<f64> {
        let last_value = historical.last().copied().unwrap_or(0.0);
        let n = historical.len();

        (0..horizon)
            .map(|h| {
                let forecasted = last_value + trend * (h + 1) as f64;
                // Clamp to [0, 1] for risk probabilities
                forecasted.max(0.0).min(1.0)
            })
            .collect()
    }

    /// Compute uncertainty bounds using simple standard deviation approach
    fn compute_uncertainty_bounds(
        &self,
        forecast: &[f64],
        historical: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        // Compute standard deviation from historical data
        let mean = historical.iter().sum::<f64>() / historical.len() as f64;
        let variance = historical.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / historical.len() as f64;
        let std_dev = variance.sqrt();

        // 95% confidence interval = ±1.96 * std_dev
        let lower: Vec<f64> = forecast.iter()
            .map(|&f| (f - 1.96 * std_dev).max(0.0))
            .collect();

        let upper: Vec<f64> = forecast.iter()
            .map(|&f| (f + 1.96 * std_dev).min(1.0))
            .collect();

        Ok((lower, upper))
    }

    /// Generate early warning alerts based on forecast
    fn generate_warnings(
        &self,
        historical: &[RiskTimePoint],
        forecasted_mortality: &[f64],
        forecasted_sepsis: &[f64],
        forecasted_severity: &[f64],
    ) -> Result<Vec<String>> {
        let mut warnings = Vec::new();

        // Get current risk (last historical point)
        let current = historical.last().unwrap();

        // Check for rapid mortality risk increase
        if !forecasted_mortality.is_empty() {
            let mortality_increase = forecasted_mortality[0] - current.mortality_risk;
            if mortality_increase > self.config.deterioration_threshold {
                warnings.push(format!(
                    "⚠️  EARLY WARNING: Mortality risk predicted to increase by {:.1}% in next hour",
                    mortality_increase * 100.0
                ));
            }

            // Check 12-hour forecast
            if forecasted_mortality.len() >= 12 {
                let mortality_12h = forecasted_mortality[11] - current.mortality_risk;
                if mortality_12h > 0.2 {
                    warnings.push(format!(
                        "⚠️  12-HOUR ALERT: Mortality risk may increase by {:.1}% (currently {:.1}%)",
                        mortality_12h * 100.0,
                        current.mortality_risk * 100.0
                    ));
                }
            }
        }

        // Check for sepsis risk escalation
        if !forecasted_sepsis.is_empty() {
            let sepsis_increase = forecasted_sepsis[0] - current.sepsis_risk;
            if sepsis_increase > 0.15 && forecasted_sepsis[0] > 0.6 {
                warnings.push(format!(
                    "⚠️  SEPSIS ALERT: Sepsis risk predicted to reach {:.1}% (currently {:.1}%)",
                    forecasted_sepsis[0] * 100.0,
                    current.sepsis_risk * 100.0
                ));
            }
        }

        // Check for severity score increase
        if !forecasted_severity.is_empty() && forecasted_severity.len() >= 6 {
            let severity_6h = forecasted_severity[5] - current.severity_score;
            if severity_6h > 5.0 {
                warnings.push(format!(
                    "⚠️  6-HOUR WARNING: Severity score may increase by {:.1} points",
                    severity_6h
                ));
            }
        }

        // Check for sustained high risk
        if forecasted_mortality.len() >= 24 {
            let sustained_high = forecasted_mortality[12..24]
                .iter()
                .filter(|&&r| r > 0.5)
                .count();

            if sustained_high > 8 {  // More than 8 hours in next 12-hour period
                warnings.push(
                    "⚠️  24-HOUR OUTLOOK: Patient likely to remain high-risk for extended period".to_string()
                );
            }
        }

        // Positive trend detection
        if forecasted_mortality.len() >= 12 {
            let improving = forecasted_mortality[..12]
                .windows(2)
                .all(|w| w[1] <= w[0]);

            if improving && current.mortality_risk > 0.3 {
                warnings.push(
                    "✓ POSITIVE TREND: Risk scores predicted to decline steadily".to_string()
                );
            }
        }

        Ok(warnings)
    }

    /// Assess treatment impact by comparing forecasts with/without intervention
    pub fn assess_treatment_impact(
        &mut self,
        baseline_trajectory: &RiskTrajectory,
        post_treatment_risk: f64,
    ) -> Result<TreatmentImpact> {
        // Compare baseline forecast with post-treatment risk
        if baseline_trajectory.forecasted_mortality.is_empty() {
            anyhow::bail!("Baseline trajectory has no forecasts");
        }

        let baseline_risk_1h = baseline_trajectory.forecasted_mortality[0];
        let risk_reduction = baseline_risk_1h - post_treatment_risk;

        let impact_category = if risk_reduction > 0.2 {
            ImpactCategory::Significant
        } else if risk_reduction > 0.1 {
            ImpactCategory::Moderate
        } else if risk_reduction > 0.05 {
            ImpactCategory::Mild
        } else {
            ImpactCategory::Minimal
        };

        Ok(TreatmentImpact {
            risk_reduction,
            impact_category,
            baseline_forecast: baseline_trajectory.forecasted_mortality.clone(),
            post_treatment_risk,
        })
    }
}

/// Treatment impact assessment
#[derive(Debug, Clone)]
pub struct TreatmentImpact {
    /// Risk reduction achieved (baseline - post-treatment)
    pub risk_reduction: f64,
    /// Impact category
    pub impact_category: ImpactCategory,
    /// Baseline forecast (what would have happened)
    pub baseline_forecast: Vec<f64>,
    /// Actual post-treatment risk
    pub post_treatment_risk: f64,
}

/// Treatment impact category
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImpactCategory {
    /// Significant improvement (>20% risk reduction)
    Significant,
    /// Moderate improvement (10-20% reduction)
    Moderate,
    /// Mild improvement (5-10% reduction)
    Mild,
    /// Minimal or no improvement (<5% reduction)
    Minimal,
}

impl RiskTrajectory {
    /// Print trajectory summary
    pub fn print_summary(&self) {
        println!("\n=== Patient Risk Trajectory Forecast ===");

        if let Some(current) = self.historical.last() {
            println!("\nCurrent Risk Status:");
            println!("  Mortality: {:.1}%", current.mortality_risk * 100.0);
            println!("  Sepsis: {:.1}%", current.sepsis_risk * 100.0);
            println!("  Severity Score: {:.1}", current.severity_score);
            println!("  Category: {:?}", current.risk_category);
        }

        println!("\nForecast ({}-hour outlook):", self.horizon_hours);

        // Show key time points
        let time_points = [1, 6, 12, 24];
        for &hour in &time_points {
            if hour <= self.forecasted_mortality.len() {
                let idx = hour - 1;
                println!("\n  +{} hours:", hour);
                println!("    Mortality: {:.1}% (95% CI: {:.1}%-{:.1}%)",
                         self.forecasted_mortality[idx] * 100.0,
                         self.mortality_lower[idx] * 100.0,
                         self.mortality_upper[idx] * 100.0);
                println!("    Sepsis: {:.1}%", self.forecasted_sepsis[idx] * 100.0);
                println!("    Severity: {:.1}", self.forecasted_severity[idx]);
            }
        }

        if !self.warnings.is_empty() {
            println!("\nEarly Warning Alerts:");
            for warning in &self.warnings {
                println!("  {}", warning);
            }
        } else {
            println!("\n✓ No immediate alerts - patient trajectory appears stable");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_stable_trajectory() -> Vec<RiskTimePoint> {
        vec![
            RiskTimePoint {
                time_hours: 0.0,
                mortality_risk: 0.15,
                sepsis_risk: 0.20,
                severity_score: 8.0,
                risk_category: RiskCategory::Low,
            },
            RiskTimePoint {
                time_hours: 4.0,
                mortality_risk: 0.14,
                sepsis_risk: 0.18,
                severity_score: 7.5,
                risk_category: RiskCategory::Low,
            },
            RiskTimePoint {
                time_hours: 8.0,
                mortality_risk: 0.13,
                sepsis_risk: 0.17,
                severity_score: 7.0,
                risk_category: RiskCategory::Low,
            },
            RiskTimePoint {
                time_hours: 12.0,
                mortality_risk: 0.12,
                sepsis_risk: 0.16,
                severity_score: 6.5,
                risk_category: RiskCategory::Low,
            },
        ]
    }

    fn create_deteriorating_trajectory() -> Vec<RiskTimePoint> {
        vec![
            RiskTimePoint {
                time_hours: 0.0,
                mortality_risk: 0.20,
                sepsis_risk: 0.30,
                severity_score: 10.0,
                risk_category: RiskCategory::Moderate,
            },
            RiskTimePoint {
                time_hours: 4.0,
                mortality_risk: 0.28,
                sepsis_risk: 0.42,
                severity_score: 14.0,
                risk_category: RiskCategory::Moderate,
            },
            RiskTimePoint {
                time_hours: 8.0,
                mortality_risk: 0.38,
                sepsis_risk: 0.55,
                severity_score: 18.5,
                risk_category: RiskCategory::High,
            },
            RiskTimePoint {
                time_hours: 12.0,
                mortality_risk: 0.48,
                sepsis_risk: 0.68,
                severity_score: 23.0,
                risk_category: RiskCategory::High,
            },
        ]
    }

    #[test]
    fn test_trajectory_forecasting_stable() {
        let history = create_stable_trajectory();
        let config = TrajectoryConfig::default();
        let mut forecaster = RiskTrajectoryForecaster::new(config);

        let trajectory = forecaster.forecast_trajectory(&history).unwrap();

        assert_eq!(trajectory.forecasted_mortality.len(), 24);
        assert_eq!(trajectory.forecasted_sepsis.len(), 24);
        assert_eq!(trajectory.forecasted_severity.len(), 24);

        // Stable patient should have few or no warnings
        println!("Stable patient warnings: {:?}", trajectory.warnings);
    }

    #[test]
    fn test_trajectory_forecasting_deteriorating() {
        let history = create_deteriorating_trajectory();
        let config = TrajectoryConfig::default();
        let mut forecaster = RiskTrajectoryForecaster::new(config);

        let trajectory = forecaster.forecast_trajectory(&history).unwrap();

        // Deteriorating patient should trigger warnings
        assert!(!trajectory.warnings.is_empty(), "Expected warnings for deteriorating patient");

        println!("Deteriorating patient warnings: {:?}", trajectory.warnings);
    }

    #[test]
    fn test_insufficient_history() {
        let history = vec![
            RiskTimePoint {
                time_hours: 0.0,
                mortality_risk: 0.15,
                sepsis_risk: 0.20,
                severity_score: 8.0,
                risk_category: RiskCategory::Low,
            },
        ];

        let config = TrajectoryConfig::default();
        let mut forecaster = RiskTrajectoryForecaster::new(config);

        let result = forecaster.forecast_trajectory(&history);
        assert!(result.is_err(), "Should fail with insufficient history");
    }

    #[test]
    fn test_treatment_impact_assessment() {
        let history = create_deteriorating_trajectory();
        let config = TrajectoryConfig::default();
        let mut forecaster = RiskTrajectoryForecaster::new(config);

        let baseline = forecaster.forecast_trajectory(&history).unwrap();

        // Simulate treatment effect (risk reduced from forecasted level)
        let post_treatment_risk = 0.35;  // Down from ~0.55 forecasted

        let impact = forecaster.assess_treatment_impact(&baseline, post_treatment_risk).unwrap();

        println!("Treatment impact: {:?}", impact.impact_category);
        println!("Risk reduction: {:.1}%", impact.risk_reduction * 100.0);

        assert!(impact.risk_reduction > 0.0, "Treatment should reduce risk");
    }
}
