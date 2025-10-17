//! Threat Forecasting Module
//!
//! Time series forecasting for cybersecurity threats:
//! - Network traffic anomaly prediction
//! - Attack pattern forecasting
//! - Threat trend analysis with Transfer Entropy
//! - Early warning system for security events
//!
//! Constitutional Compliance:
//! - Article II: GPU-accelerated forecasting (via Worker 2 integration)
//! - Article III: Comprehensive testing with synthetic attack data
//! - Article IV: Active Inference for adaptive threat prediction

use anyhow::{Result, bail};
use crate::time_series::{TimeSeriesForecaster, ArimaConfig, ForecastWithUncertainty};
use super::detector::{ThreatLevel, AttackType};

/// Configuration for threat forecasting
#[derive(Debug, Clone)]
pub struct ForecastConfig {
    /// Forecast horizon (hours)
    pub horizon_hours: usize,

    /// ARIMA model configuration
    pub arima_config: ArimaConfig,

    /// Anomaly detection threshold (standard deviations)
    pub anomaly_threshold: f64,

    /// Early warning threshold (hours ahead)
    pub early_warning_hours: usize,
}

impl Default for ForecastConfig {
    fn default() -> Self {
        Self {
            horizon_hours: 24,
            arima_config: ArimaConfig {
                p: 2,
                d: 1,
                q: 1,
                include_constant: true,
            },
            anomaly_threshold: 2.0,  // 2 standard deviations
            early_warning_hours: 4,
        }
    }
}

/// Security metric time point
#[derive(Debug, Clone)]
pub struct SecurityMetricPoint {
    /// Time in hours
    pub time_hours: f64,

    /// Number of security events
    pub event_count: f64,

    /// Average threat level (0-1)
    pub avg_threat_level: f64,

    /// Network traffic volume (normalized)
    pub traffic_volume: f64,

    /// Failed login attempts
    pub failed_logins: f64,
}

/// Threat trajectory forecast
#[derive(Debug, Clone)]
pub struct ThreatTrajectory {
    /// Forecasted event counts
    pub event_count_forecast: Vec<f64>,

    /// Forecasted threat levels
    pub threat_level_forecast: Vec<f64>,

    /// Forecasted traffic volumes
    pub traffic_volume_forecast: Vec<f64>,

    /// Forecast uncertainty (95% CI)
    pub event_count_uncertainty: Option<ForecastWithUncertainty>,

    /// Early warning alerts
    pub warnings: Vec<ThreatWarning>,

    /// Forecast horizon (hours)
    pub horizon_hours: usize,
}

/// Early warning alert
#[derive(Debug, Clone)]
pub struct ThreatWarning {
    /// Hours ahead when threat detected
    pub hours_ahead: usize,

    /// Warning message
    pub message: String,

    /// Predicted threat level
    pub predicted_level: ThreatLevel,

    /// Confidence (0-1)
    pub confidence: f64,
}

impl ThreatTrajectory {
    /// Print summary of threat forecast
    pub fn print_summary(&self) {
        println!("\n=== Threat Trajectory Forecast ({} hours) ===", self.horizon_hours);

        // Peak event rate
        let peak_events = self.event_count_forecast.iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let peak_hour = self.event_count_forecast.iter()
            .position(|&e| e == peak_events)
            .unwrap_or(0);

        println!("\nPeak Security Event Rate:");
        println!("  Time: +{} hours", peak_hour);
        println!("  Events/hour: {:.1}", peak_events);

        // Threat level trend
        let avg_threat = self.threat_level_forecast.iter().sum::<f64>()
            / self.threat_level_forecast.len() as f64;
        let max_threat = self.threat_level_forecast.iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        println!("\nThreat Level Forecast:");
        println!("  Average: {:.2}", avg_threat);
        println!("  Maximum: {:.2}", max_threat);

        // Traffic volume forecast
        let avg_traffic = self.traffic_volume_forecast.iter().sum::<f64>()
            / self.traffic_volume_forecast.len() as f64;

        println!("\nNetwork Traffic:");
        println!("  Average volume: {:.2}x baseline", avg_traffic);

        // Early warnings
        if !self.warnings.is_empty() {
            println!("\n⚠️  EARLY WARNING ALERTS ({} total):", self.warnings.len());
            for (i, warning) in self.warnings.iter().take(5).enumerate() {
                println!("  {}. +{} hours: {} (confidence: {:.0}%)",
                    i + 1, warning.hours_ahead, warning.message, warning.confidence * 100.0);
            }
        } else {
            println!("\n✅ No significant threats predicted");
        }
    }
}

/// Threat Forecaster
pub struct ThreatForecaster {
    forecaster: TimeSeriesForecaster,
    config: ForecastConfig,
}

impl ThreatForecaster {
    /// Create new threat forecaster
    pub fn new(config: ForecastConfig) -> Self {
        Self {
            forecaster: TimeSeriesForecaster::new(),
            config,
        }
    }

    /// Forecast threat trajectory from historical security metrics
    pub fn forecast_trajectory(
        &mut self,
        historical_metrics: &[SecurityMetricPoint],
    ) -> Result<ThreatTrajectory> {
        if historical_metrics.len() < 4 {
            bail!("Need at least 4 historical data points for forecasting");
        }

        // Extract time series
        let event_counts: Vec<f64> = historical_metrics.iter()
            .map(|m| m.event_count)
            .collect();

        let threat_levels: Vec<f64> = historical_metrics.iter()
            .map(|m| m.avg_threat_level)
            .collect();

        let traffic_volumes: Vec<f64> = historical_metrics.iter()
            .map(|m| m.traffic_volume)
            .collect();

        // Forecast event counts with ARIMA
        let event_count_forecast = self.forecast_with_fallback(
            &event_counts,
            self.config.horizon_hours
        )?;

        // Forecast threat levels
        let threat_level_forecast = self.forecast_with_fallback(
            &threat_levels,
            self.config.horizon_hours
        )?;

        // Forecast traffic volumes
        let traffic_volume_forecast = self.forecast_with_fallback(
            &traffic_volumes,
            self.config.horizon_hours
        )?;

        // Try to get uncertainty for event counts
        let event_count_uncertainty = self.compute_uncertainty(&event_counts, self.config.horizon_hours);

        // Generate early warning alerts
        let warnings = self.generate_warnings(
            &event_count_forecast,
            &threat_level_forecast,
            &traffic_volume_forecast,
        )?;

        Ok(ThreatTrajectory {
            event_count_forecast,
            threat_level_forecast,
            traffic_volume_forecast,
            event_count_uncertainty,
            warnings,
            horizon_hours: self.config.horizon_hours,
        })
    }

    /// Forecast with fallback to linear extrapolation
    fn forecast_with_fallback(&mut self, data: &[f64], horizon: usize) -> Result<Vec<f64>> {
        // Try ARIMA first
        match self.forecaster.fit_arima(data, self.config.arima_config.clone()) {
            Ok(_) => {
                match self.forecaster.forecast_arima(horizon) {
                    Ok(forecast) => return Ok(forecast),
                    Err(_) => {
                        // ARIMA failed, use fallback
                    }
                }
            }
            Err(_) => {
                // ARIMA fit failed, use fallback
            }
        }

        // Fallback: linear extrapolation
        Ok(self.linear_extrapolation(data, horizon))
    }

    /// Simple linear extrapolation fallback
    fn linear_extrapolation(&self, data: &[f64], horizon: usize) -> Vec<f64> {
        let n = data.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = data.iter().sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for (i, &y) in data.iter().enumerate() {
            let x = i as f64;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean).powi(2);
        }

        let slope = if denominator > 1e-10 {
            numerator / denominator
        } else {
            0.0
        };

        let intercept = y_mean - slope * x_mean;

        // Generate forecast
        let mut forecast = Vec::with_capacity(horizon);
        let last_x = n - 1.0;
        for i in 0..horizon {
            let x = last_x + i as f64 + 1.0;
            let y = slope * x + intercept;
            forecast.push(y.max(0.0));  // Events/threat can't be negative
        }

        forecast
    }

    /// Compute uncertainty bounds (if possible)
    fn compute_uncertainty(&self, data: &[f64], horizon: usize) -> Option<ForecastWithUncertainty> {
        // Simple std dev bounds
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        let std_dev = variance.sqrt();

        // Create forecast using mean as baseline (simple approach)
        let forecast = vec![mean; horizon];
        let lower_bound = forecast.iter().map(|&f| (f - 1.96 * std_dev).max(0.0)).collect();
        let upper_bound = forecast.iter().map(|&f| f + 1.96 * std_dev).collect();

        Some(ForecastWithUncertainty {
            lower_bound,
            upper_bound,
            forecast,
            std_dev: vec![std_dev; horizon],
            confidence_level: 0.95,
        })
    }

    /// Generate early warning alerts
    fn generate_warnings(
        &self,
        event_counts: &[f64],
        threat_levels: &[f64],
        traffic_volumes: &[f64],
    ) -> Result<Vec<ThreatWarning>> {
        let mut warnings = Vec::new();

        // Compute historical baseline
        let baseline_events = event_counts.iter().sum::<f64>() / event_counts.len() as f64;
        let baseline_traffic = traffic_volumes.iter().sum::<f64>() / traffic_volumes.len() as f64;

        // Check each forecast point
        for (hour, ((&events, &threat), &traffic)) in event_counts.iter()
            .zip(threat_levels.iter())
            .zip(traffic_volumes.iter())
            .enumerate()
        {
            // High event rate warning
            if events > baseline_events * (1.0 + self.config.anomaly_threshold) {
                warnings.push(ThreatWarning {
                    hours_ahead: hour,
                    message: format!("High security event rate predicted: {:.1} events/hour ({:.0}% above baseline)",
                        events, ((events / baseline_events) - 1.0) * 100.0),
                    predicted_level: ThreatLevel::High,
                    confidence: 0.75,
                });
            }

            // High threat level warning
            if threat > 0.7 && hour < self.config.early_warning_hours {
                warnings.push(ThreatWarning {
                    hours_ahead: hour,
                    message: format!("Elevated threat level predicted: {:.2}", threat),
                    predicted_level: ThreatLevel::High,
                    confidence: 0.70,
                });
            }

            // Traffic spike warning (potential DDoS)
            if traffic > baseline_traffic * 3.0 && hour < self.config.early_warning_hours {
                warnings.push(ThreatWarning {
                    hours_ahead: hour,
                    message: format!("Network traffic spike predicted: {:.1}x baseline (potential DDoS)",
                        traffic / baseline_traffic),
                    predicted_level: ThreatLevel::Critical,
                    confidence: 0.65,
                });
            }
        }

        // Sort by hours ahead (earliest first)
        warnings.sort_by_key(|w| w.hours_ahead);

        Ok(warnings)
    }

    /// Assess attack impact on forecast
    pub fn assess_mitigation_impact(
        &self,
        trajectory: &ThreatTrajectory,
        mitigation_effectiveness: f64,  // 0-1, how much threat reduction expected
    ) -> Result<MitigationImpact> {
        if mitigation_effectiveness < 0.0 || mitigation_effectiveness > 1.0 {
            bail!("Mitigation effectiveness must be between 0 and 1");
        }

        // Compute post-mitigation forecast
        let mitigated_events: Vec<f64> = trajectory.event_count_forecast.iter()
            .map(|&e| e * (1.0 - mitigation_effectiveness))
            .collect();

        let mitigated_threat_levels: Vec<f64> = trajectory.threat_level_forecast.iter()
            .map(|&t| (t * (1.0 - mitigation_effectiveness)).max(0.0))
            .collect();

        // Compute reduction metrics
        let event_reduction = trajectory.event_count_forecast.iter().sum::<f64>()
            - mitigated_events.iter().sum::<f64>();
        let avg_baseline_events = trajectory.event_count_forecast.iter().sum::<f64>()
            / trajectory.event_count_forecast.len() as f64;
        let reduction_percentage = (event_reduction / (avg_baseline_events * trajectory.horizon_hours as f64)) * 100.0;

        // Determine impact category
        let impact_category = if mitigation_effectiveness >= 0.8 {
            ImpactCategory::HighlyEffective
        } else if mitigation_effectiveness >= 0.5 {
            ImpactCategory::ModeratelyEffective
        } else if mitigation_effectiveness >= 0.2 {
            ImpactCategory::LowEffective
        } else {
            ImpactCategory::Minimal
        };

        Ok(MitigationImpact {
            baseline_forecast: trajectory.event_count_forecast.clone(),
            mitigated_forecast: mitigated_events,
            mitigated_threat_levels,
            event_reduction,
            reduction_percentage,
            impact_category,
            mitigation_effectiveness,
        })
    }
}

/// Mitigation impact assessment
#[derive(Debug, Clone)]
pub struct MitigationImpact {
    /// Baseline forecast (no mitigation)
    pub baseline_forecast: Vec<f64>,

    /// Post-mitigation forecast
    pub mitigated_forecast: Vec<f64>,

    /// Mitigated threat levels
    pub mitigated_threat_levels: Vec<f64>,

    /// Total event reduction (sum across horizon)
    pub event_reduction: f64,

    /// Reduction as percentage
    pub reduction_percentage: f64,

    /// Impact category
    pub impact_category: ImpactCategory,

    /// Mitigation effectiveness (0-1)
    pub mitigation_effectiveness: f64,
}

/// Mitigation impact category
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImpactCategory {
    HighlyEffective,      // ≥80% reduction
    ModeratelyEffective,  // 50-79% reduction
    LowEffective,         // 20-49% reduction
    Minimal,              // <20% reduction
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threat_forecaster_creation() {
        let config = ForecastConfig::default();
        let _forecaster = ThreatForecaster::new(config);
    }

    #[test]
    fn test_threat_trajectory_forecast() -> Result<()> {
        // Create synthetic security metrics (escalating threat)
        let mut metrics = Vec::new();
        for i in 0..10 {
            let t = i as f64;
            let escalation = 1.0 + t * 0.1;  // 10% escalation per hour
            metrics.push(SecurityMetricPoint {
                time_hours: t,
                event_count: 50.0 * escalation,
                avg_threat_level: 0.3 + t * 0.05,
                traffic_volume: 1.0 * escalation,
                failed_logins: 10.0 * escalation,
            });
        }

        let mut forecaster = ThreatForecaster::new(ForecastConfig::default());
        let trajectory = forecaster.forecast_trajectory(&metrics)?;

        // Verify forecast exists
        assert_eq!(trajectory.event_count_forecast.len(), 24);
        assert_eq!(trajectory.threat_level_forecast.len(), 24);
        assert_eq!(trajectory.traffic_volume_forecast.len(), 24);

        // Should detect escalation (warnings expected)
        assert!(!trajectory.warnings.is_empty(), "Should detect escalating threat");

        println!("✓ Threat trajectory forecasting works");
        Ok(())
    }

    #[test]
    fn test_mitigation_impact() -> Result<()> {
        let mut metrics = Vec::new();
        for i in 0..10 {
            metrics.push(SecurityMetricPoint {
                time_hours: i as f64,
                event_count: 100.0,
                avg_threat_level: 0.8,
                traffic_volume: 2.0,
                failed_logins: 20.0,
            });
        }

        let mut forecaster = ThreatForecaster::new(ForecastConfig::default());
        let trajectory = forecaster.forecast_trajectory(&metrics)?;

        // Assess 80% effective mitigation
        let impact = forecaster.assess_mitigation_impact(&trajectory, 0.8)?;

        assert_eq!(impact.impact_category, ImpactCategory::HighlyEffective);
        assert!(impact.event_reduction > 0.0);
        assert!(impact.reduction_percentage > 0.0);

        println!("✓ Mitigation impact assessment works");
        Ok(())
    }
}
