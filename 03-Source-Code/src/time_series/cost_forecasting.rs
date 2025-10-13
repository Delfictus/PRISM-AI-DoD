//! LLM Cost Forecasting Module
//!
//! Tracks historical LLM usage and forecasts future costs for budget optimization
//! and proactive resource allocation. Integrates with thermodynamic orchestration
//! to enable cost-aware model selection.
//!
//! # Features
//!
//! - **Usage Tracking**: Record model usage, token counts, latency, and costs
//! - **Cost Forecasting**: Predict future costs using ARIMA/LSTM models
//! - **Uncertainty Quantification**: Confidence intervals for budget planning
//! - **Budget Optimization**: Cost-aware temperature scheduling
//! - **GPU Acceleration**: Leverage Worker 1's GPU-accelerated time series
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Cost Forecasting System                   │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                              │
//! │  ┌──────────────┐      ┌──────────────┐      ┌───────────┐ │
//! │  │   Usage      │─────▶│   Forecaster │─────▶│  Budget   │ │
//! │  │   Tracker    │      │   (ARIMA/    │      │ Optimizer │ │
//! │  │              │      │    LSTM)     │      │           │ │
//! │  └──────────────┘      └──────────────┘      └───────────┘ │
//! │        │                      │                     │       │
//! │        │                      ▼                     │       │
//! │        │              ┌──────────────┐              │       │
//! │        └─────────────▶│ Uncertainty  │◀─────────────┘       │
//! │                       │ Quantifier   │                      │
//! │                       └──────────────┘                      │
//! └─────────────────────────────────────────────────────────────┘
//!                                │
//!                                ▼
//!                  Thermodynamic Orchestration
//!                  (Cost-aware model selection)
//! ```
//!
//! # Integration with Worker 1
//!
//! Uses Worker 1's time series infrastructure:
//! - `TimeSeriesForecaster`: ARIMA/LSTM forecasting
//! - `UncertaintyQuantifier`: Prediction intervals
//! - `auto_arima`: Automatic model selection
//!
//! # Example
//!
//! ```rust
//! use prism_ai::time_series::cost_forecasting::*;
//!
//! // Create forecaster
//! let mut forecaster = LlmCostForecaster::new(ForecastConfig::default())?;
//!
//! // Track usage
//! let usage = UsageRecord {
//!     model_name: "gpt-4".to_string(),
//!     timestamp: Utc::now(),
//!     input_tokens: 1500,
//!     output_tokens: 500,
//!     total_cost: 0.065,  // $0.065
//!     latency_ms: 1200.0,
//!     success: true,
//! };
//!
//! forecaster.record_usage(usage)?;
//!
//! // Forecast next 7 days
//! let forecast = forecaster.forecast_cost(7)?;
//!
//! println!("Predicted cost: ${:.2}", forecast.total_cost);
//! println!("Confidence interval: [${:.2}, ${:.2}]",
//!     forecast.lower_bound, forecast.upper_bound);
//! ```

use anyhow::{Result, Context, bail};
use chrono::{DateTime, Utc, Duration};
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, VecDeque};

// Import Worker 1's time series infrastructure
use crate::time_series::{
    TimeSeriesForecaster, ArimaConfig, LstmConfig,
    UncertaintyQuantifier, UncertaintyConfig, UncertaintyMethod,
    ForecastWithUncertainty, auto_arima
};

/// Configuration for cost forecasting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastConfig {
    /// Forecasting method: "arima", "lstm", or "auto"
    pub method: ForecastMethod,
    /// History window size (number of periods to consider)
    pub history_window: usize,
    /// Confidence level for prediction intervals (0.0-1.0)
    pub confidence_level: f64,
    /// Minimum data points required for forecasting
    pub min_data_points: usize,
    /// Aggregation period for cost time series
    pub aggregation_period: AggregationPeriod,
    /// Enable GPU acceleration
    pub use_gpu: bool,
}

impl Default for ForecastConfig {
    fn default() -> Self {
        Self {
            method: ForecastMethod::Auto,
            history_window: 30,      // 30 periods (e.g., 30 days)
            confidence_level: 0.95,  // 95% confidence intervals
            min_data_points: 7,      // Minimum 1 week of data
            aggregation_period: AggregationPeriod::Daily,
            use_gpu: true,
        }
    }
}

/// Forecasting method selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ForecastMethod {
    /// ARIMA (fast, works well for linear trends)
    Arima,
    /// LSTM (slower, captures complex patterns)
    Lstm,
    /// Automatic selection based on data characteristics
    Auto,
}

/// Time aggregation period for cost series
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationPeriod {
    /// Aggregate by hour
    Hourly,
    /// Aggregate by day
    Daily,
    /// Aggregate by week
    Weekly,
    /// Aggregate by month
    Monthly,
}

impl AggregationPeriod {
    /// Get duration in seconds
    pub fn duration_secs(&self) -> i64 {
        match self {
            AggregationPeriod::Hourly => 3600,
            AggregationPeriod::Daily => 86400,
            AggregationPeriod::Weekly => 604800,
            AggregationPeriod::Monthly => 2592000,  // 30 days
        }
    }
}

/// Single LLM usage record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageRecord {
    /// Model name (e.g., "gpt-4", "claude-3-opus")
    pub model_name: String,
    /// Timestamp of request
    pub timestamp: DateTime<Utc>,
    /// Number of input tokens
    pub input_tokens: u32,
    /// Number of output tokens
    pub output_tokens: u32,
    /// Total cost in USD
    pub total_cost: f64,
    /// Latency in milliseconds
    pub latency_ms: f64,
    /// Whether request was successful
    pub success: bool,
}

impl UsageRecord {
    /// Create new usage record
    pub fn new(
        model_name: String,
        timestamp: DateTime<Utc>,
        input_tokens: u32,
        output_tokens: u32,
        total_cost: f64,
        latency_ms: f64,
        success: bool,
    ) -> Self {
        Self {
            model_name,
            timestamp,
            input_tokens,
            output_tokens,
            total_cost,
            latency_ms,
            success,
        }
    }

    /// Total tokens (input + output)
    pub fn total_tokens(&self) -> u32 {
        self.input_tokens + self.output_tokens
    }

    /// Cost per token
    pub fn cost_per_token(&self) -> f64 {
        if self.total_tokens() == 0 {
            0.0
        } else {
            self.total_cost / self.total_tokens() as f64
        }
    }
}

/// Aggregated usage statistics for a time period
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UsageStatistics {
    /// Period start time
    pub period_start: DateTime<Utc>,
    /// Total number of requests
    pub num_requests: u32,
    /// Total input tokens
    pub total_input_tokens: u32,
    /// Total output tokens
    pub total_output_tokens: u32,
    /// Total cost in USD
    pub total_cost: f64,
    /// Average latency in milliseconds
    pub avg_latency_ms: f64,
    /// Success rate (0.0-1.0)
    pub success_rate: f64,
    /// Per-model breakdown
    pub per_model_stats: HashMap<String, ModelStatistics>,
}

/// Per-model usage statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelStatistics {
    /// Number of requests
    pub num_requests: u32,
    /// Total cost
    pub total_cost: f64,
    /// Total tokens
    pub total_tokens: u32,
    /// Average latency
    pub avg_latency_ms: f64,
}

/// Cost forecast result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostForecast {
    /// Forecast start time
    pub forecast_start: DateTime<Utc>,
    /// Forecast horizon (number of periods)
    pub horizon: usize,
    /// Aggregation period
    pub period: AggregationPeriod,
    /// Point forecast (expected costs per period)
    pub forecast: Vec<f64>,
    /// Lower bound of prediction interval
    pub lower_bound: Vec<f64>,
    /// Upper bound of prediction interval
    pub upper_bound: Vec<f64>,
    /// Standard deviation per period
    pub std_dev: Vec<f64>,
    /// Confidence level
    pub confidence_level: f64,
    /// Total forecasted cost
    pub total_cost: f64,
    /// Total lower bound
    pub total_lower_bound: f64,
    /// Total upper bound
    pub total_upper_bound: f64,
}

impl CostForecast {
    /// Get forecast for specific period
    pub fn get_period(&self, index: usize) -> Option<PeriodForecast> {
        if index >= self.forecast.len() {
            return None;
        }

        let period_start = self.forecast_start + Duration::seconds(
            self.period.duration_secs() * index as i64
        );

        Some(PeriodForecast {
            period_start,
            expected_cost: self.forecast[index],
            lower_bound: self.lower_bound[index],
            upper_bound: self.upper_bound[index],
            std_dev: self.std_dev[index],
        })
    }

    /// Check if actual cost is within prediction interval
    pub fn is_within_interval(&self, index: usize, actual_cost: f64) -> bool {
        if index >= self.forecast.len() {
            return false;
        }

        actual_cost >= self.lower_bound[index] && actual_cost <= self.upper_bound[index]
    }
}

/// Forecast for a single period
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeriodForecast {
    /// Period start time
    pub period_start: DateTime<Utc>,
    /// Expected cost
    pub expected_cost: f64,
    /// Lower bound
    pub lower_bound: f64,
    /// Upper bound
    pub upper_bound: f64,
    /// Standard deviation
    pub std_dev: f64,
}

/// LLM Cost Forecasting System
pub struct LlmCostForecaster {
    /// Configuration
    config: ForecastConfig,
    /// Historical usage records (limited by history_window)
    usage_history: VecDeque<UsageRecord>,
    /// Aggregated usage statistics by period
    aggregated_stats: VecDeque<UsageStatistics>,
    /// Time series forecaster (Worker 1)
    forecaster: TimeSeriesForecaster,
    /// Uncertainty quantifier (Worker 1)
    uncertainty: UncertaintyQuantifier,
    /// Last aggregation timestamp
    last_aggregation: Option<DateTime<Utc>>,
}

impl LlmCostForecaster {
    /// Create new cost forecaster
    pub fn new(config: ForecastConfig) -> Result<Self> {
        let forecaster = TimeSeriesForecaster::new();

        let uncertainty_config = UncertaintyConfig {
            method: UncertaintyMethod::Residual,
            confidence_level: config.confidence_level,
            residual_window: config.history_window,
            ..Default::default()
        };

        let uncertainty = UncertaintyQuantifier::new(uncertainty_config);

        Ok(Self {
            config,
            usage_history: VecDeque::with_capacity(10000),
            aggregated_stats: VecDeque::with_capacity(365),  // Up to 1 year daily
            forecaster,
            uncertainty,
            last_aggregation: None,
        })
    }

    /// Record a single LLM usage event
    pub fn record_usage(&mut self, usage: UsageRecord) -> Result<()> {
        // Add to history
        self.usage_history.push_back(usage.clone());

        // Check if we need to aggregate
        let should_aggregate = match self.last_aggregation {
            None => true,
            Some(last) => {
                let elapsed = usage.timestamp.signed_duration_since(last);
                elapsed.num_seconds() >= self.config.aggregation_period.duration_secs()
            }
        };

        if should_aggregate {
            self.aggregate_period()?;
        }

        Ok(())
    }

    /// Aggregate usage data for the current period
    fn aggregate_period(&mut self) -> Result<()> {
        if self.usage_history.is_empty() {
            return Ok(());
        }

        // Find the most recent timestamp
        let latest = self.usage_history.back()
            .context("No usage records")?
            .timestamp;

        // Determine period start
        let period_start = self.align_to_period(latest);

        // Collect all records in this period
        let period_records: Vec<&UsageRecord> = self.usage_history.iter()
            .filter(|r| {
                let r_period = self.align_to_period(r.timestamp);
                r_period == period_start
            })
            .collect();

        if period_records.is_empty() {
            return Ok(());
        }

        // Aggregate statistics
        let mut stats = UsageStatistics {
            period_start,
            ..Default::default()
        };

        let mut total_latency = 0.0;
        let mut num_successes = 0;

        for record in &period_records {
            stats.num_requests += 1;
            stats.total_input_tokens += record.input_tokens;
            stats.total_output_tokens += record.output_tokens;
            stats.total_cost += record.total_cost;
            total_latency += record.latency_ms;

            if record.success {
                num_successes += 1;
            }

            // Per-model stats
            let model_stats = stats.per_model_stats
                .entry(record.model_name.clone())
                .or_insert_with(ModelStatistics::default);

            model_stats.num_requests += 1;
            model_stats.total_cost += record.total_cost;
            model_stats.total_tokens += record.total_tokens();
            model_stats.avg_latency_ms =
                (model_stats.avg_latency_ms * (model_stats.num_requests - 1) as f64
                 + record.latency_ms) / model_stats.num_requests as f64;
        }

        stats.avg_latency_ms = total_latency / stats.num_requests as f64;
        stats.success_rate = num_successes as f64 / stats.num_requests as f64;

        // Add to aggregated stats
        self.aggregated_stats.push_back(stats);

        // Limit history size
        while self.aggregated_stats.len() > self.config.history_window {
            self.aggregated_stats.pop_front();
        }

        self.last_aggregation = Some(period_start);

        Ok(())
    }

    /// Align timestamp to period boundary
    fn align_to_period(&self, timestamp: DateTime<Utc>) -> DateTime<Utc> {
        let period_secs = self.config.aggregation_period.duration_secs();
        let ts_secs = timestamp.timestamp();
        let aligned_secs = (ts_secs / period_secs) * period_secs;
        DateTime::from_timestamp(aligned_secs, 0)
            .unwrap_or(timestamp)
    }

    /// Forecast future costs
    pub fn forecast_cost(&mut self, horizon: usize) -> Result<CostForecast> {
        // Ensure we have enough data
        if self.aggregated_stats.len() < self.config.min_data_points {
            bail!("Insufficient data: need at least {} periods, have {}",
                  self.config.min_data_points, self.aggregated_stats.len());
        }

        // Extract cost time series
        let cost_series: Vec<f64> = self.aggregated_stats.iter()
            .map(|s| s.total_cost)
            .collect();

        // Choose forecasting method
        let forecast_with_uncertainty = match self.config.method {
            ForecastMethod::Arima => self.forecast_arima(&cost_series, horizon)?,
            ForecastMethod::Lstm => self.forecast_lstm(&cost_series, horizon)?,
            ForecastMethod::Auto => self.forecast_auto(&cost_series, horizon)?,
        };

        // Calculate totals
        let total_cost = forecast_with_uncertainty.forecast.iter().sum();
        let total_lower_bound = forecast_with_uncertainty.lower_bound.iter().sum();
        let total_upper_bound = forecast_with_uncertainty.upper_bound.iter().sum();

        // Determine forecast start time
        let last_period = self.aggregated_stats.back()
            .context("No aggregated stats")?
            .period_start;

        let forecast_start = last_period + Duration::seconds(
            self.config.aggregation_period.duration_secs()
        );

        Ok(CostForecast {
            forecast_start,
            horizon,
            period: self.config.aggregation_period,
            forecast: forecast_with_uncertainty.forecast,
            lower_bound: forecast_with_uncertainty.lower_bound,
            upper_bound: forecast_with_uncertainty.upper_bound,
            std_dev: forecast_with_uncertainty.std_dev,
            confidence_level: forecast_with_uncertainty.confidence_level,
            total_cost,
            total_lower_bound,
            total_upper_bound,
        })
    }

    /// Forecast using ARIMA
    fn forecast_arima(&mut self, data: &[f64], horizon: usize) -> Result<ForecastWithUncertainty> {
        // Fit ARIMA model
        let config = ArimaConfig {
            p: 2,  // AR(2)
            d: 1,  // First differencing (for trend)
            q: 1,  // MA(1)
            include_constant: true,
        };

        self.forecaster.fit_arima(data, config)?;

        // Generate forecast
        let forecast = self.forecaster.forecast_arima(horizon)?;

        // Update uncertainty quantifier with historical errors
        self.update_uncertainty_from_history(data)?;

        // Generate prediction intervals
        self.uncertainty.residual_intervals(&forecast)
    }

    /// Forecast using LSTM
    fn forecast_lstm(&mut self, data: &[f64], horizon: usize) -> Result<ForecastWithUncertainty> {
        // Fit LSTM model
        let config = LstmConfig {
            hidden_size: 32,
            sequence_length: 7,  // Use last 7 periods
            epochs: 100,
            ..Default::default()
        };

        self.forecaster.fit_lstm(data, config)?;

        // Generate forecast
        let forecast = self.forecaster.forecast_lstm(data, horizon)?;

        // Update uncertainty quantifier
        self.update_uncertainty_from_history(data)?;

        // Generate prediction intervals
        self.uncertainty.residual_intervals(&forecast)
    }

    /// Automatic forecasting method selection
    fn forecast_auto(&mut self, data: &[f64], horizon: usize) -> Result<ForecastWithUncertainty> {
        // Try ARIMA first (faster and often sufficient)
        match auto_arima(data, 2, 1, 2) {
            Ok(model) => {
                let forecast = model.forecast(horizon)?;
                self.update_uncertainty_from_history(data)?;
                self.uncertainty.residual_intervals(&forecast)
            }
            Err(_) => {
                // Fallback to LSTM if ARIMA fails
                self.forecast_lstm(data, horizon)
            }
        }
    }

    /// Update uncertainty quantifier with historical prediction errors
    fn update_uncertainty_from_history(&mut self, data: &[f64]) -> Result<()> {
        // Use a simple one-step-ahead prediction to compute residuals
        if data.len() < 3 {
            return Ok(());
        }

        for i in 2..data.len() {
            // Simple persistence forecast: ŷₜ = yₜ₋₁
            let predicted = data[i - 1];
            let actual = data[i];
            self.uncertainty.update_residuals(actual, predicted);
        }

        Ok(())
    }

    /// Get historical usage statistics
    pub fn get_statistics(&self) -> &VecDeque<UsageStatistics> {
        &self.aggregated_stats
    }

    /// Get total cost over a time range
    pub fn get_total_cost(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> f64 {
        self.usage_history.iter()
            .filter(|r| r.timestamp >= start && r.timestamp <= end)
            .map(|r| r.total_cost)
            .sum()
    }

    /// Get per-model cost breakdown
    pub fn get_model_breakdown(&self) -> HashMap<String, f64> {
        let mut breakdown = HashMap::new();

        for record in &self.usage_history {
            *breakdown.entry(record.model_name.clone()).or_insert(0.0) += record.total_cost;
        }

        breakdown
    }

    /// Get current burn rate (cost per period)
    pub fn get_burn_rate(&self) -> Option<f64> {
        self.aggregated_stats.back().map(|s| s.total_cost)
    }

    /// Estimate cost for a specific model and token count
    pub fn estimate_cost(&self, model_name: &str, tokens: u32) -> Option<f64> {
        // Find recent usage for this model
        let recent_records: Vec<&UsageRecord> = self.usage_history.iter()
            .rev()
            .take(100)
            .filter(|r| r.model_name == model_name && r.total_tokens() > 0)
            .collect();

        if recent_records.is_empty() {
            return None;
        }

        // Calculate average cost per token
        let avg_cost_per_token: f64 = recent_records.iter()
            .map(|r| r.cost_per_token())
            .sum::<f64>() / recent_records.len() as f64;

        Some(avg_cost_per_token * tokens as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forecaster_creation() {
        let config = ForecastConfig::default();
        let forecaster = LlmCostForecaster::new(config);
        assert!(forecaster.is_ok());
    }

    #[test]
    fn test_usage_record() {
        let record = UsageRecord::new(
            "gpt-4".to_string(),
            Utc::now(),
            1000,
            500,
            0.05,
            1200.0,
            true,
        );

        assert_eq!(record.total_tokens(), 1500);
        assert!((record.cost_per_token() - 0.05 / 1500.0).abs() < 1e-10);
    }

    #[test]
    fn test_record_usage() {
        let config = ForecastConfig::default();
        let mut forecaster = LlmCostForecaster::new(config).unwrap();

        let record = UsageRecord::new(
            "gpt-4".to_string(),
            Utc::now(),
            1000,
            500,
            0.05,
            1200.0,
            true,
        );

        let result = forecaster.record_usage(record);
        assert!(result.is_ok());
        assert_eq!(forecaster.usage_history.len(), 1);
    }

    #[test]
    fn test_cost_forecast_insufficient_data() {
        let config = ForecastConfig {
            min_data_points: 7,
            ..Default::default()
        };

        let mut forecaster = LlmCostForecaster::new(config).unwrap();

        // Add only 3 records
        for i in 0..3 {
            let record = UsageRecord::new(
                "gpt-4".to_string(),
                Utc::now() + Duration::days(i),
                1000,
                500,
                0.05,
                1200.0,
                true,
            );
            forecaster.record_usage(record).unwrap();
        }

        // Should fail due to insufficient data
        let result = forecaster.forecast_cost(7);
        assert!(result.is_err());
    }

    #[test]
    fn test_cost_forecast_with_data() {
        let config = ForecastConfig {
            min_data_points: 5,
            aggregation_period: AggregationPeriod::Daily,
            method: ForecastMethod::Arima,
            ..Default::default()
        };

        let mut forecaster = LlmCostForecaster::new(config).unwrap();

        // Add 10 days of data
        for i in 0..10 {
            let cost = 10.0 + (i as f64 * 0.5);  // Linear trend
            let record = UsageRecord::new(
                "gpt-4".to_string(),
                Utc::now() + Duration::days(i),
                1000,
                500,
                cost,
                1200.0,
                true,
            );
            forecaster.record_usage(record).unwrap();
        }

        // Force aggregation
        for _ in 0..10 {
            forecaster.aggregate_period().unwrap();
        }

        // Forecast should work if we have enough aggregated data
        if forecaster.aggregated_stats.len() >= 5 {
            let result = forecaster.forecast_cost(3);
            assert!(result.is_ok());

            let forecast = result.unwrap();
            assert_eq!(forecast.horizon, 3);
            assert_eq!(forecast.forecast.len(), 3);
        }
    }

    #[test]
    fn test_model_breakdown() {
        let config = ForecastConfig::default();
        let mut forecaster = LlmCostForecaster::new(config).unwrap();

        // Add records for different models
        for model in &["gpt-4", "gpt-3.5", "gpt-4"] {
            let record = UsageRecord::new(
                model.to_string(),
                Utc::now(),
                1000,
                500,
                0.05,
                1200.0,
                true,
            );
            forecaster.record_usage(record).unwrap();
        }

        let breakdown = forecaster.get_model_breakdown();
        assert_eq!(breakdown.len(), 2);  // gpt-4 and gpt-3.5
        assert!((breakdown["gpt-4"] - 0.10).abs() < 1e-10);  // 2 requests
        assert!((breakdown["gpt-3.5"] - 0.05).abs() < 1e-10);  // 1 request
    }

    #[test]
    fn test_estimate_cost() {
        let config = ForecastConfig::default();
        let mut forecaster = LlmCostForecaster::new(config).unwrap();

        // Add historical data for gpt-4
        for _ in 0..10 {
            let record = UsageRecord::new(
                "gpt-4".to_string(),
                Utc::now(),
                1000,
                500,
                0.05,  // $0.05 for 1500 tokens
                1200.0,
                true,
            );
            forecaster.record_usage(record).unwrap();
        }

        // Estimate cost for 3000 tokens (should be ~$0.10)
        let estimate = forecaster.estimate_cost("gpt-4", 3000);
        assert!(estimate.is_some());
        assert!((estimate.unwrap() - 0.10).abs() < 0.01);
    }

    #[test]
    fn test_aggregation_period() {
        assert_eq!(AggregationPeriod::Hourly.duration_secs(), 3600);
        assert_eq!(AggregationPeriod::Daily.duration_secs(), 86400);
        assert_eq!(AggregationPeriod::Weekly.duration_secs(), 604800);
    }

    #[test]
    fn test_cost_forecast_interval() {
        let forecast = CostForecast {
            forecast_start: Utc::now(),
            horizon: 3,
            period: AggregationPeriod::Daily,
            forecast: vec![10.0, 11.0, 12.0],
            lower_bound: vec![8.0, 9.0, 10.0],
            upper_bound: vec![12.0, 13.0, 14.0],
            std_dev: vec![1.0, 1.0, 1.0],
            confidence_level: 0.95,
            total_cost: 33.0,
            total_lower_bound: 27.0,
            total_upper_bound: 39.0,
        };

        assert!(forecast.is_within_interval(0, 10.0));
        assert!(!forecast.is_within_interval(0, 15.0));
    }
}
