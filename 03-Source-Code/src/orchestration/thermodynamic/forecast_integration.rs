//! Thermodynamic-Forecast Integration
//!
//! Integrates LLM cost forecasting with thermodynamic orchestration to enable
//! proactive, cost-aware model selection and temperature scheduling.
//!
//! # Features
//!
//! - **Cost-Aware Model Selection**: Prefer cheaper models when budget is constrained
//! - **Proactive Temperature Adjustment**: Increase exploration when costs are low
//! - **Budget Alerts**: Warn when projected costs exceed budget
//! - **Cost-Quality Tradeoffs**: Balance model quality vs cost dynamically
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │          Thermodynamic-Forecast Integration                  │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                              │
//! │  ┌──────────────┐      ┌──────────────┐      ┌───────────┐ │
//! │  │     Cost     │─────▶│  Temperature │─────▶│   Model   │ │
//! │  │  Forecaster  │      │   Scheduler  │      │ Selector  │ │
//! │  │              │      │              │      │           │ │
//! │  └──────────────┘      └──────────────┘      └───────────┘ │
//! │        │                      │                     │       │
//! │        │                      ▼                     │       │
//! │        │              ┌──────────────┐              │       │
//! │        └─────────────▶│    Budget    │◀─────────────┘       │
//! │                       │   Monitor    │                      │
//! │                       └──────────────┘                      │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust
//! use prism_ai::orchestration::thermodynamic::forecast_integration::*;
//! use prism_ai::time_series::cost_forecasting::*;
//!
//! // Create forecaster
//! let forecaster = LlmCostForecaster::new(ForecastConfig::default())?;
//!
//! // Create cost-aware orchestrator
//! let config = CostAwareConfig {
//!     daily_budget: 100.0,  // $100/day
//!     cost_sensitivity: 0.5,  // Moderate cost sensitivity
//!     forecast_horizon: 7,  // 7 days ahead
//!     ..Default::default()
//! };
//!
//! let mut orchestrator = CostAwareOrchestrator::new(config, forecaster)?;
//!
//! // Select model with cost awareness
//! let models = vec!["gpt-4", "gpt-3.5", "llama-70b"];
//! let selected = orchestrator.select_model_cost_aware(&models, task_complexity)?;
//!
//! // Adjust temperature based on budget
//! let temp = orchestrator.compute_cost_aware_temperature()?;
//! ```

use anyhow::{Result, Context, bail};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

use crate::time_series::cost_forecasting::{
    LlmCostForecaster, UsageRecord, CostForecast
};
use crate::orchestration::thermodynamic::adaptive_temperature_control::AdaptiveTemperatureController;

/// Configuration for cost-aware orchestration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostAwareConfig {
    /// Daily budget in USD
    pub daily_budget: f64,
    /// Cost sensitivity factor (0.0-1.0)
    /// 0.0 = ignore costs, 1.0 = highly cost-sensitive
    pub cost_sensitivity: f64,
    /// Forecast horizon (number of periods)
    pub forecast_horizon: usize,
    /// Alert threshold (fraction of budget)
    pub alert_threshold: f64,
    /// Minimum temperature (even under budget constraints)
    pub min_temperature: f64,
    /// Maximum temperature (even with low costs)
    pub max_temperature: f64,
    /// Model cost estimates (cost per 1K tokens)
    pub model_costs: HashMap<String, f64>,
}

impl Default for CostAwareConfig {
    fn default() -> Self {
        let mut model_costs = HashMap::new();

        // Default cost estimates (per 1K tokens)
        model_costs.insert("gpt-4".to_string(), 0.03);
        model_costs.insert("gpt-4-turbo".to_string(), 0.01);
        model_costs.insert("gpt-3.5-turbo".to_string(), 0.0015);
        model_costs.insert("claude-3-opus".to_string(), 0.015);
        model_costs.insert("claude-3-sonnet".to_string(), 0.003);
        model_costs.insert("claude-3-haiku".to_string(), 0.00025);
        model_costs.insert("llama-70b".to_string(), 0.0007);
        model_costs.insert("mixtral-8x7b".to_string(), 0.0006);

        Self {
            daily_budget: 100.0,
            cost_sensitivity: 0.5,
            forecast_horizon: 7,
            alert_threshold: 0.8,  // Alert at 80% of budget
            min_temperature: 0.1,
            max_temperature: 2.0,
            model_costs,
        }
    }
}

/// Budget status and alerts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetStatus {
    /// Current burn rate (cost per period)
    pub current_burn_rate: f64,
    /// Forecasted total cost
    pub forecasted_cost: f64,
    /// Budget utilization (0.0-1.0+)
    pub budget_utilization: f64,
    /// Whether over budget
    pub over_budget: bool,
    /// Whether approaching budget limit
    pub approaching_limit: bool,
    /// Recommended action
    pub recommendation: BudgetRecommendation,
}

/// Budget-driven recommendations
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BudgetRecommendation {
    /// Continue normal operations
    Continue,
    /// Reduce usage of expensive models
    ReduceExpensive,
    /// Switch to cheaper models
    UseCheaper,
    /// Increase temperature (explore cheaper alternatives)
    IncreaseExploration,
    /// Alert: approaching budget limit
    ApproachingLimit,
    /// Alert: over budget
    OverBudget,
}

/// Model selection result with cost justification
#[derive(Debug, Clone)]
pub struct ModelSelection {
    /// Selected model name
    pub model: String,
    /// Estimated cost for this request
    pub estimated_cost: f64,
    /// Selection rationale
    pub rationale: String,
    /// Alternative models considered
    pub alternatives: Vec<(String, f64, String)>,
}

/// Cost-Aware Thermodynamic Orchestrator
pub struct CostAwareOrchestrator {
    /// Configuration
    config: CostAwareConfig,
    /// Cost forecaster
    forecaster: LlmCostForecaster,
    /// Temperature controller
    temperature_controller: Option<AdaptiveTemperatureController>,
    /// Latest forecast
    latest_forecast: Option<CostForecast>,
    /// Latest budget status
    latest_status: Option<BudgetStatus>,
}

impl CostAwareOrchestrator {
    /// Create new cost-aware orchestrator
    pub fn new(config: CostAwareConfig, forecaster: LlmCostForecaster) -> Result<Self> {
        Ok(Self {
            config,
            forecaster,
            temperature_controller: None,
            latest_forecast: None,
            latest_status: None,
        })
    }

    /// Set temperature controller for adaptive adjustment
    pub fn set_temperature_controller(&mut self, controller: AdaptiveTemperatureController) {
        self.temperature_controller = Some(controller);
    }

    /// Record LLM usage and update forecasts
    pub fn record_usage(&mut self, usage: UsageRecord) -> Result<()> {
        self.forecaster.record_usage(usage)?;

        // Update forecast
        self.update_forecast()?;

        // Update budget status
        self.update_budget_status()?;

        Ok(())
    }

    /// Update cost forecast
    fn update_forecast(&mut self) -> Result<()> {
        match self.forecaster.forecast_cost(self.config.forecast_horizon) {
            Ok(forecast) => {
                self.latest_forecast = Some(forecast);
                Ok(())
            }
            Err(e) => {
                // Not enough data yet - this is expected early on
                log::debug!("Could not update forecast: {}", e);
                Ok(())
            }
        }
    }

    /// Update budget status
    fn update_budget_status(&mut self) -> Result<()> {
        let forecast = match &self.latest_forecast {
            Some(f) => f,
            None => return Ok(()),  // No forecast available yet
        };

        let current_burn_rate = self.forecaster.get_burn_rate().unwrap_or(0.0);
        let forecasted_cost = forecast.total_cost;
        let daily_budget = self.config.daily_budget * self.config.forecast_horizon as f64;

        let budget_utilization = forecasted_cost / daily_budget;
        let over_budget = budget_utilization > 1.0;
        let approaching_limit = budget_utilization > self.config.alert_threshold;

        let recommendation = self.determine_recommendation(
            budget_utilization,
            over_budget,
            approaching_limit
        );

        self.latest_status = Some(BudgetStatus {
            current_burn_rate,
            forecasted_cost,
            budget_utilization,
            over_budget,
            approaching_limit,
            recommendation,
        });

        Ok(())
    }

    /// Determine budget recommendation
    fn determine_recommendation(
        &self,
        utilization: f64,
        over_budget: bool,
        approaching_limit: bool,
    ) -> BudgetRecommendation {
        if over_budget {
            BudgetRecommendation::OverBudget
        } else if utilization > 0.9 {
            BudgetRecommendation::UseCheaper
        } else if approaching_limit {
            BudgetRecommendation::ApproachingLimit
        } else if utilization > 0.7 {
            BudgetRecommendation::ReduceExpensive
        } else if utilization < 0.3 {
            BudgetRecommendation::IncreaseExploration
        } else {
            BudgetRecommendation::Continue
        }
    }

    /// Select model with cost awareness
    pub fn select_model_cost_aware(
        &self,
        available_models: &[&str],
        task_complexity: f64,  // 0.0-1.0
    ) -> Result<ModelSelection> {
        if available_models.is_empty() {
            bail!("No models available for selection");
        }

        let budget_status = self.latest_status.as_ref();

        // Calculate cost-quality score for each model
        let mut scored_models: Vec<(String, f64, f64, String)> = available_models.iter()
            .filter_map(|&model| {
                let cost = self.get_model_cost(model)?;
                let quality = self.estimate_model_quality(model, task_complexity);

                let cost_weight = self.compute_cost_weight(budget_status);
                let score = quality * (1.0 - cost_weight) - cost * cost_weight;

                let rationale = self.explain_score(model, quality, cost, cost_weight);

                Some((model.to_string(), score, cost, rationale))
            })
            .collect();

        if scored_models.is_empty() {
            bail!("No valid models found");
        }

        // Sort by score (descending)
        scored_models.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let best = scored_models[0].clone();
        let alternatives = scored_models.iter()
            .skip(1)
            .take(2)
            .map(|(name, _score, cost, rationale)| (name.clone(), *cost, rationale.clone()))
            .collect();

        Ok(ModelSelection {
            model: best.0,
            estimated_cost: best.2,
            rationale: best.3,
            alternatives,
        })
    }

    /// Get model cost per 1K tokens
    fn get_model_cost(&self, model: &str) -> Option<f64> {
        self.config.model_costs.get(model).copied()
    }

    /// Estimate model quality for task
    fn estimate_model_quality(&self, model: &str, task_complexity: f64) -> f64 {
        // Simple heuristic: larger/more expensive models are higher quality
        let base_quality = match model {
            m if m.contains("gpt-4") => 0.95,
            m if m.contains("claude-3-opus") => 0.95,
            m if m.contains("claude-3-sonnet") => 0.85,
            m if m.contains("gpt-3.5") => 0.75,
            m if m.contains("llama-70b") => 0.80,
            m if m.contains("mixtral") => 0.75,
            m if m.contains("claude-3-haiku") => 0.70,
            _ => 0.60,
        };

        // Adjust for task complexity
        // Complex tasks benefit more from high-quality models
        base_quality - (1.0 - base_quality) * (1.0 - task_complexity) * 0.5
    }

    /// Compute cost weight for model selection
    fn compute_cost_weight(&self, budget_status: Option<&BudgetStatus>) -> f64 {
        let base_weight = self.config.cost_sensitivity;

        match budget_status {
            Some(status) if status.over_budget => 0.9,  // Heavily prioritize cost
            Some(status) if status.budget_utilization > 0.8 => {
                base_weight + (1.0 - base_weight) * 0.5
            }
            Some(status) if status.budget_utilization < 0.3 => {
                base_weight * 0.5  // Reduce cost sensitivity when under budget
            }
            _ => base_weight,
        }
    }

    /// Explain model selection score
    fn explain_score(&self, model: &str, quality: f64, cost: f64, cost_weight: f64) -> String {
        format!(
            "{}: quality={:.2}, cost=${:.4}/1K tokens, cost_weight={:.2}",
            model, quality, cost, cost_weight
        )
    }

    /// Compute cost-aware temperature
    pub fn compute_cost_aware_temperature(&self) -> Result<f64> {
        let budget_status = self.latest_status.as_ref()
            .context("No budget status available")?;

        // Base temperature from controller (if available)
        let base_temp = match &self.temperature_controller {
            Some(controller) => controller.get_current_temperature(),
            None => 1.0,  // Default
        };

        // Adjust based on budget utilization
        let adjustment = self.compute_temperature_adjustment(budget_status);

        let adjusted_temp = base_temp * adjustment;

        // Clamp to configured range
        Ok(adjusted_temp.clamp(self.config.min_temperature, self.config.max_temperature))
    }

    /// Compute temperature adjustment factor
    fn compute_temperature_adjustment(&self, status: &BudgetStatus) -> f64 {
        match status.recommendation {
            BudgetRecommendation::OverBudget => 0.5,  // Reduce exploration
            BudgetRecommendation::UseCheaper => 0.7,
            BudgetRecommendation::ApproachingLimit => 0.8,
            BudgetRecommendation::ReduceExpensive => 0.9,
            BudgetRecommendation::Continue => 1.0,  // No adjustment
            BudgetRecommendation::IncreaseExploration => 1.3,  // Explore more
        }
    }

    /// Get latest budget status
    pub fn get_budget_status(&self) -> Option<&BudgetStatus> {
        self.latest_status.as_ref()
    }

    /// Get latest forecast
    pub fn get_forecast(&self) -> Option<&CostForecast> {
        self.latest_forecast.as_ref()
    }

    /// Get forecaster (for direct access)
    pub fn forecaster(&self) -> &LlmCostForecaster {
        &self.forecaster
    }

    /// Get forecaster (mutable, for recording usage)
    pub fn forecaster_mut(&mut self) -> &mut LlmCostForecaster {
        &mut self.forecaster
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time_series::cost_forecasting::{ForecastConfig, AggregationPeriod};
    use chrono::{Utc, Duration};

    fn create_test_orchestrator() -> Result<CostAwareOrchestrator> {
        let forecast_config = ForecastConfig {
            min_data_points: 3,  // Lower threshold for testing
            aggregation_period: AggregationPeriod::Daily,
            ..Default::default()
        };

        let forecaster = LlmCostForecaster::new(forecast_config)?;

        let cost_config = CostAwareConfig {
            daily_budget: 50.0,
            cost_sensitivity: 0.5,
            ..Default::default()
        };

        CostAwareOrchestrator::new(cost_config, forecaster)
    }

    #[test]
    fn test_orchestrator_creation() {
        let result = create_test_orchestrator();
        assert!(result.is_ok());
    }

    #[test]
    fn test_record_usage() {
        let mut orchestrator = create_test_orchestrator().unwrap();

        let usage = UsageRecord::new(
            "gpt-4".to_string(),
            Utc::now(),
            1000,
            500,
            0.045,
            1200.0,
            true,
        );

        let result = orchestrator.record_usage(usage);
        assert!(result.is_ok());
    }

    #[test]
    fn test_model_selection() {
        let orchestrator = create_test_orchestrator().unwrap();

        let models = vec!["gpt-4", "gpt-3.5-turbo", "llama-70b"];
        let selection = orchestrator.select_model_cost_aware(&models, 0.7);

        assert!(selection.is_ok());
        let sel = selection.unwrap();
        assert!(models.contains(&sel.model.as_str()));
    }

    #[test]
    fn test_model_selection_high_complexity() {
        let orchestrator = create_test_orchestrator().unwrap();

        let models = vec!["gpt-4", "gpt-3.5-turbo"];
        let selection = orchestrator.select_model_cost_aware(&models, 0.95);

        assert!(selection.is_ok());
        // High complexity should favor quality (gpt-4)
        let sel = selection.unwrap();
        assert_eq!(sel.model, "gpt-4");
    }

    #[test]
    fn test_cost_weight_over_budget() {
        let orchestrator = create_test_orchestrator().unwrap();

        let status = BudgetStatus {
            current_burn_rate: 60.0,
            forecasted_cost: 420.0,  // Over 7-day budget of 350
            budget_utilization: 1.2,
            over_budget: true,
            approaching_limit: true,
            recommendation: BudgetRecommendation::OverBudget,
        };

        let weight = orchestrator.compute_cost_weight(Some(&status));
        assert!(weight > 0.8);  // High cost sensitivity when over budget
    }

    #[test]
    fn test_temperature_adjustment() {
        let orchestrator = create_test_orchestrator().unwrap();

        let status_over = BudgetStatus {
            current_burn_rate: 60.0,
            forecasted_cost: 420.0,
            budget_utilization: 1.2,
            over_budget: true,
            approaching_limit: true,
            recommendation: BudgetRecommendation::OverBudget,
        };

        let adj_over = orchestrator.compute_temperature_adjustment(&status_over);
        assert!(adj_over < 1.0);  // Reduce temperature when over budget

        let status_under = BudgetStatus {
            current_burn_rate: 10.0,
            forecasted_cost: 70.0,
            budget_utilization: 0.2,
            over_budget: false,
            approaching_limit: false,
            recommendation: BudgetRecommendation::IncreaseExploration,
        };

        let adj_under = orchestrator.compute_temperature_adjustment(&status_under);
        assert!(adj_under > 1.0);  // Increase temperature when under budget
    }

    #[test]
    fn test_budget_recommendation() {
        let orchestrator = create_test_orchestrator().unwrap();

        // Over budget
        let rec_over = orchestrator.determine_recommendation(1.2, true, true);
        assert_eq!(rec_over, BudgetRecommendation::OverBudget);

        // Approaching limit
        let rec_approach = orchestrator.determine_recommendation(0.85, false, true);
        assert_eq!(rec_approach, BudgetRecommendation::ApproachingLimit);

        // Under budget
        let rec_under = orchestrator.determine_recommendation(0.25, false, false);
        assert_eq!(rec_under, BudgetRecommendation::IncreaseExploration);
    }
}
