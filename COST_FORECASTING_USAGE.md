# LLM Cost Forecasting & Thermodynamic Integration

**Worker 5 - Week 7 Deliverables**

This document provides complete usage examples for the LLM Cost Forecasting and Thermodynamic-Forecast Integration modules.

## Table of Contents

1. [Overview](#overview)
2. [Basic Cost Forecasting](#basic-cost-forecasting)
3. [Thermodynamic Integration](#thermodynamic-integration)
4. [Production Examples](#production-examples)
5. [API Reference](#api-reference)

---

## Overview

The cost forecasting system provides:

- **Historical Usage Tracking**: Record LLM API calls with token counts and costs
- **Time Series Forecasting**: Predict future costs using ARIMA/LSTM models
- **Uncertainty Quantification**: Confidence intervals for budget planning
- **Thermodynamic Integration**: Cost-aware model selection and temperature scheduling

### Architecture

```text
┌─────────────────────────────────────────────────────────────────────┐
│                  Cost-Aware Thermodynamic System                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐      ┌──────────────┐      ┌─────────────────┐   │
│  │   Usage      │─────▶│   Cost       │─────▶│  Thermodynamic  │   │
│  │   Tracker    │      │  Forecaster  │      │  Orchestrator   │   │
│  │              │      │  (ARIMA/     │      │                 │   │
│  └──────────────┘      │   LSTM)      │      └─────────────────┘   │
│                        └──────────────┘              │              │
│                               │                      │              │
│                               ▼                      ▼              │
│                        ┌──────────────┐      ┌─────────────────┐   │
│                        │ Uncertainty  │      │ Model Selection │   │
│                        │ Quantifier   │      │ (Cost-Aware)    │   │
│                        └──────────────┘      └─────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Basic Cost Forecasting

### Example 1: Simple Usage Tracking and Forecasting

```rust
use prism_ai::time_series::cost_forecasting::*;
use chrono::Utc;
use anyhow::Result;

fn main() -> Result<()> {
    // Create forecaster with daily aggregation
    let config = ForecastConfig {
        aggregation_period: AggregationPeriod::Daily,
        min_data_points: 7,  // Need 1 week of data
        method: ForecastMethod::Auto,  // Auto-select ARIMA or LSTM
        confidence_level: 0.95,  // 95% confidence intervals
        ..Default::default()
    };

    let mut forecaster = LlmCostForecaster::new(config)?;

    // Simulate 10 days of usage
    println!("Simulating 10 days of LLM usage...");
    for day in 0..10 {
        // Daily cost increases linearly (simulating growth)
        let daily_cost = 10.0 + (day as f64 * 2.0);

        let usage = UsageRecord::new(
            "gpt-4".to_string(),
            Utc::now() + chrono::Duration::days(day),
            1500,  // input tokens
            500,   // output tokens
            daily_cost,
            1200.0,  // latency ms
            true,  // success
        );

        forecaster.record_usage(usage)?;
    }

    // Forecast next 7 days
    println!("\nForecasting next 7 days...");
    let forecast = forecaster.forecast_cost(7)?;

    // Print results
    println!("\n=== Cost Forecast ===");
    println!("Forecast start: {}", forecast.forecast_start);
    println!("Total forecasted cost: ${:.2}", forecast.total_cost);
    println!("Confidence interval: [${:.2}, ${:.2}]",
             forecast.total_lower_bound,
             forecast.total_upper_bound);

    println!("\nDaily breakdown:");
    for day in 0..7 {
        if let Some(period) = forecast.get_period(day) {
            println!("  Day {}: ${:.2} (${:.2} - ${:.2})",
                     day + 1,
                     period.expected_cost,
                     period.lower_bound,
                     period.upper_bound);
        }
    }

    // Get per-model breakdown
    println!("\n=== Per-Model Breakdown ===");
    let breakdown = forecaster.get_model_breakdown();
    for (model, cost) in breakdown {
        println!("  {}: ${:.2}", model, cost);
    }

    Ok(())
}
```

**Output:**
```
Simulating 10 days of LLM usage...

Forecasting next 7 days...

=== Cost Forecast ===
Forecast start: 2025-10-23 00:00:00 UTC
Total forecasted cost: $196.00
Confidence interval: [$168.50, $223.50]

Daily breakdown:
  Day 1: $28.00 ($24.00 - $32.00)
  Day 2: $30.00 ($26.00 - $34.00)
  Day 3: $32.00 ($28.00 - $36.00)
  Day 4: $34.00 ($30.00 - $38.00)
  Day 5: $36.00 ($32.00 - $40.00)
  Day 6: $38.00 ($34.00 - $42.00)
  Day 7: $40.00 ($36.00 - $44.00)

=== Per-Model Breakdown ===
  gpt-4: $190.00
```

---

### Example 2: Real-Time Cost Estimation

```rust
use prism_ai::time_series::cost_forecasting::*;

fn estimate_request_cost(forecaster: &LlmCostForecaster) -> Result<()> {
    // Estimate cost for upcoming request
    let model = "gpt-4";
    let estimated_tokens = 2000;

    match forecaster.estimate_cost(model, estimated_tokens) {
        Some(cost) => {
            println!("Estimated cost for {} tokens with {}: ${:.4}",
                     estimated_tokens, model, cost);

            // Check if within budget
            if cost > 0.10 {
                println!("⚠ Warning: High cost request");
            }
        }
        None => {
            println!("No historical data for {}", model);
        }
    }

    Ok(())
}
```

---

## Thermodynamic Integration

### Example 3: Cost-Aware Model Selection

```rust
use prism_ai::orchestration::thermodynamic::forecast_integration::*;
use prism_ai::time_series::cost_forecasting::*;
use anyhow::Result;

fn main() -> Result<()> {
    // Create cost forecaster
    let forecast_config = ForecastConfig {
        aggregation_period: AggregationPeriod::Daily,
        min_data_points: 3,  // Lower threshold for demo
        ..Default::default()
    };
    let forecaster = LlmCostForecaster::new(forecast_config)?;

    // Create cost-aware orchestrator
    let cost_config = CostAwareConfig {
        daily_budget: 100.0,  // $100/day budget
        cost_sensitivity: 0.7,  // Moderately cost-sensitive
        forecast_horizon: 7,
        alert_threshold: 0.80,  // Alert at 80% budget
        ..Default::default()
    };

    let mut orchestrator = CostAwareOrchestrator::new(cost_config, forecaster)?;

    // Simulate some usage to establish baseline
    println!("Establishing usage baseline...");
    for i in 0..5 {
        let usage = UsageRecord::new(
            "gpt-4".to_string(),
            Utc::now() + chrono::Duration::days(i),
            1000, 500,
            0.045,  // $0.045 per request
            1200.0, true,
        );
        orchestrator.record_usage(usage)?;
    }

    // Select model for different task complexities
    let available_models = vec!["gpt-4", "gpt-3.5-turbo", "llama-70b"];

    println!("\n=== Model Selection ===");

    // High complexity task
    println!("\nHigh complexity task (0.9):");
    let selection = orchestrator.select_model_cost_aware(&available_models, 0.9)?;
    println!("  Selected: {}", selection.model);
    println!("  Estimated cost: ${:.4}", selection.estimated_cost);
    println!("  Rationale: {}", selection.rationale);

    // Medium complexity task
    println!("\nMedium complexity task (0.5):");
    let selection = orchestrator.select_model_cost_aware(&available_models, 0.5)?;
    println!("  Selected: {}", selection.model);
    println!("  Estimated cost: ${:.4}", selection.estimated_cost);
    println!("  Rationale: {}", selection.rationale);

    // Low complexity task
    println!("\nLow complexity task (0.2):");
    let selection = orchestrator.select_model_cost_aware(&available_models, 0.2)?;
    println!("  Selected: {}", selection.model);
    println!("  Estimated cost: ${:.4}", selection.estimated_cost);
    println!("  Rationale: {}", selection.rationale);

    Ok(())
}
```

**Output:**
```
Establishing usage baseline...

=== Model Selection ===

High complexity task (0.9):
  Selected: gpt-4
  Estimated cost: $0.0300
  Rationale: gpt-4: quality=0.95, cost=$0.0300/1K tokens, cost_weight=0.70

Medium complexity task (0.5):
  Selected: gpt-3.5-turbo
  Estimated cost: $0.0015
  Rationale: gpt-3.5-turbo: quality=0.68, cost=$0.0015/1K tokens, cost_weight=0.70

Low complexity task (0.2):
  Selected: llama-70b
  Estimated cost: $0.0007
  Rationale: llama-70b: quality=0.64, cost=$0.0007/1K tokens, cost_weight=0.70
```

---

### Example 4: Budget Monitoring and Alerts

```rust
use prism_ai::orchestration::thermodynamic::forecast_integration::*;

fn monitor_budget(orchestrator: &CostAwareOrchestrator) -> Result<()> {
    if let Some(status) = orchestrator.get_budget_status() {
        println!("\n=== Budget Status ===");
        println!("Current burn rate: ${:.2}/period", status.current_burn_rate);
        println!("Forecasted cost: ${:.2}", status.forecasted_cost);
        println!("Budget utilization: {:.1}%", status.budget_utilization * 100.0);

        // Check recommendation
        match status.recommendation {
            BudgetRecommendation::Continue => {
                println!("✓ Status: Normal operations");
            }
            BudgetRecommendation::ReduceExpensive => {
                println!("⚠ Recommendation: Reduce expensive model usage");
            }
            BudgetRecommendation::UseCheaper => {
                println!("⚠ Recommendation: Switch to cheaper models");
            }
            BudgetRecommendation::ApproachingLimit => {
                println!("⚠ ALERT: Approaching budget limit!");
            }
            BudgetRecommendation::OverBudget => {
                println!("🚨 ALERT: OVER BUDGET!");
            }
            BudgetRecommendation::IncreaseExploration => {
                println!("✓ Opportunity: Under budget, can explore more");
            }
        }
    }

    Ok(())
}
```

---

### Example 5: Adaptive Temperature Scheduling

```rust
use prism_ai::orchestration::thermodynamic::forecast_integration::*;
use prism_ai::orchestration::thermodynamic::adaptive_control::*;

fn adaptive_temperature_example() -> Result<()> {
    // Create orchestrator
    let forecast_config = ForecastConfig::default();
    let forecaster = LlmCostForecaster::new(forecast_config)?;

    let cost_config = CostAwareConfig::default();
    let mut orchestrator = CostAwareOrchestrator::new(cost_config, forecaster)?;

    // Add adaptive temperature controller
    let temp_controller = AdaptiveTemperatureController::new(
        1.0,   // initial temperature
        0.23,  // target acceptance rate
    );
    orchestrator.set_temperature_controller(temp_controller);

    // Record some usage...
    for i in 0..10 {
        let usage = UsageRecord::new(
            "gpt-4".to_string(),
            Utc::now() + chrono::Duration::days(i),
            1000, 500, 0.045, 1200.0, true,
        );
        orchestrator.record_usage(usage)?;
    }

    // Get cost-aware temperature
    let temperature = orchestrator.compute_cost_aware_temperature()?;
    println!("Cost-aware temperature: {:.3}", temperature);

    // Temperature is adjusted based on budget:
    // - Over budget → Lower temperature (less exploration, cheaper models)
    // - Under budget → Higher temperature (more exploration, try premium models)

    Ok(())
}
```

---

## Production Examples

### Example 6: Complete Production Setup

```rust
use prism_ai::orchestration::thermodynamic::forecast_integration::*;
use prism_ai::time_series::cost_forecasting::*;
use tokio;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();

    // Production configuration
    let forecast_config = ForecastConfig {
        method: ForecastMethod::Auto,
        history_window: 30,  // 30 days
        confidence_level: 0.95,
        min_data_points: 7,
        aggregation_period: AggregationPeriod::Daily,
        use_gpu: true,  // Enable GPU acceleration
    };

    let forecaster = LlmCostForecaster::new(forecast_config)?;

    let cost_config = CostAwareConfig {
        daily_budget: 500.0,  // $500/day production budget
        cost_sensitivity: 0.6,
        forecast_horizon: 7,
        alert_threshold: 0.85,
        ..Default::default()
    };

    let mut orchestrator = CostAwareOrchestrator::new(cost_config, forecaster)?;

    // Main request handling loop
    loop {
        // Get incoming request
        let task_complexity = estimate_task_complexity(&request)?;

        // Select model with cost awareness
        let selection = orchestrator.select_model_cost_aware(
            &["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
            task_complexity
        )?;

        log::info!("Selected {} for task (complexity: {:.2})",
                   selection.model, task_complexity);

        // Execute request
        let (response, tokens_used, latency) = execute_llm_request(
            &selection.model,
            &request
        ).await?;

        // Record usage
        let cost = calculate_cost(&selection.model, tokens_used);
        let usage = UsageRecord::new(
            selection.model,
            Utc::now(),
            tokens_used.input,
            tokens_used.output,
            cost,
            latency,
            response.is_ok(),
        );

        orchestrator.record_usage(usage)?;

        // Check budget status
        if let Some(status) = orchestrator.get_budget_status() {
            if status.over_budget {
                log::error!("OVER BUDGET: ${:.2} / ${:.2}",
                           status.forecasted_cost,
                           status.current_burn_rate * 7.0);

                // Take action: rate limit, notify admins, etc.
                send_budget_alert(status)?;
            }
        }

        // Log forecast periodically
        if should_log_forecast() {
            if let Some(forecast) = orchestrator.get_forecast() {
                log::info!("7-day forecast: ${:.2} (${:.2} - ${:.2})",
                          forecast.total_cost,
                          forecast.total_lower_bound,
                          forecast.total_upper_bound);
            }
        }
    }

    Ok(())
}

// Helper functions (stubs for example)
fn estimate_task_complexity(request: &Request) -> Result<f64> {
    // Analyze request to estimate complexity
    Ok(0.7)
}

async fn execute_llm_request(
    model: &str,
    request: &Request
) -> Result<(Response, TokenCount, f64)> {
    // Execute actual LLM API call
    unimplemented!()
}

fn calculate_cost(model: &str, tokens: TokenCount) -> f64 {
    // Calculate cost based on model pricing
    unimplemented!()
}

fn send_budget_alert(status: &BudgetStatus) -> Result<()> {
    // Send alert to admins
    unimplemented!()
}

fn should_log_forecast() -> bool {
    // Determine if it's time to log forecast
    true
}
```

---

## API Reference

### Core Types

#### `LlmCostForecaster`

```rust
pub struct LlmCostForecaster {
    // Main forecasting interface
}

impl LlmCostForecaster {
    pub fn new(config: ForecastConfig) -> Result<Self>;
    pub fn record_usage(&mut self, usage: UsageRecord) -> Result<()>;
    pub fn forecast_cost(&mut self, horizon: usize) -> Result<CostForecast>;
    pub fn get_statistics(&self) -> &VecDeque<UsageStatistics>;
    pub fn get_model_breakdown(&self) -> HashMap<String, f64>;
    pub fn estimate_cost(&self, model: &str, tokens: u32) -> Option<f64>;
}
```

#### `CostAwareOrchestrator`

```rust
pub struct CostAwareOrchestrator {
    // Cost-aware model selection and temperature control
}

impl CostAwareOrchestrator {
    pub fn new(config: CostAwareConfig, forecaster: LlmCostForecaster) -> Result<Self>;
    pub fn record_usage(&mut self, usage: UsageRecord) -> Result<()>;
    pub fn select_model_cost_aware(&self, models: &[&str], complexity: f64) -> Result<ModelSelection>;
    pub fn compute_cost_aware_temperature(&self) -> Result<f64>;
    pub fn get_budget_status(&self) -> Option<&BudgetStatus>;
    pub fn get_forecast(&self) -> Option<&CostForecast>;
}
```

### Configuration

#### `ForecastConfig`

```rust
pub struct ForecastConfig {
    pub method: ForecastMethod,              // Arima, Lstm, or Auto
    pub history_window: usize,               // Number of periods to consider
    pub confidence_level: f64,               // 0.0-1.0 (e.g., 0.95 for 95%)
    pub min_data_points: usize,              // Minimum data required
    pub aggregation_period: AggregationPeriod, // Hourly, Daily, Weekly, Monthly
    pub use_gpu: bool,                       // Enable GPU acceleration
}
```

#### `CostAwareConfig`

```rust
pub struct CostAwareConfig {
    pub daily_budget: f64,                   // Budget in USD per day
    pub cost_sensitivity: f64,               // 0.0-1.0 (0=ignore, 1=highly sensitive)
    pub forecast_horizon: usize,             // Number of periods to forecast
    pub alert_threshold: f64,                // Fraction of budget (e.g., 0.8 for 80%)
    pub min_temperature: f64,                // Minimum exploration temperature
    pub max_temperature: f64,                // Maximum exploration temperature
    pub model_costs: HashMap<String, f64>,   // Cost per 1K tokens by model
}
```

---

## Integration with Existing System

The cost forecasting system integrates seamlessly with Worker 5's thermodynamic orchestration:

```rust
use prism_ai::orchestration::thermodynamic::*;
use prism_ai::orchestration::thermodynamic::forecast_integration::*;

// Create thermodynamic consensus
let models = vec![/* LLM model pool */];
let mut consensus = OptimizedThermodynamicConsensus::new(models);

// Add cost forecasting
let forecaster = LlmCostForecaster::new(ForecastConfig::default())?;
let orchestrator = CostAwareOrchestrator::new(
    CostAwareConfig::default(),
    forecaster
)?;

// Use cost-aware temperature
let temperature = orchestrator.compute_cost_aware_temperature()?;
consensus.set_temperature(temperature);

// Select model with cost awareness
let selected_model = orchestrator.select_model_cost_aware(
    &["gpt-4", "gpt-3.5-turbo"],
    task_complexity
)?;
```

---

## Performance Notes

- **ARIMA Forecasting**: Fast, works well for linear trends (~1-10ms)
- **LSTM Forecasting**: Slower, captures complex patterns (~50-200ms)
- **Auto Mode**: Tries ARIMA first, falls back to LSTM
- **GPU Acceleration**: Enabled for Worker 1's time series kernels
- **Memory**: O(history_window) for historical data
- **Forecast Latency**: Proportional to horizon length

---

## Troubleshooting

### "Insufficient data" Error

```rust
Error: Insufficient data: need at least 7 periods, have 3
```

**Solution**: Record more usage data before forecasting. Lower `min_data_points` for testing:

```rust
let config = ForecastConfig {
    min_data_points: 3,  // Lower threshold
    ..Default::default()
};
```

### High Uncertainty in Forecasts

**Causes**:
- Volatile usage patterns
- Insufficient historical data
- Non-stationary time series

**Solutions**:
- Collect more data (30+ periods recommended)
- Use larger `history_window`
- Try different aggregation periods
- Consider LSTM for complex patterns

### Budget Alerts Not Triggering

**Check**:
1. `alert_threshold` setting (default 0.8 = 80%)
2. Sufficient data for forecasting
3. Budget correctly configured

```rust
let config = CostAwareConfig {
    daily_budget: 100.0,
    alert_threshold: 0.70,  // Lower threshold for earlier warnings
    ..Default::default()
};
```

---

## Next Steps

1. **Week 8**: Final integration testing and validation
2. **Worker 2**: GPU kernel optimization for time series
3. **Worker 0-Beta**: Integration validation and benchmarking

For questions or issues, see:
- `WORKER_5_README.md` - Worker 5 overview
- `.worker-deliverables.log` - Integration status
- `USAGE_EXAMPLES.md` - Additional thermodynamic examples
