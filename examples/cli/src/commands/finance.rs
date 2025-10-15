/*!
 * Finance command implementations
 */

use anyhow::{Context, Result};
use serde_json::{json, Value};
use crate::client::PrismClient;
use crate::output;
use crate::FinanceCommands;

pub async fn execute(client: &PrismClient, cmd: FinanceCommands, format: &str) -> Result<()> {
    match cmd {
        FinanceCommands::Optimize { assets, constraints, objective } => {
            optimize_portfolio(client, &assets, &constraints, &objective, format).await
        }
        FinanceCommands::Risk { portfolio_id, positions, metrics } => {
            assess_risk(client, &portfolio_id, &positions, &metrics, format).await
        }
        FinanceCommands::Backtest { strategy_id, parameters, historical_data, initial_capital } => {
            backtest_strategy(client, &strategy_id, &parameters, &historical_data, initial_capital, format).await
        }
    }
}

async fn optimize_portfolio(
    client: &PrismClient,
    assets_path: &str,
    constraints_path: &str,
    objective: &str,
    format: &str,
) -> Result<()> {
    let assets: Value = load_json_file(assets_path)?;
    let constraints: Value = load_json_file(constraints_path)?;

    let body = json!({
        "assets": assets,
        "constraints": constraints,
        "objective": objective,
    });

    let response = client.post("/api/v1/finance/optimize", &body).await?;

    if format == "table" {
        output::print_success("Portfolio optimization complete");
        if let Some(data) = response.get("data") {
            print_portfolio_summary(data);
        }
    } else {
        output::print_value(&response, format)?;
    }

    Ok(())
}

async fn assess_risk(
    client: &PrismClient,
    portfolio_id: &str,
    positions_path: &str,
    metrics: &str,
    format: &str,
) -> Result<()> {
    let positions: Value = load_json_file(positions_path)?;
    let risk_metrics: Vec<String> = metrics.split(',').map(|s| s.trim().to_string()).collect();

    let body = json!({
        "portfolio_id": portfolio_id,
        "positions": positions,
        "risk_metrics": risk_metrics,
    });

    let response = client.post("/api/v1/finance/risk", &body).await?;

    if format == "table" {
        output::print_success("Risk assessment complete");
        if let Some(data) = response.get("data") {
            print_risk_summary(data);
        }
    } else {
        output::print_value(&response, format)?;
    }

    Ok(())
}

async fn backtest_strategy(
    client: &PrismClient,
    strategy_id: &str,
    parameters_path: &str,
    historical_data_path: &str,
    initial_capital: f64,
    format: &str,
) -> Result<()> {
    let parameters: Value = load_json_file(parameters_path)?;
    let historical_data: Value = load_json_file(historical_data_path)?;

    let body = json!({
        "strategy_id": strategy_id,
        "parameters": parameters,
        "historical_data": historical_data,
        "initial_capital": initial_capital,
    });

    let response = client.post("/api/v1/finance/backtest", &body).await?;

    if format == "table" {
        output::print_success("Strategy backtest complete");
    } else {
        output::print_value(&response, format)?;
    }

    Ok(())
}

fn load_json_file(path: &str) -> Result<Value> {
    let content = std::fs::read_to_string(path)
        .context(format!("Failed to read file: {}", path))?;
    serde_json::from_str(&content)
        .context("Failed to parse JSON")
}

fn print_portfolio_summary(data: &Value) {
    if let Some(expected_return) = data.get("expected_return").and_then(|v| v.as_f64()) {
        println!("  Expected Return: {:.2}%", expected_return * 100.0);
    }
    if let Some(expected_risk) = data.get("expected_risk").and_then(|v| v.as_f64()) {
        println!("  Expected Risk: {:.2}%", expected_risk * 100.0);
    }
    if let Some(sharpe_ratio) = data.get("sharpe_ratio").and_then(|v| v.as_f64()) {
        println!("  Sharpe Ratio: {:.2}", sharpe_ratio);
    }
}

fn print_risk_summary(data: &Value) {
    if let Some(var) = data.get("var").and_then(|v| v.as_f64()) {
        println!("  VaR: {:.2}%", var * 100.0);
    }
    if let Some(cvar) = data.get("cvar").and_then(|v| v.as_f64()) {
        println!("  CVaR: {:.2}%", cvar * 100.0);
    }
    if let Some(max_drawdown) = data.get("max_drawdown").and_then(|v| v.as_f64()) {
        println!("  Max Drawdown: {:.2}%", max_drawdown * 100.0);
    }
}
