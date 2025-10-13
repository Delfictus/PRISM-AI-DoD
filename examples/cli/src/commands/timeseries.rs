/*!
 * Time series command implementations
 */

use anyhow::{Context, Result};
use serde_json::{json, Value};
use crate::client::PrismClient;
use crate::output;
use crate::TimeseriesCommands;

pub async fn execute(client: &PrismClient, cmd: TimeseriesCommands, format: &str) -> Result<()> {
    match cmd {
        TimeseriesCommands::Forecast { series_id, data, horizon, method } => {
            forecast_timeseries(client, &series_id, &data, horizon, &method, format).await
        }
    }
}

async fn forecast_timeseries(
    client: &PrismClient,
    series_id: &str,
    data_path: &str,
    horizon: i32,
    method: &str,
    format: &str,
) -> Result<()> {
    let historical_data: Value = load_json_file(data_path)?;

    let body = json!({
        "series_id": series_id,
        "historical_data": historical_data,
        "horizon": horizon,
        "method": { method: {} },
    });

    let response = client.post("/api/v1/timeseries/forecast", &body).await?;

    if format == "table" {
        output::print_success("Time series forecast complete");
        if let Some(data) = response.get("data") {
            print_forecast_summary(data);
        }
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

fn print_forecast_summary(data: &Value) {
    if let Some(predictions) = data.get("predictions").and_then(|v| v.as_array()) {
        println!("  Predictions ({} values):", predictions.len());
        for (i, pred) in predictions.iter().take(10).enumerate() {
            if let Some(val) = pred.as_f64() {
                println!("    [{}]: {:.2}", i + 1, val);
            }
        }
        if predictions.len() > 10 {
            println!("    ... ({} more)", predictions.len() - 10);
        }
    }
}
