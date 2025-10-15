/*!
 * PWSA command implementations
 */

use anyhow::{Context, Result};
use serde_json::{json, Value};
use crate::client::PrismClient;
use crate::output;
use crate::{PwsaCommands};

pub async fn execute(client: &PrismClient, cmd: PwsaCommands, format: &str) -> Result<()> {
    match cmd {
        PwsaCommands::Detect { sv_id, timestamp, ir_frame } => {
            detect_threat(client, sv_id, timestamp, &ir_frame, format).await
        }
        PwsaCommands::Fuse { sv_id, timestamp, sensors } => {
            fuse_sensors(client, sv_id, timestamp, &sensors, format).await
        }
        PwsaCommands::Predict { track_id, history, horizon, model } => {
            predict_trajectory(client, &track_id, &history, horizon, &model, format).await
        }
        PwsaCommands::Prioritize { threats, strategy } => {
            prioritize_threats(client, &threats, &strategy, format).await
        }
    }
}

async fn detect_threat(
    client: &PrismClient,
    sv_id: i32,
    timestamp: i64,
    ir_frame_path: &str,
    format: &str,
) -> Result<()> {
    let ir_frame: Value = load_json_file(ir_frame_path)?;

    let body = json!({
        "sv_id": sv_id,
        "timestamp": timestamp,
        "ir_frame": ir_frame,
    });

    let response = client.post("/api/v1/pwsa/detect", &body).await?;

    if format == "table" {
        output::print_success("Threat detection complete");
        if let Some(data) = response.get("data") {
            print_threat_summary(data);
        }
    } else {
        output::print_value(&response, format)?;
    }

    Ok(())
}

async fn fuse_sensors(
    client: &PrismClient,
    sv_id: i32,
    timestamp: i64,
    sensors_path: &str,
    format: &str,
) -> Result<()> {
    let sensors: Value = load_json_file(sensors_path)?;

    let body = json!({
        "sv_id": sv_id,
        "timestamp": timestamp,
        "sensors": sensors,
    });

    let response = client.post("/api/v1/pwsa/fuse", &body).await?;

    if format == "table" {
        output::print_success("Sensor fusion complete");
        if let Some(data) = response.get("data") {
            print_fusion_summary(data);
        }
    } else {
        output::print_value(&response, format)?;
    }

    Ok(())
}

async fn predict_trajectory(
    client: &PrismClient,
    track_id: &str,
    history_path: &str,
    horizon: i32,
    model: &str,
    format: &str,
) -> Result<()> {
    let history: Value = load_json_file(history_path)?;

    let body = json!({
        "track_id": track_id,
        "history": history,
        "prediction_horizon": horizon,
        "model": model,
    });

    let response = client.post("/api/v1/pwsa/predict", &body).await?;

    if format == "table" {
        output::print_success("Trajectory prediction complete");
        if let Some(data) = response.get("data") {
            print_prediction_summary(data);
        }
    } else {
        output::print_value(&response, format)?;
    }

    Ok(())
}

async fn prioritize_threats(
    client: &PrismClient,
    threats_path: &str,
    strategy: &str,
    format: &str,
) -> Result<()> {
    let threats: Value = load_json_file(threats_path)?;

    let body = json!({
        "threats": threats,
        "prioritization_strategy": strategy,
    });

    let response = client.post("/api/v1/pwsa/prioritize", &body).await?;

    if format == "table" {
        output::print_success("Threat prioritization complete");
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

fn print_threat_summary(data: &Value) {
    if let Some(threat_id) = data.get("threat_id").and_then(|v| v.as_str()) {
        println!("  Threat ID: {}", threat_id);
    }
    if let Some(threat_type) = data.get("threat_type").and_then(|v| v.as_str()) {
        println!("  Type: {}", threat_type);
    }
    if let Some(confidence) = data.get("confidence").and_then(|v| v.as_f64()) {
        println!("  Confidence: {:.1}%", confidence * 100.0);
    }
}

fn print_fusion_summary(data: &Value) {
    if let Some(num_tracks) = data.get("num_tracks").and_then(|v| v.as_i64()) {
        println!("  Tracks: {}", num_tracks);
    }
    if let Some(quality) = data.get("fusion_quality").and_then(|v| v.as_f64()) {
        println!("  Fusion Quality: {:.1}%", quality * 100.0);
    }
}

fn print_prediction_summary(data: &Value) {
    if let Some(confidence) = data.get("confidence").and_then(|v| v.as_f64()) {
        println!("  Confidence: {:.1}%", confidence * 100.0);
    }
    if let Some(time_to_impact) = data.get("time_to_impact").and_then(|v| v.as_f64()) {
        println!("  Time to Impact: {:.0}s", time_to_impact);
    }
}
