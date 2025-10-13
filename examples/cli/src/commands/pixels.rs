/*!
 * Pixel processing command implementations
 */

use anyhow::{Context, Result};
use serde_json::{json, Value};
use crate::client::PrismClient;
use crate::output;
use crate::PixelsCommands;

pub async fn execute(client: &PrismClient, cmd: PixelsCommands, format: &str) -> Result<()> {
    match cmd {
        PixelsCommands::Process {
            frame_id,
            width,
            height,
            pixels,
            detect_hotspots,
            compute_entropy,
            apply_tda,
        } => {
            process_pixels(
                client,
                frame_id,
                width,
                height,
                &pixels,
                detect_hotspots,
                compute_entropy,
                apply_tda,
                format,
            )
            .await
        }
    }
}

async fn process_pixels(
    client: &PrismClient,
    frame_id: i32,
    width: i32,
    height: i32,
    pixels_path: &str,
    detect_hotspots: bool,
    compute_entropy: bool,
    apply_tda: bool,
    format: &str,
) -> Result<()> {
    let pixels: Value = load_json_file(pixels_path)?;

    let mut processing_options = json!({});
    if detect_hotspots {
        processing_options["detect_hotspots"] = json!(true);
    }
    if compute_entropy {
        processing_options["compute_entropy"] = json!(true);
    }
    if apply_tda {
        processing_options["apply_tda"] = json!(true);
    }

    let body = json!({
        "frame_id": frame_id,
        "width": width,
        "height": height,
        "pixels": pixels,
        "processing_options": processing_options,
    });

    let response = client.post("/api/v1/pixels/process", &body).await?;

    if format == "table" {
        output::print_success("Pixel processing complete");
        if let Some(data) = response.get("data") {
            print_pixels_summary(data);
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

fn print_pixels_summary(data: &Value) {
    if let Some(hotspots) = data.get("hotspots").and_then(|v| v.as_array()) {
        println!("  Hotspots detected: {}", hotspots.len());
    }
    if let Some(entropy) = data.get("entropy").and_then(|v| v.as_f64()) {
        println!("  Entropy: {:.3}", entropy);
    }
    if let Some(tda_features) = data.get("tda_features").and_then(|v| v.as_array()) {
        println!("  TDA features: {}", tda_features.len());
    }
}
