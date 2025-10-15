/*!
 * Health check command
 */

use anyhow::Result;
use crate::client::PrismClient;
use crate::output;

pub async fn execute(client: &PrismClient, format: &str) -> Result<()> {
    let response = client.get("/health").await?;

    if format == "table" {
        if let Some(status) = response.get("status").and_then(|v| v.as_str()) {
            if status == "healthy" {
                output::print_success(&format!("API is healthy (status: {})", status));
            } else {
                output::print_error(&format!("API status: {}", status));
            }
        }

        if let Some(version) = response.get("version").and_then(|v| v.as_str()) {
            println!("Version: {}", version);
        }

        if let Some(uptime) = response.get("uptime_seconds") {
            println!("Uptime: {} seconds", uptime);
        }
    } else {
        output::print_value(&response, format)?;
    }

    Ok(())
}
