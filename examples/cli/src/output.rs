/*!
 * Output formatting utilities
 */

use anyhow::Result;
use colored::Colorize;
use serde_json::Value;
use tabled::{Table, Tabled};

pub fn format_json(value: &Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_else(|_| "{}".to_string())
}

pub fn format_yaml(value: &Value) -> Result<String> {
    serde_yaml::to_string(value)
        .map_err(|e| anyhow::anyhow!("Failed to format as YAML: {}", e))
}

pub fn print_success(message: &str) {
    println!("{} {}", "✓".green().bold(), message);
}

pub fn print_error(message: &str) {
    eprintln!("{} {}", "✗".red().bold(), message);
}

pub fn print_info(message: &str) {
    println!("{} {}", "ℹ".blue().bold(), message);
}

pub fn format_table<T: Tabled>(data: Vec<T>) -> String {
    Table::new(data).to_string()
}

pub fn print_value(value: &Value, format: &str) -> Result<()> {
    match format {
        "json" => println!("{}", format_json(value)),
        "yaml" => println!("{}", format_yaml(value)?),
        "table" => {
            // For table format, try to extract data and format nicely
            if let Some(data) = value.get("data") {
                println!("{}", format_json(data));
            } else {
                println!("{}", format_json(value));
            }
        }
        _ => println!("{}", format_json(value)),
    }
    Ok(())
}
