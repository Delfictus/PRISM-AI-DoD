//! PRISM-AI REST API Server
//!
//! Standalone binary for running the API server
//!
//! Usage:
//!   cargo run --bin api_server --features api_server
//!
//! Environment variables:
//!   API_HOST - Server host (default: 0.0.0.0)
//!   API_PORT - Server port (default: 8080)
//!   API_KEY  - API authentication key
//!   RUST_LOG - Logging level (default: info)

use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    log::info!("Starting PRISM-AI REST API Server");
    log::info!("Version: {}", prism_ai::VERSION);

    // Load configuration from environment
    let host = env::var("API_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port: u16 = env::var("API_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);

    let api_key = env::var("API_KEY").ok();

    let config = prism_ai::api_server::ApiConfig {
        host,
        port,
        auth_enabled: api_key.is_some(),
        api_key,
        cors_enabled: true,
        max_body_size: 10 * 1024 * 1024, // 10MB
        timeout_secs: 60,
    };

    log::info!("Configuration:");
    log::info!("  Host: {}", config.host);
    log::info!("  Port: {}", config.port);
    log::info!("  Auth: {}", if config.auth_enabled { "enabled" } else { "disabled" });
    log::info!("  CORS: {}", if config.cors_enabled { "enabled" } else { "disabled" });

    // Start the server
    log::info!("Server starting on {}:{}", config.host, config.port);
    log::info!("API documentation: http://{}:{}/docs",
        if config.host == "0.0.0.0" { "localhost" } else { &config.host },
        config.port
    );

    prism_ai::api_server::start_server(config).await?;

    Ok(())
}
