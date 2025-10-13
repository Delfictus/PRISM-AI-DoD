//! PRISM-AI REST API Server
//!
//! Provides HTTP REST and WebSocket endpoints for all PRISM-AI capabilities:
//! - PWSA threat detection and sensor fusion
//! - Finance portfolio optimization
//! - Telecom network optimization
//! - Robotics motion planning
//! - LLM orchestration
//! - Time series forecasting
//! - Pixel-level IR processing

pub mod routes;
pub mod websocket;
pub mod auth;
pub mod middleware;
pub mod models;
pub mod error;
pub mod info_theory;
pub mod advanced_info_theory;
pub mod kalman;
pub mod advanced_kalman;
pub mod portfolio;
pub mod rate_limit;
pub mod performance;
pub mod graphql_schema;
pub mod dual_api;

use axum::{
    Router,
    routing::{get, post},
    Extension,
    http::StatusCode,
};
use std::sync::Arc;
use tokio::sync::RwLock;
use tower::ServiceBuilder;
use tower_http::{
    trace::TraceLayer,
    cors::CorsLayer,
};

pub use error::{ApiError, Result};

/// API server configuration
#[derive(Debug, Clone)]
pub struct ApiConfig {
    /// Server host (default: 0.0.0.0)
    pub host: String,
    /// Server port (default: 8080)
    pub port: u16,
    /// Enable authentication
    pub auth_enabled: bool,
    /// API key for authentication
    pub api_key: Option<String>,
    /// Enable CORS
    pub cors_enabled: bool,
    /// Max request body size (bytes)
    pub max_body_size: usize,
    /// Request timeout (seconds)
    pub timeout_secs: u64,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            auth_enabled: true,
            api_key: None,
            cors_enabled: true,
            max_body_size: 10 * 1024 * 1024, // 10MB
            timeout_secs: 60,
        }
    }
}

/// Shared application state
#[derive(Clone)]
pub struct AppState {
    pub config: ApiConfig,
    // Add platform components as they're integrated
    // pub pwsa_platform: Arc<RwLock<PwsaFusionPlatform>>,
    // pub llm_orchestrator: Arc<RwLock<LLMOrchestrator>>,
}

impl AppState {
    pub fn new(config: ApiConfig) -> Self {
        Self { config }
    }
}

/// Build the API router with all routes
pub fn build_router(state: AppState) -> Router {
    let shared_state = Arc::new(state);

    let base_router = Router::new()
        // Health check
        .route("/health", get(health_check))
        .route("/", get(root_handler))

        // API v1 routes (REST)
        .nest("/api/v1/pwsa", routes::pwsa::routes())
        .nest("/api/v1/finance", routes::finance::routes())
        .nest("/api/v1/telecom", routes::telecom::routes())
        .nest("/api/v1/robotics", routes::robotics::routes())
        .nest("/api/v1/llm", routes::llm::routes())
        .nest("/api/v1/timeseries", routes::time_series::routes())
        .nest("/api/v1/pixels", routes::pixels::routes())
        .nest("/api/v1/gpu", routes::gpu_monitoring::routes())

        // WebSocket endpoint
        .route("/ws", get(websocket::ws_handler))

        // Apply state to REST routes
        .with_state(shared_state.clone());

    // GraphQL API (dual API support) - merged after state application
    let graphql_router = dual_api::routes(shared_state.clone());

    base_router
        .merge(graphql_router)
        // Apply middleware
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(CorsLayer::permissive())
        )
}

/// Health check endpoint
async fn health_check() -> (StatusCode, &'static str) {
    (StatusCode::OK, "PRISM-AI API Server - Healthy")
}

/// Root endpoint
async fn root_handler() -> (StatusCode, &'static str) {
    (StatusCode::OK, "PRISM-AI REST API v1.0 - Worker 8 Deployment")
}

/// Start the API server
pub async fn start_server(config: ApiConfig) -> Result<()> {
    let addr = format!("{}:{}", config.host, config.port);
    let state = AppState::new(config);
    let app = build_router(state);

    log::info!("Starting PRISM-AI API server on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| ApiError::ServerError(format!("Failed to bind to {}: {}", addr, e)))?;

    axum::serve(listener, app)
        .await
        .map_err(|e| ApiError::ServerError(format!("Server error: {}", e)))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ApiConfig::default();
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 8080);
        assert!(config.auth_enabled);
    }

    #[tokio::test]
    async fn test_health_check() {
        let (status, body) = health_check().await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("Healthy"));
    }
}
