//! Dual API Handler - REST + GraphQL
//!
//! Provides unified access to PRISM-AI capabilities through:
//! - REST API: Simple HTTP endpoints (existing)
//! - GraphQL API: Flexible queries and mutations (new)
//!
//! Routes:
//! - /graphql - GraphQL playground and endpoint
//! - /api/v1/* - REST endpoints (existing)

use axum::{
    response::{Html, IntoResponse},
    routing::get,
    Json, Router,
};
use async_graphql::{
    http::{playground_source, GraphQLPlaygroundConfig},
    EmptySubscription, Schema,
};
use async_graphql_axum::GraphQL;
use std::sync::Arc;

use crate::api_server::{graphql_schema, AppState};

/// GraphQL schema type
pub type ApiSchema = Schema<
    graphql_schema::QueryRoot,
    graphql_schema::MutationRoot,
    EmptySubscription,
>;

/// Configure dual API routes (REST + GraphQL)
pub fn routes(_state: Arc<AppState>) -> Router<Arc<AppState>> {
    let schema = graphql_schema::create_schema();

    Router::new()
        .route("/graphql",
            get(graphql_playground).post_service(GraphQL::new(schema))
        )
        .route("/graphql/schema", get(graphql_schema_sdl))
}

/// GraphQL playground UI
async fn graphql_playground() -> impl IntoResponse {
    Html(playground_source(GraphQLPlaygroundConfig::new("/graphql")))
}

/// GraphQL schema SDL (Schema Definition Language)
async fn graphql_schema_sdl() -> impl IntoResponse {
    let schema = graphql_schema::create_schema();
    Json(serde_json::json!({
        "sdl": schema.sdl(),
        "endpoints": {
            "playground": "/graphql",
            "endpoint": "/graphql",
            "schema": "/graphql/schema"
        }
    }))
}

/// API capability comparison
#[derive(Debug, serde::Serialize)]
pub struct ApiCapabilities {
    pub rest: RestCapabilities,
    pub graphql: GraphQLCapabilities,
}

#[derive(Debug, serde::Serialize)]
pub struct RestCapabilities {
    pub endpoints: Vec<String>,
    pub features: Vec<String>,
    pub advantages: Vec<String>,
}

#[derive(Debug, serde::Serialize)]
pub struct GraphQLCapabilities {
    pub queries: Vec<String>,
    pub mutations: Vec<String>,
    pub features: Vec<String>,
    pub advantages: Vec<String>,
}

/// Get API capabilities comparison
pub fn get_api_capabilities() -> ApiCapabilities {
    ApiCapabilities {
        rest: RestCapabilities {
            endpoints: vec![
                "POST /api/v1/timeseries/forecast".to_string(),
                "POST /api/v1/finance/optimize".to_string(),
                "POST /api/v1/robotics/plan".to_string(),
                "GET /api/v1/gpu/status".to_string(),
                "GET /api/v1/gpu/metrics".to_string(),
            ],
            features: vec![
                "Simple HTTP requests".to_string(),
                "Standard REST conventions".to_string(),
                "Easy to cache".to_string(),
                "Wide tool support".to_string(),
            ],
            advantages: vec![
                "Simplicity - easy to understand and use".to_string(),
                "Caching - HTTP caching works out of the box".to_string(),
                "Tooling - curl, Postman, etc work perfectly".to_string(),
                "Stateless - no session management needed".to_string(),
            ],
        },
        graphql: GraphQLCapabilities {
            queries: vec![
                "health".to_string(),
                "gpuStatus".to_string(),
                "forecastTimeSeries".to_string(),
                "optimizePortfolio".to_string(),
                "planRobotMotion".to_string(),
                "performanceMetrics".to_string(),
            ],
            mutations: vec![
                "submitForecast".to_string(),
                "submitPortfolioOptimization".to_string(),
                "submitMotionPlan".to_string(),
            ],
            features: vec![
                "Flexible queries".to_string(),
                "Nested data fetching".to_string(),
                "Type-safe contracts".to_string(),
                "Single endpoint".to_string(),
                "Introspection".to_string(),
            ],
            advantages: vec![
                "Flexibility - request exactly the data you need".to_string(),
                "Efficiency - single request for complex data".to_string(),
                "Type safety - strong schema validation".to_string(),
                "Introspection - self-documenting API".to_string(),
                "Versioning - no need for /v1, /v2 endpoints".to_string(),
            ],
        },
    }
}

/// API usage examples
pub fn get_usage_examples() -> serde_json::Value {
    serde_json::json!({
        "rest": {
            "time_series_forecast": {
                "method": "POST",
                "url": "/api/v1/timeseries/forecast",
                "body": {
                    "historical_data": [100.0, 102.0, 101.0, 105.0],
                    "horizon": 5,
                    "method": { "Arima": { "p": 2, "d": 1, "q": 1 } }
                },
                "example": "curl -X POST http://localhost:8080/api/v1/timeseries/forecast \\\n  -H 'Content-Type: application/json' \\\n  -d '{\"historical_data\":[100,102,101,105],\"horizon\":5,\"method\":{\"Arima\":{\"p\":2,\"d\":1,\"q\":1}}}'"
            },
            "portfolio_optimization": {
                "method": "POST",
                "url": "/api/v1/finance/optimize",
                "body": {
                    "assets": [
                        { "symbol": "AAPL", "expected_return": 0.12, "volatility": 0.20, "current_price": 150.0 }
                    ],
                    "objective": "MaximizeSharpe"
                }
            }
        },
        "graphql": {
            "time_series_forecast": {
                "query": "query ForecastTimeSeries($input: TimeSeriesForecastInput!) {\n  forecastTimeSeries(input: $input) {\n    predictions\n    method\n    horizon\n    confidenceIntervals {\n      lower\n      upper\n    }\n  }\n}",
                "variables": {
                    "input": {
                        "historicalData": [100.0, 102.0, 101.0, 105.0],
                        "horizon": 5,
                        "method": "ARIMA"
                    }
                },
                "example": "curl -X POST http://localhost:8080/graphql \\\n  -H 'Content-Type: application/json' \\\n  -d '{\"query\":\"query { forecastTimeSeries(input: {historicalData: [100,102,101,105], horizon: 5, method: \\\"ARIMA\\\"}) { predictions method } }\"}'"
            },
            "combined_query": {
                "query": "query Dashboard {\n  health {\n    status\n    version\n  }\n  gpuStatus {\n    available\n    utilizationPercent\n  }\n  performanceMetrics {\n    endpoint\n    avgResponseTimeMs\n    requestsPerSecond\n  }\n}",
                "description": "Single query fetching multiple resources - not possible with REST"
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_api_capabilities() {
        let capabilities = get_api_capabilities();
        assert!(!capabilities.rest.endpoints.is_empty());
        assert!(!capabilities.graphql.queries.is_empty());
        assert!(!capabilities.graphql.mutations.is_empty());
    }

    #[test]
    fn test_get_usage_examples() {
        let examples = get_usage_examples();
        assert!(examples["rest"]["time_series_forecast"].is_object());
        assert!(examples["graphql"]["time_series_forecast"].is_object());
    }

    #[tokio::test]
    async fn test_graphql_schema_creation() {
        let schema = graphql_schema::create_schema();
        let sdl = schema.sdl();
        assert!(sdl.contains("Query"));
        assert!(sdl.contains("Mutation"));
    }
}
