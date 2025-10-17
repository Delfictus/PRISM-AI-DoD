//! API error types and handling

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use std::fmt;

/// API error types
#[derive(Debug)]
pub enum ApiError {
    /// Invalid request (400)
    BadRequest(String),
    /// Unauthorized (401)
    Unauthorized(String),
    /// Forbidden (403)
    Forbidden(String),
    /// Not found (404)
    NotFound(String),
    /// Internal server error (500)
    ServerError(String),
    /// Service unavailable (503)
    ServiceUnavailable(String),
}

impl fmt::Display for ApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ApiError::BadRequest(msg) => write!(f, "Bad Request: {}", msg),
            ApiError::Unauthorized(msg) => write!(f, "Unauthorized: {}", msg),
            ApiError::Forbidden(msg) => write!(f, "Forbidden: {}", msg),
            ApiError::NotFound(msg) => write!(f, "Not Found: {}", msg),
            ApiError::ServerError(msg) => write!(f, "Server Error: {}", msg),
            ApiError::ServiceUnavailable(msg) => write!(f, "Service Unavailable: {}", msg),
        }
    }
}

impl std::error::Error for ApiError {}

/// API error response format
#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, error_type) = match &self {
            ApiError::BadRequest(_) => (StatusCode::BAD_REQUEST, "BadRequest"),
            ApiError::Unauthorized(_) => (StatusCode::UNAUTHORIZED, "Unauthorized"),
            ApiError::Forbidden(_) => (StatusCode::FORBIDDEN, "Forbidden"),
            ApiError::NotFound(_) => (StatusCode::NOT_FOUND, "NotFound"),
            ApiError::ServerError(_) => (StatusCode::INTERNAL_SERVER_ERROR, "ServerError"),
            ApiError::ServiceUnavailable(_) => (StatusCode::SERVICE_UNAVAILABLE, "ServiceUnavailable"),
        };

        let body = Json(ErrorResponse {
            error: error_type.to_string(),
            message: self.to_string(),
            details: None,
        });

        (status, body).into_response()
    }
}

pub type Result<T> = std::result::Result<T, ApiError>;

/// Convert anyhow errors to API errors
impl From<anyhow::Error> for ApiError {
    fn from(err: anyhow::Error) -> Self {
        ApiError::ServerError(err.to_string())
    }
}
