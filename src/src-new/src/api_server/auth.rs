//! API authentication and authorization

use axum::{
    extract::{Request, FromRequestParts},
    http::{header, StatusCode},
    middleware::Next,
    response::Response,
};
use axum::http::request::Parts;
use async_trait::async_trait;

use crate::api_server::{ApiError, Result};

/// API key authentication token
#[derive(Debug, Clone)]
pub struct ApiKey(pub String);

/// Extract API key from request headers
#[async_trait]
impl<S> FromRequestParts<S> for ApiKey
where
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self> {
        // Try Authorization header
        if let Some(auth_header) = parts.headers.get(header::AUTHORIZATION) {
            let auth_str = auth_header
                .to_str()
                .map_err(|_| ApiError::Unauthorized("Invalid authorization header".to_string()))?;

            // Support both "Bearer <token>" and raw token
            let token = if auth_str.starts_with("Bearer ") {
                auth_str.strip_prefix("Bearer ").unwrap()
            } else {
                auth_str
            };

            return Ok(ApiKey(token.to_string()));
        }

        // Try X-API-Key header
        if let Some(api_key_header) = parts.headers.get("X-API-Key") {
            let key = api_key_header
                .to_str()
                .map_err(|_| ApiError::Unauthorized("Invalid API key header".to_string()))?;
            return Ok(ApiKey(key.to_string()));
        }

        Err(ApiError::Unauthorized("Missing API key".to_string()))
    }
}

/// Middleware to validate API key
pub async fn auth_middleware(
    api_key: Result<ApiKey>,
    request: Request,
    next: Next,
) -> std::result::Result<Response, StatusCode> {
    match api_key {
        Ok(key) => {
            // TODO: Validate key against configured API key
            // For now, just check it's not empty
            if key.0.is_empty() {
                return Err(StatusCode::UNAUTHORIZED);
            }
            Ok(next.run(request).await)
        }
        Err(_) => Err(StatusCode::UNAUTHORIZED),
    }
}

/// Role-based access control
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Admin,
    User,
    ReadOnly,
}

impl Role {
    pub fn can_read(&self) -> bool {
        matches!(self, Role::Admin | Role::User | Role::ReadOnly)
    }

    pub fn can_write(&self) -> bool {
        matches!(self, Role::Admin | Role::User)
    }

    pub fn can_admin(&self) -> bool {
        matches!(self, Role::Admin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_role_permissions() {
        assert!(Role::Admin.can_read());
        assert!(Role::Admin.can_write());
        assert!(Role::Admin.can_admin());

        assert!(Role::User.can_read());
        assert!(Role::User.can_write());
        assert!(!Role::User.can_admin());

        assert!(Role::ReadOnly.can_read());
        assert!(!Role::ReadOnly.can_write());
        assert!(!Role::ReadOnly.can_admin());
    }
}
