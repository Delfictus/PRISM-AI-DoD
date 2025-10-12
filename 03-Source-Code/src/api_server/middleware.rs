//! API middleware for logging, rate limiting, etc.

use axum::{
    body::Body,
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::Response,
};
use std::time::{Duration, Instant};

/// Request ID middleware - adds unique ID to each request
pub async fn request_id_middleware(
    mut request: Request,
    next: Next,
) -> Response {
    let request_id = uuid::Uuid::new_v4().to_string();
    request.extensions_mut().insert(RequestId(request_id.clone()));

    let response = next.run(request).await;

    // Could add request_id to response headers here
    response
}

/// Request ID wrapper
#[derive(Debug, Clone)]
pub struct RequestId(pub String);

/// Timing middleware - logs request duration
pub async fn timing_middleware(
    request: Request,
    next: Next,
) -> Response {
    let start = Instant::now();
    let method = request.method().clone();
    let uri = request.uri().clone();

    let response = next.run(request).await;

    let duration = start.elapsed();
    log::info!(
        "{} {} - {}ms - {}",
        method,
        uri,
        duration.as_millis(),
        response.status()
    );

    response
}

/// Rate limiting middleware (simple token bucket)
pub struct RateLimiter {
    requests_per_second: u32,
    last_request: Instant,
    tokens: f64,
}

impl RateLimiter {
    pub fn new(requests_per_second: u32) -> Self {
        Self {
            requests_per_second,
            last_request: Instant::now(),
            tokens: requests_per_second as f64,
        }
    }

    pub fn check_rate_limit(&mut self) -> bool {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_request).as_secs_f64();

        // Refill tokens based on elapsed time
        self.tokens += elapsed * self.requests_per_second as f64;
        self.tokens = self.tokens.min(self.requests_per_second as f64);

        self.last_request = now;

        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

/// Request size limit middleware
pub async fn size_limit_middleware(
    request: Request,
    next: Next,
) -> std::result::Result<Response, StatusCode> {
    const MAX_SIZE: u64 = 10 * 1024 * 1024; // 10MB

    if let Some(content_length) = request.headers().get("content-length") {
        if let Ok(size) = content_length.to_str() {
            if let Ok(size) = size.parse::<u64>() {
                if size > MAX_SIZE {
                    log::warn!("Request too large: {} bytes", size);
                    return Err(StatusCode::PAYLOAD_TOO_LARGE);
                }
            }
        }
    }

    Ok(next.run(request).await)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limiter() {
        let mut limiter = RateLimiter::new(10); // 10 req/s

        // Should allow first request
        assert!(limiter.check_rate_limit());

        // Should allow multiple requests initially
        for _ in 0..9 {
            assert!(limiter.check_rate_limit());
        }

        // Should block after limit reached
        assert!(!limiter.check_rate_limit());
    }
}
