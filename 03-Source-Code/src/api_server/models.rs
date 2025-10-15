//! Common API request/response models

use serde::{Deserialize, Serialize};

/// Standard API response wrapper
#[derive(Debug, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: T,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<ResponseMetadata>,
}

impl<T> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data,
            metadata: None,
        }
    }

    pub fn with_metadata(data: T, metadata: ResponseMetadata) -> Self {
        Self {
            success: true,
            data,
            metadata: Some(metadata),
        }
    }
}

/// Response metadata (timing, version, etc.)
#[derive(Debug, Serialize, Deserialize)]
pub struct ResponseMetadata {
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// API version
    pub api_version: String,
    /// Request ID for tracing
    pub request_id: String,
}

/// Pagination parameters
#[derive(Debug, Deserialize)]
pub struct PaginationParams {
    #[serde(default = "default_page")]
    pub page: usize,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_page() -> usize { 0 }
fn default_limit() -> usize { 20 }

/// Paginated response
#[derive(Debug, Serialize, Deserialize)]
pub struct PaginatedResponse<T> {
    pub items: Vec<T>,
    pub total: usize,
    pub page: usize,
    pub limit: usize,
    pub pages: usize,
}

impl<T> PaginatedResponse<T> {
    pub fn new(items: Vec<T>, total: usize, page: usize, limit: usize) -> Self {
        let pages = (total + limit - 1) / limit;
        Self {
            items,
            total,
            page,
            limit,
            pages,
        }
    }
}
