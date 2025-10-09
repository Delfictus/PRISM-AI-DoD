//! Core types for data ingestion

use std::collections::HashMap;
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Generic data point from any source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    /// Unix timestamp in milliseconds
    pub timestamp: i64,
    /// Numerical feature values
    pub values: Vec<f64>,
    /// Additional metadata (source, symbols, etc.)
    pub metadata: HashMap<String, String>,
}

impl DataPoint {
    /// Create a new data point
    pub fn new(timestamp: i64, values: Vec<f64>) -> Self {
        Self {
            timestamp,
            values,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata field
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Get dimensionality
    pub fn dimension(&self) -> usize {
        self.values.len()
    }
}

/// Information about a data source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceInfo {
    /// Human-readable name
    pub name: String,
    /// Type of data (e.g., "financial_trades", "sensor_readings")
    pub data_type: String,
    /// Expected sampling rate in Hz
    pub sampling_rate_hz: f64,
    /// Number of dimensions per data point
    pub dimensions: usize,
}

/// Generic data source trait
#[async_trait::async_trait]
pub trait DataSource: Send + Sync {
    /// Connect to the data source
    async fn connect(&mut self) -> Result<()>;

    /// Read a batch of data points
    async fn read_batch(&mut self) -> Result<Vec<DataPoint>>;

    /// Disconnect from the data source
    async fn disconnect(&mut self) -> Result<()>;

    /// Get source information
    fn get_source_info(&self) -> SourceInfo;

    /// Check if source is connected
    fn is_connected(&self) -> bool {
        true // Default implementation
    }
}
