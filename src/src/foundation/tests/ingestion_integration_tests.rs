//! Integration tests for the ingestion engine
//!
//! Tests the complete ingestion pipeline with error handling and recovery

use platform_foundation::{
    CircuitBreaker, CircuitBreakerState, DataPoint, DataSource, IngestionEngine, IngestionError,
    RetryPolicy, SyntheticDataSource,
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Test source that fails after a certain number of calls
struct FailAfterNSource {
    counter: Arc<AtomicUsize>,
    fail_after: usize,
    name: String,
}

impl FailAfterNSource {
    fn new(fail_after: usize, name: String) -> Self {
        Self {
            counter: Arc::new(AtomicUsize::new(0)),
            fail_after,
            name,
        }
    }
}

#[async_trait::async_trait]
impl DataSource for FailAfterNSource {
    async fn connect(&mut self) -> anyhow::Result<()> {
        Ok(())
    }

    async fn read_batch(&mut self) -> anyhow::Result<Vec<DataPoint>> {
        let count = self.counter.fetch_add(1, Ordering::SeqCst);

        if count >= self.fail_after {
            return Err(anyhow::anyhow!("Failed after {} calls", self.fail_after));
        }

        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), self.name.clone());
        metadata.insert("counter".to_string(), count.to_string());

        Ok(vec![DataPoint {
            timestamp: chrono::Utc::now().timestamp_millis(),
            values: vec![count as f64],
            metadata,
        }])
    }

    async fn disconnect(&mut self) -> anyhow::Result<()> {
        Ok(())
    }

    fn get_source_info(&self) -> platform_foundation::SourceInfo {
        platform_foundation::SourceInfo {
            name: self.name.clone(),
            data_type: "test".to_string(),
            sampling_rate_hz: 100.0,
            dimensions: 1,
        }
    }

    fn is_connected(&self) -> bool {
        true
    }
}

#[tokio::test]
async fn test_basic_ingestion_flow() {
    let mut engine = IngestionEngine::new(100, 1000);

    let source = Box::new(SyntheticDataSource::sine_wave(3, 1.0));
    engine.start_source(source).await.unwrap();

    tokio::time::sleep(Duration::from_millis(200)).await;

    let batch = engine
        .get_batch(10, Duration::from_millis(500))
        .await
        .unwrap();

    assert!(!batch.is_empty());
    assert!(batch.len() <= 10);

    let stats = engine.get_stats().await;
    assert!(stats.total_points > 0);
    assert_eq!(stats.active_sources, 1);
}

#[tokio::test]
async fn test_multi_source_ingestion() {
    let mut engine = IngestionEngine::new(200, 2000);

    // Start three different sources
    let source1 = Box::new(SyntheticDataSource::sine_wave(5, 1.0));
    engine.start_source(source1).await.unwrap();

    let source2 = Box::new(SyntheticDataSource::random_walk(3));
    engine.start_source(source2).await.unwrap();

    let source3 = Box::new(SyntheticDataSource::gaussian(2));
    engine.start_source(source3).await.unwrap();

    tokio::time::sleep(Duration::from_millis(300)).await;

    let stats = engine.get_stats().await;
    assert_eq!(stats.active_sources, 3);
    assert!(stats.total_points > 0);

    // Verify we get data from multiple sources
    let batch = engine
        .get_batch(30, Duration::from_millis(500))
        .await
        .unwrap();

    let mut sources = std::collections::HashSet::new();
    for point in batch {
        if let Some(source) = point.metadata.get("source") {
            sources.insert(source.clone());
        }
    }

    assert!(sources.len() > 1, "Should have data from multiple sources");
}

#[tokio::test]
async fn test_circuit_breaker_integration() {
    let mut engine = IngestionEngine::with_retry_policy(
        100,
        1000,
        RetryPolicy {
            max_attempts: 2,
            initial_backoff_ms: 10,
            max_backoff_ms: 50,
            backoff_multiplier: 2.0,
        },
    );

    // This source will fail after 5 successful reads
    let source = Box::new(FailAfterNSource::new(5, "FailingSource".to_string()));
    engine.start_source(source).await.unwrap();

    // Let it run and accumulate errors
    tokio::time::sleep(Duration::from_millis(500)).await;

    let stats = engine.get_stats().await;
    println!("Circuit breaker test stats: {:?}", stats);

    // Check circuit breaker status
    if let Some(status) = engine
        .get_circuit_breaker_status("FailingSource")
        .await
    {
        println!("Circuit breaker status: {}", status);
        assert!(status.contains("state="));
    }
}

#[tokio::test]
async fn test_retry_policy_backoff() {
    let policy = RetryPolicy::default();

    assert_eq!(policy.backoff_delay(0), 100);
    assert_eq!(policy.backoff_delay(1), 100);
    assert_eq!(policy.backoff_delay(2), 200);
    assert_eq!(policy.backoff_delay(3), 400);
    assert_eq!(policy.backoff_delay(10), 5000); // Capped at max
}

#[tokio::test]
async fn test_circuit_breaker_state_machine() {
    let mut cb = CircuitBreaker::new(3, 100);

    assert_eq!(cb.state(), CircuitBreakerState::Closed);
    assert!(cb.is_closed());

    // Record failures
    cb.record_failure();
    assert_eq!(cb.state(), CircuitBreakerState::Closed);

    cb.record_failure();
    assert_eq!(cb.state(), CircuitBreakerState::Closed);

    cb.record_failure();
    assert_eq!(cb.state(), CircuitBreakerState::Open);
    assert!(!cb.is_closed());

    // Record success to reset
    cb.record_success();
    assert_eq!(cb.state(), CircuitBreakerState::Closed);
    assert!(cb.is_closed());
    assert_eq!(cb.error_count(), 0);
}

#[tokio::test]
async fn test_error_types() {
    let err1 = IngestionError::ConnectionFailed {
        source: "test".to_string(),
        reason: "timeout".to_string(),
        retryable: true,
    };
    assert!(err1.is_retryable());

    let err2 = IngestionError::CircuitBreakerOpen {
        source: "test".to_string(),
        error_count: 5,
        threshold: 3,
    };
    assert!(!err2.is_retryable());

    let err3 = IngestionError::Timeout {
        source: "test".to_string(),
        timeout_ms: 1000,
    };
    assert!(err3.is_retryable());
}

#[tokio::test]
async fn test_historical_buffer() {
    let mut engine = IngestionEngine::new(100, 500);

    let source = Box::new(SyntheticDataSource::sine_wave(2, 1.0));
    engine.start_source(source).await.unwrap();

    tokio::time::sleep(Duration::from_millis(300)).await;

    // Consume some data to populate the buffer
    let _ = engine
        .get_batch(20, Duration::from_millis(200))
        .await
        .unwrap();

    // Get history
    let history = engine.get_history(100).await;
    assert!(!history.is_empty());
    assert!(history.len() <= 100);

    // Verify timestamps are ordered
    for i in 1..history.len() {
        assert!(history[i].timestamp >= history[i - 1].timestamp);
    }
}

#[tokio::test]
async fn test_buffer_size_tracking() {
    let mut engine = IngestionEngine::new(100, 1000);

    let source = Box::new(SyntheticDataSource::gaussian(1));
    engine.start_source(source).await.unwrap();

    tokio::time::sleep(Duration::from_millis(200)).await;

    // Consume some data to populate the buffer
    let _ = engine
        .get_batch(10, Duration::from_millis(200))
        .await
        .unwrap();

    let buffer_size = engine.buffer_size().await;
    assert!(buffer_size > 0);

    // Clear buffer
    engine.clear_buffer().await;
    let buffer_size_after_clear = engine.buffer_size().await;
    assert_eq!(buffer_size_after_clear, 0);
}

#[tokio::test]
async fn test_ingestion_rate_calculation() {
    let mut engine = IngestionEngine::new(100, 1000);

    let source = Box::new(SyntheticDataSource::sine_wave(3, 2.0));
    engine.start_source(source).await.unwrap();

    tokio::time::sleep(Duration::from_secs(1)).await;

    let stats = engine.get_stats().await;
    assert!(stats.average_rate_hz > 0.0);
    println!("Ingestion rate: {:.1} points/sec", stats.average_rate_hz);
}

#[tokio::test]
async fn test_aggressive_retry_policy() {
    let aggressive = RetryPolicy::aggressive();

    assert_eq!(aggressive.max_attempts, 5);
    assert_eq!(aggressive.initial_backoff_ms, 50);
    assert_eq!(aggressive.max_backoff_ms, 10000);

    let backoff = aggressive.backoff_delay(4);
    assert!(backoff > aggressive.initial_backoff_ms);
}

#[tokio::test]
async fn test_no_retry_policy() {
    let no_retry = RetryPolicy::no_retry();

    assert_eq!(no_retry.max_attempts, 1);
    assert_eq!(no_retry.backoff_delay(0), 0);
}
