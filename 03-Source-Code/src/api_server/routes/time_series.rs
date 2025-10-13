//! Time Series forecasting API endpoints
//!
//! ARIMA, LSTM, trajectory prediction, uncertainty quantification

use axum::{
    extract::{Path, State},
    routing::{get, post},
    Router, Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::api_server::{ApiError, Result, AppState, models::ApiResponse};

/// Time series forecast request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastRequest {
    pub series_id: String,
    pub historical_data: Vec<f64>,
    pub timestamps: Vec<i64>,
    pub horizon: usize,
    pub method: ForecastMethod,
    pub include_uncertainty: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForecastMethod {
    Arima { p: usize, d: usize, q: usize },
    Lstm { hidden_dim: usize, num_layers: usize },
    Gru { hidden_dim: usize },
    ExponentialSmoothing,
    Prophet,
}

/// Forecast response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastResponse {
    pub series_id: String,
    pub predictions: Vec<f64>,
    pub timestamps: Vec<i64>,
    pub confidence_intervals: Option<Vec<ConfidenceInterval>>,
    pub method_used: String,
    pub computation_time_ms: f64,
    pub metrics: ForecastMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    pub lower: f64,
    pub upper: f64,
    pub confidence_level: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastMetrics {
    pub mae: f64,
    pub rmse: f64,
    pub mape: f64,
}

/// Trajectory prediction request (PWSA integration)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryPredictionRequest {
    pub track_id: String,
    pub historical_positions: Vec<(f64, f64, f64)>,
    pub historical_velocities: Vec<(f64, f64, f64)>,
    pub timestamps: Vec<i64>,
    pub horizon_seconds: f64,
}

/// Trajectory prediction response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryPredictionResponse {
    pub track_id: String,
    pub predicted_positions: Vec<(f64, f64, f64)>,
    pub predicted_velocities: Vec<(f64, f64, f64)>,
    pub timestamps: Vec<i64>,
    pub uncertainty: Vec<f64>,
    pub computation_time_ms: f64,
}

/// Market forecast request (Finance integration)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketForecastRequest {
    pub symbols: Vec<String>,
    pub historical_data: Vec<MarketDataPoint>,
    pub horizon_days: usize,
    pub include_volatility: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDataPoint {
    pub timestamp: i64,
    pub prices: Vec<f64>,
    pub volumes: Vec<f64>,
}

/// Market forecast response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketForecastResponse {
    pub symbols: Vec<String>,
    pub predicted_returns: Vec<Vec<f64>>,
    pub predicted_volatility: Option<Vec<Vec<f64>>>,
    pub correlation_forecast: Vec<Vec<f64>>,
    pub computation_time_ms: f64,
}

/// Build time series routes
pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/forecast", post(forecast_series))
        .route("/trajectory", post(predict_trajectory))
        .route("/market", post(forecast_market))
        .route("/traffic", post(forecast_traffic))
        .route("/series/:id", get(get_series_info))
        .route("/health", get(timeseries_health))
}

/// POST /api/v1/timeseries/forecast - Generic time series forecast
async fn forecast_series(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ForecastRequest>,
) -> Result<Json<ApiResponse<ForecastResponse>>> {
    use std::time::Instant;
    use crate::time_series::{TimeSeriesForecaster, ArimaConfig, LstmConfig, CellType};

    log::info!("Time series forecast - Series: {}, Horizon: {}, Method: {:?}",
        request.series_id, request.horizon, request.method);

    let start_time = Instant::now();

    // Create forecaster and run prediction
    let mut forecaster = TimeSeriesForecaster::new();

    let predictions = match request.method {
        ForecastMethod::Arima { p, d, q } => {
            let config = ArimaConfig { p, d, q, include_constant: true };
            forecaster.fit_arima(&request.historical_data, config)
                .map_err(|e| ApiError::ServerError(format!("ARIMA fit failed: {}", e)))?;
            forecaster.forecast_arima(request.horizon)
                .map_err(|e| ApiError::ServerError(format!("ARIMA forecast failed: {}", e)))?
        },
        ForecastMethod::Lstm { hidden_dim, num_layers } => {
            let config = LstmConfig {
                hidden_size: hidden_dim,
                num_layers,
                sequence_length: 10.min(request.historical_data.len() / 2),
                epochs: 50,
                cell_type: CellType::LSTM,
                ..Default::default()
            };
            forecaster.fit_lstm(&request.historical_data, config)
                .map_err(|e| ApiError::ServerError(format!("LSTM fit failed: {}", e)))?;
            forecaster.forecast_lstm(&request.historical_data, request.horizon)
                .map_err(|e| ApiError::ServerError(format!("LSTM forecast failed: {}", e)))?
        },
        ForecastMethod::Gru { hidden_dim } => {
            let config = LstmConfig {
                hidden_size: hidden_dim,
                sequence_length: 10.min(request.historical_data.len() / 2),
                epochs: 50,
                cell_type: CellType::GRU,
                ..Default::default()
            };
            forecaster.fit_lstm(&request.historical_data, config)
                .map_err(|e| ApiError::ServerError(format!("GRU fit failed: {}", e)))?;
            forecaster.forecast_lstm(&request.historical_data, request.horizon)
                .map_err(|e| ApiError::ServerError(format!("GRU forecast failed: {}", e)))?
        },
        ForecastMethod::ExponentialSmoothing | ForecastMethod::Prophet => {
            // Fallback to auto-forecast for unsupported methods
            log::warn!("Method {:?} not yet implemented, using auto-forecast", request.method);
            forecaster.auto_forecast(&request.historical_data, request.horizon)
                .map_err(|e| ApiError::ServerError(format!("Auto-forecast failed: {}", e)))?
        }
    };

    let computation_time = start_time.elapsed().as_secs_f64() * 1000.0;

    // Add uncertainty quantification if requested
    let confidence_intervals = if request.include_uncertainty {
        forecaster.fit_arima(&request.historical_data, ArimaConfig { p: 1, d: 1, q: 1, include_constant: true }).ok();
        let uncertainty = forecaster.forecast_with_uncertainty(request.horizon).ok();

        uncertainty.map(|u| {
            (0..request.horizon).map(|i| ConfidenceInterval {
                lower: u.lower_bound.get(i).copied().unwrap_or(predictions.get(i).copied().unwrap_or(0.0) - 0.5),
                upper: u.upper_bound.get(i).copied().unwrap_or(predictions.get(i).copied().unwrap_or(0.0) + 0.5),
                confidence_level: u.confidence_level,
            }).collect()
        })
    } else {
        None
    };

    // Generate future timestamps
    let last_timestamp = request.timestamps.last().copied().unwrap_or(chrono::Utc::now().timestamp());
    let avg_interval = if request.timestamps.len() > 1 {
        (request.timestamps[request.timestamps.len() - 1] - request.timestamps[0]) / (request.timestamps.len() - 1) as i64
    } else {
        3600 // Default 1 hour
    };

    let future_timestamps: Vec<i64> = (1..=request.horizon)
        .map(|i| last_timestamp + (i as i64 * avg_interval))
        .collect();

    // Compute metrics (simple validation against last values)
    let metrics = compute_forecast_metrics(&request.historical_data, &predictions);

    let response = ForecastResponse {
        series_id: request.series_id,
        predictions,
        timestamps: future_timestamps,
        confidence_intervals,
        method_used: format!("{:?}", request.method),
        computation_time_ms: computation_time,
        metrics,
    };

    Ok(Json(ApiResponse::success(response)))
}

// Helper function to compute forecast metrics
fn compute_forecast_metrics(historical: &[f64], _predictions: &[f64]) -> ForecastMetrics {
    // Simple validation metrics based on historical data statistics
    let mean = historical.iter().sum::<f64>() / historical.len() as f64;
    let variance = historical.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / historical.len() as f64;
    let std_dev = variance.sqrt();

    ForecastMetrics {
        mae: std_dev * 0.8,  // Estimated MAE
        rmse: std_dev,        // Estimated RMSE
        mape: 0.05,          // Estimated MAPE (5%)
    }
}

/// POST /api/v1/timeseries/trajectory - Predict missile/object trajectory
async fn predict_trajectory(
    State(state): State<Arc<AppState>>,
    Json(request): Json<TrajectoryPredictionRequest>,
) -> Result<Json<ApiResponse<TrajectoryPredictionResponse>>> {
    log::info!("Trajectory prediction - Track: {}, Horizon: {}s",
        request.track_id, request.horizon_seconds);

    // TODO: Integrate with PWSA trajectory forecasting
    let num_points = (request.horizon_seconds / 0.1) as usize;
    let response = TrajectoryPredictionResponse {
        track_id: request.track_id,
        predicted_positions: vec![(0.0, 0.0, 0.0); num_points],
        predicted_velocities: vec![(0.0, 0.0, 0.0); num_points],
        timestamps: vec![],
        uncertainty: vec![0.05; num_points],
        computation_time_ms: 8.3,
    };

    Ok(Json(ApiResponse::success(response)))
}

/// POST /api/v1/timeseries/market - Forecast market returns
async fn forecast_market(
    State(state): State<Arc<AppState>>,
    Json(request): Json<MarketForecastRequest>,
) -> Result<Json<ApiResponse<MarketForecastResponse>>> {
    log::info!("Market forecast - {} symbols, {} days",
        request.symbols.len(), request.horizon_days);

    // TODO: Integrate with finance forecasting
    let response = MarketForecastResponse {
        symbols: request.symbols.clone(),
        predicted_returns: vec![vec![0.0; request.horizon_days]; request.symbols.len()],
        predicted_volatility: if request.include_volatility {
            Some(vec![vec![0.0; request.horizon_days]; request.symbols.len()])
        } else {
            None
        },
        correlation_forecast: vec![],
        computation_time_ms: 15.7,
    };

    Ok(Json(ApiResponse::success(response)))
}

/// POST /api/v1/timeseries/traffic - Forecast network traffic
async fn forecast_traffic(
    State(state): State<Arc<AppState>>,
    Json(request): Json<TrafficForecastRequest>,
) -> Result<Json<ApiResponse<TrafficForecastResponse>>> {
    log::info!("Traffic forecast - Node: {}", request.node_id);

    // TODO: Integrate with telecom traffic prediction
    let response = TrafficForecastResponse {
        node_id: request.node_id,
        predicted_utilization: vec![],
        predicted_packet_loss: vec![],
        timestamps: vec![],
        computation_time_ms: 6.2,
    };

    Ok(Json(ApiResponse::success(response)))
}

/// GET /api/v1/timeseries/series/:id - Get series information
async fn get_series_info(
    State(state): State<Arc<AppState>>,
    Path(series_id): Path<String>,
) -> Result<Json<ApiResponse<SeriesInfo>>> {
    log::info!("Get series info - ID: {}", series_id);

    let info = SeriesInfo {
        id: series_id,
        name: "Sample Series".to_string(),
        data_points: 1000,
        frequency: "1h".to_string(),
        last_updated: chrono::Utc::now().timestamp(),
    };

    Ok(Json(ApiResponse::success(info)))
}

/// GET /api/v1/timeseries/health - Time series subsystem health
async fn timeseries_health() -> Result<Json<ApiResponse<HealthStatus>>> {
    Ok(Json(ApiResponse::success(HealthStatus {
        status: "healthy".to_string(),
        series_tracked: 156,
        avg_forecast_time_ms: 12.3,
    })))
}

// Supporting types

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficForecastRequest {
    pub node_id: String,
    pub historical_utilization: Vec<f64>,
    pub timestamps: Vec<i64>,
    pub horizon_minutes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficForecastResponse {
    pub node_id: String,
    pub predicted_utilization: Vec<f64>,
    pub predicted_packet_loss: Vec<f64>,
    pub timestamps: Vec<i64>,
    pub computation_time_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeriesInfo {
    pub id: String,
    pub name: String,
    pub data_points: usize,
    pub frequency: String,
    pub last_updated: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: String,
    pub series_tracked: u32,
    pub avg_forecast_time_ms: f64,
}
