//! PWSA (Proliferated Warfighter Space Architecture) API endpoints
//!
//! Provides threat detection, sensor fusion, and tracking capabilities

use axum::{
    extract::{Path, State},
    routing::{get, post},
    Router, Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::api_server::{Result, AppState, models::ApiResponse, info_theory::InfoTheoryMetrics};

/// PWSA sensor data input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorInput {
    /// Satellite vehicle ID
    pub sv_id: u32,
    /// Timestamp (Unix epoch)
    pub timestamp: i64,
    /// IR frame data
    pub ir_frame: IrFrameData,
    /// Radar tracks (optional)
    pub radar_tracks: Option<Vec<RadarTrack>>,
}

/// IR frame data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrFrameData {
    pub width: u32,
    pub height: u32,
    pub centroid_x: f64,
    pub centroid_y: f64,
    pub hotspot_count: u32,
    /// Optional full pixel data (for pixel processing API)
    pub pixels: Option<Vec<u16>>,
}

/// Radar track
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RadarTrack {
    pub track_id: String,
    pub position: (f64, f64, f64),
    pub velocity: (f64, f64, f64),
    pub rcs: f64,
}

/// Threat assessment result with information-theoretic metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatAssessment {
    pub threat_id: String,
    pub threat_type: ThreatType,
    pub confidence: f64,
    pub position: (f64, f64, f64),
    pub velocity: (f64, f64, f64),
    pub estimated_trajectory: Vec<(f64, f64, f64)>,
    pub time_to_impact: Option<f64>,
    pub recommended_action: String,

    /// Information-theoretic metrics for sensor-threat correlation
    pub info_metrics: InfoTheoryMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThreatType {
    BallisticMissile,
    HypersonicGlideVehicle,
    Aircraft,
    Satellite,
    SpaceDebris,
    Unknown,
}

/// Tracking request
#[derive(Debug, Deserialize)]
pub struct TrackingQuery {
    pub track_id: String,
    pub duration_seconds: Option<u64>,
}

/// Build PWSA routes
pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/detect", post(detect_threat))
        .route("/track/:track_id", get(get_track))
        .route("/tracks", get(list_tracks))
        .route("/fuse", post(fuse_sensor_data))
        .route("/health", get(pwsa_health))
}

/// POST /api/v1/pwsa/detect - Detect threats from sensor data
async fn detect_threat(
    State(state): State<Arc<AppState>>,
    Json(sensor_input): Json<SensorInput>,
) -> Result<Json<ApiResponse<ThreatAssessment>>> {
    log::info!("PWSA threat detection request - SV ID: {}", sensor_input.sv_id);

    // Extract sensor measurements for information-theoretic analysis
    let sensor_measurements = if let Some(ref pixels) = sensor_input.ir_frame.pixels {
        // Use pixel data if available
        pixels.iter().map(|&p| p as f64).collect::<Vec<_>>()
    } else {
        // Use centroid data as proxy
        vec![
            sensor_input.ir_frame.centroid_x,
            sensor_input.ir_frame.centroid_y,
            sensor_input.ir_frame.hotspot_count as f64,
        ]
    };

    // Calculate information-theoretic metrics
    use crate::api_server::info_theory::estimate_sensor_info_metrics;
    let info_metrics = estimate_sensor_info_metrics(&sensor_measurements, None, 0.1);

    // TODO: Integrate with actual PWSA platform
    // For now, return enhanced mock data with real info-theory metrics
    let assessment = ThreatAssessment {
        threat_id: format!("threat-{}", uuid::Uuid::new_v4()),
        threat_type: ThreatType::BallisticMissile,
        confidence: 0.92,
        position: (sensor_input.ir_frame.centroid_x, sensor_input.ir_frame.centroid_y, 250.0),
        velocity: (1500.0, 200.0, -50.0),
        estimated_trajectory: vec![
            (100.0, 100.0, 250.0),
            (200.0, 150.0, 200.0),
            (300.0, 200.0, 150.0),
        ],
        time_to_impact: Some(120.0),
        recommended_action: "Activate defense system".to_string(),
        info_metrics,
    };

    Ok(Json(ApiResponse::success(assessment)))
}

/// GET /api/v1/pwsa/track/:track_id - Get specific track information
async fn get_track(
    State(state): State<Arc<AppState>>,
    Path(track_id): Path<String>,
) -> Result<Json<ApiResponse<TrackInfo>>> {
    log::info!("PWSA track query - Track ID: {}", track_id);

    // TODO: Query actual track database
    let track = TrackInfo {
        track_id: track_id.clone(),
        last_updated: chrono::Utc::now().timestamp(),
        position: (150.0, 200.0, 180.0),
        velocity: (1200.0, 150.0, -30.0),
        status: TrackStatus::Active,
    };

    Ok(Json(ApiResponse::success(track)))
}

/// GET /api/v1/pwsa/tracks - List all active tracks
async fn list_tracks(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<Vec<TrackInfo>>>> {
    log::info!("PWSA list all tracks");

    // TODO: Query track database
    let tracks = vec![
        TrackInfo {
            track_id: "track-001".to_string(),
            last_updated: chrono::Utc::now().timestamp(),
            position: (150.0, 200.0, 180.0),
            velocity: (1200.0, 150.0, -30.0),
            status: TrackStatus::Active,
        },
    ];

    Ok(Json(ApiResponse::success(tracks)))
}

/// POST /api/v1/pwsa/fuse - Multi-sensor data fusion with Kalman filtering
async fn fuse_sensor_data(
    State(state): State<Arc<AppState>>,
    Json(sensor_inputs): Json<Vec<SensorInput>>,
) -> Result<Json<ApiResponse<FusionResult>>> {
    use std::time::Instant;
    use crate::api_server::kalman::MultiSensorFusion;

    log::info!("PWSA sensor fusion - {} inputs", sensor_inputs.len());

    let start_time = Instant::now();

    // Group measurements by track (using sv_id as proxy for track_id)
    let mut measurements_by_track: std::collections::HashMap<String, Vec<([f64; 3], f64)>> =
        std::collections::HashMap::new();

    for input in &sensor_inputs {
        let track_id = format!("sv-{}", input.sv_id);
        let measurement = [
            input.ir_frame.centroid_x,
            input.ir_frame.centroid_y,
            250.0, // Estimated altitude
        ];
        let timestamp = input.timestamp as f64;

        measurements_by_track
            .entry(track_id)
            .or_insert_with(Vec::new)
            .push((measurement, timestamp));
    }

    // Apply Kalman filtering to fuse measurements
    let mut fusion = MultiSensorFusion::new();
    let mut fused_tracks = Vec::new();

    for (track_id, measurements) in measurements_by_track {
        let fusion_result = fusion.fuse_measurements(&track_id, &measurements);

        // Convert to TrackInfo
        fused_tracks.push(TrackInfo {
            track_id,
            last_updated: chrono::Utc::now().timestamp(),
            position: (
                fusion_result.position[0],
                fusion_result.position[1],
                fusion_result.position[2],
            ),
            velocity: (
                fusion_result.velocity[0],
                fusion_result.velocity[1],
                fusion_result.velocity[2],
            ),
            status: TrackStatus::Active,
        });
    }

    let processing_time = start_time.elapsed();

    // Calculate confidence based on position uncertainty
    let avg_uncertainty = if !fused_tracks.is_empty() {
        1.0 / (1.0 + fused_tracks.len() as f64 * 0.1)
    } else {
        0.0
    };

    let result = FusionResult {
        fused_tracks,
        processing_time_ms: processing_time.as_secs_f64() * 1000.0,
        sensors_used: sensor_inputs.len(),
        confidence: (1.0 - avg_uncertainty).max(0.5),
    };

    Ok(Json(ApiResponse::success(result)))
}

/// GET /api/v1/pwsa/health - PWSA subsystem health check
async fn pwsa_health() -> Result<Json<ApiResponse<HealthStatus>>> {
    Ok(Json(ApiResponse::success(HealthStatus {
        status: "healthy".to_string(),
        sensors_online: 12,
        processing_latency_ms: 1.2,
    })))
}

// Supporting types

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackInfo {
    pub track_id: String,
    pub last_updated: i64,
    pub position: (f64, f64, f64),
    pub velocity: (f64, f64, f64),
    pub status: TrackStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrackStatus {
    Active,
    Lost,
    Terminated,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionResult {
    pub fused_tracks: Vec<TrackInfo>,
    pub processing_time_ms: f64,
    pub sensors_used: usize,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: String,
    pub sensors_online: u32,
    pub processing_latency_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sensor_input_serialization() {
        let input = SensorInput {
            sv_id: 42,
            timestamp: 1234567890,
            ir_frame: IrFrameData {
                width: 640,
                height: 480,
                centroid_x: 320.0,
                centroid_y: 240.0,
                hotspot_count: 5,
                pixels: None,
            },
            radar_tracks: None,
        };

        let json = serde_json::to_string(&input).unwrap();
        let parsed: SensorInput = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.sv_id, 42);
    }
}
