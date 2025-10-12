//! Robotics motion planning and control API endpoints

use axum::{
    extract::{Path, State},
    routing::{get, post},
    Router, Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::api_server::{ApiError, Result, AppState, models::ApiResponse};

/// Robot state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobotState {
    pub position: (f64, f64, f64),
    pub orientation: (f64, f64, f64, f64), // Quaternion
    pub joint_angles: Vec<f64>,
    pub velocity: (f64, f64, f64),
}

/// Motion planning request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionPlanRequest {
    pub robot_id: String,
    pub start_state: RobotState,
    pub goal_state: RobotState,
    pub obstacles: Vec<Obstacle>,
    pub constraints: MotionConstraints,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Obstacle {
    pub id: String,
    pub position: (f64, f64, f64),
    pub size: (f64, f64, f64),
    pub obstacle_type: ObstacleType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObstacleType {
    Static,
    Dynamic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionConstraints {
    pub max_velocity: f64,
    pub max_acceleration: f64,
    pub max_jerk: f64,
    pub collision_margin: f64,
}

/// Motion plan result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionPlan {
    pub trajectory: Vec<TrajectoryPoint>,
    pub total_time: f64,
    pub total_distance: f64,
    pub is_collision_free: bool,
    pub planning_time_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryPoint {
    pub time: f64,
    pub state: RobotState,
}

/// Build robotics routes
pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/plan", post(plan_motion))
        .route("/execute", post(execute_trajectory))
        .route("/robot/:id", get(get_robot_status))
        .route("/robots", get(list_robots))
        .route("/health", get(robotics_health))
}

/// POST /api/v1/robotics/plan - Plan robot motion
async fn plan_motion(
    State(state): State<Arc<AppState>>,
    Json(request): Json<MotionPlanRequest>,
) -> Result<Json<ApiResponse<MotionPlan>>> {
    log::info!("Motion planning - Robot: {}", request.robot_id);

    // TODO: Integrate with actual motion planner
    let plan = MotionPlan {
        trajectory: vec![
            TrajectoryPoint {
                time: 0.0,
                state: request.start_state.clone(),
            },
            TrajectoryPoint {
                time: 1.0,
                state: request.goal_state.clone(),
            },
        ],
        total_time: 5.2,
        total_distance: 10.5,
        is_collision_free: true,
        planning_time_ms: 15.3,
    };

    Ok(Json(ApiResponse::success(plan)))
}

/// POST /api/v1/robotics/execute - Execute trajectory
async fn execute_trajectory(
    State(state): State<Arc<AppState>>,
    Json(plan): Json<MotionPlan>,
) -> Result<Json<ApiResponse<ExecutionStatus>>> {
    log::info!("Execute trajectory - {} points", plan.trajectory.len());

    // TODO: Send to robot controller
    let status = ExecutionStatus {
        status: "executing".to_string(),
        progress: 0.0,
        estimated_completion_time: plan.total_time,
    };

    Ok(Json(ApiResponse::success(status)))
}

/// GET /api/v1/robotics/robot/:id - Get robot status
async fn get_robot_status(
    State(state): State<Arc<AppState>>,
    Path(robot_id): Path<String>,
) -> Result<Json<ApiResponse<RobotStatus>>> {
    log::info!("Get robot status - ID: {}", robot_id);

    let status = RobotStatus {
        id: robot_id,
        online: true,
        current_state: RobotState {
            position: (0.0, 0.0, 0.0),
            orientation: (0.0, 0.0, 0.0, 1.0),
            joint_angles: vec![],
            velocity: (0.0, 0.0, 0.0),
        },
        battery_level: 0.85,
        task_status: "idle".to_string(),
    };

    Ok(Json(ApiResponse::success(status)))
}

/// GET /api/v1/robotics/robots - List all robots
async fn list_robots(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ApiResponse<Vec<RobotSummary>>>> {
    log::info!("List all robots");

    let robots = vec![
        RobotSummary {
            id: "robot-001".to_string(),
            name: "Manipulator 1".to_string(),
            online: true,
            battery_level: 0.85,
        },
    ];

    Ok(Json(ApiResponse::success(robots)))
}

/// GET /api/v1/robotics/health - Robotics subsystem health
async fn robotics_health() -> Result<Json<ApiResponse<HealthStatus>>> {
    Ok(Json(ApiResponse::success(HealthStatus {
        status: "healthy".to_string(),
        robots_online: 5,
        avg_planning_time_ms: 15.2,
    })))
}

// Supporting types

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStatus {
    pub status: String,
    pub progress: f64,
    pub estimated_completion_time: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobotStatus {
    pub id: String,
    pub online: bool,
    pub current_state: RobotState,
    pub battery_level: f64,
    pub task_status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobotSummary {
    pub id: String,
    pub name: String,
    pub online: bool,
    pub battery_level: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: String,
    pub robots_online: u32,
    pub avg_planning_time_ms: f64,
}
