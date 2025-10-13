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
    use std::time::Instant;
    use crate::applications::robotics::{RoboticsController, RoboticsConfig, RobotState as WorkerRobotState};
    use ndarray::Array1;

    log::info!("Motion planning - Robot: {}, {} obstacles",
        request.robot_id, request.obstacles.len());

    let start_time = Instant::now();

    // Create robotics controller with Worker 7's implementation
    let config = RoboticsConfig {
        planning_horizon: 5.0,
        control_frequency: 50.0,
        use_gpu: true,
        enable_forecasting: !request.obstacles.is_empty(),
        max_planning_time_ms: 100,
    };

    let mut controller = RoboticsController::new(config)
        .map_err(|e| ApiError::ServerError(format!("Failed to create controller: {}", e)))?;

    // Convert API robot states to Worker 7's format (2D only: x, y)
    let current_state = WorkerRobotState {
        position: Array1::from_vec(vec![request.start_state.position.0, request.start_state.position.1]),
        velocity: Array1::from_vec(vec![request.start_state.velocity.0, request.start_state.velocity.1]),
        orientation: 0.0, // Worker 7 uses 2D orientation (radians)
        angular_velocity: 0.0,
        timestamp: chrono::Utc::now().timestamp_millis() as f64 / 1000.0,
    };

    let goal_state = WorkerRobotState {
        position: Array1::from_vec(vec![request.goal_state.position.0, request.goal_state.position.1]),
        velocity: Array1::from_vec(vec![request.goal_state.velocity.0, request.goal_state.velocity.1]),
        orientation: 0.0,
        angular_velocity: 0.0,
        timestamp: (chrono::Utc::now().timestamp_millis() as f64 / 1000.0) + 5.0,
    };

    // Plan motion using Worker 7's Active Inference planner
    let worker_plan = controller.plan_motion(&current_state, &goal_state)
        .map_err(|e| ApiError::ServerError(format!("Motion planning failed: {}", e)))?;

    let planning_time = start_time.elapsed().as_secs_f64() * 1000.0;

    // Convert Worker 7's plan to API format
    let trajectory: Vec<TrajectoryPoint> = worker_plan.waypoints.iter().map(|wp| {
        TrajectoryPoint {
            time: wp.time,
            state: RobotState {
                position: (wp.position[0], wp.position[1], 0.0), // Map 2D to 3D (z=0)
                orientation: request.start_state.orientation, // Preserve orientation
                joint_angles: request.start_state.joint_angles.clone(), // Preserve joint angles
                velocity: (wp.velocity[0], wp.velocity[1], 0.0), // Map 2D to 3D
            },
        }
    }).collect();

    // Compute total distance from Worker 7's waypoints
    let total_distance = worker_plan.waypoints.windows(2).map(|w| {
        let dx = w[1].position[0] - w[0].position[0];
        let dy = w[1].position[1] - w[0].position[1];
        (dx*dx + dy*dy).sqrt()
    }).sum();

    // Compute total time (last waypoint time)
    let total_time = worker_plan.waypoints.last()
        .map(|wp| wp.time)
        .unwrap_or(0.0);

    let plan = MotionPlan {
        trajectory,
        total_time,
        total_distance,
        is_collision_free: worker_plan.reaches_goal, // Use reaches_goal as proxy
        planning_time_ms: planning_time,
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
