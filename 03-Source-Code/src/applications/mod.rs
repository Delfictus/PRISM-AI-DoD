//! Applications Module - Worker 7
//!
//! Domain-specific applications built on PRISM-AI foundation:
//! - Robotics: Motion planning and control
//! - Scientific Discovery: Experiment design and optimization
//!
//! Constitution: Worker 7 Development
//! Time: 268 hours (228h base + 40h time series enhancements)
//! Focus: Motion planning, Environment prediction, Scientific tools

pub mod robotics;
pub mod scientific;

pub use robotics::{RoboticsController, RoboticsConfig, MotionPlanner, MotionPlan};
pub use scientific::{ScientificDiscovery, ScientificConfig};

/// Applications module version
pub const VERSION: &str = "0.1.0";

/// Worker 7 identifier
pub const WORKER_ID: u8 = 7;

/// Worker 7 responsibilities
pub const RESPONSIBILITIES: &[&str] = &[
    "Motion planning with Active Inference",
    "Environment dynamics prediction",
    "Trajectory forecasting",
    "ROS integration",
    "Scientific discovery tools",
    "Experimental design optimization",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_identity() {
        assert_eq!(WORKER_ID, 7);
        assert_eq!(VERSION, "0.1.0");
    }

    #[test]
    fn test_responsibilities_defined() {
        assert!(RESPONSIBILITIES.len() > 0);
        assert!(RESPONSIBILITIES.contains(&"Motion planning with Active Inference"));
    }
}
