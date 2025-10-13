//! Application Domains
//!
//! Domain-specific implementations for:
//! - Robotics: Motion planning and trajectory forecasting (Worker 7)
//! - Scientific Discovery: Experiment design and optimization (Worker 7)
//! - Drug discovery: Molecular optimization and ADMET prediction (Workers 3 & 7)
//! - Telecom: Network optimization and traffic engineering (Worker 3)
//! - Healthcare: Patient risk prediction and clinical decision support (Worker 3)
//! - Supply chain: Inventory optimization and logistics routing (Worker 3)
//! - Energy grid: Power grid optimization and renewable integration (Worker 3)
//! - Manufacturing: Production scheduling and predictive maintenance (Worker 3)
//! - Cybersecurity: Network intrusion detection and threat response (Worker 3)
//! - Agriculture: Precision agriculture and crop optimization (Worker 3)

// Worker 7 Application Domains
pub mod robotics;
pub mod scientific;
pub mod information_metrics;

// Worker 3 Application Domains
pub mod drug_discovery;
pub mod telecom;
pub mod healthcare;
pub mod supply_chain;
pub mod energy_grid;
pub mod manufacturing;
pub mod cybersecurity;
pub mod agriculture;

// Worker 7 Exports
pub use robotics::{
    RoboticsController, RoboticsConfig, MotionPlanner, MotionPlan,
    AdvancedTrajectoryForecaster, TrajectoryForecastConfig,
};
pub use scientific::{ScientificDiscovery, ScientificConfig};
pub use information_metrics::{
    ExperimentInformationMetrics,
    MolecularInformationMetrics,
    RoboticsInformationMetrics,
};

// Worker 3 & 7 Drug Discovery (both workers contributed)
pub use drug_discovery::{DrugDiscoveryController, DrugDiscoveryConfig};

/// Applications module version
pub const VERSION: &str = "0.2.0";
