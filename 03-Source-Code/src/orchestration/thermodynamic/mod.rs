//! Thermodynamic module

pub mod hamiltonian;
pub mod quantum_consensus;
pub mod gpu_thermodynamic_consensus;
pub mod thermodynamic_consensus;
pub mod optimized_thermodynamic_consensus;

// Worker 5 - Advanced Thermodynamic Schedules
pub mod advanced_simulated_annealing;
pub mod advanced_parallel_tempering;
pub mod advanced_hmc;
pub mod advanced_bayesian_optimization;
pub mod advanced_multi_objective;
pub mod advanced_replica_exchange;
pub mod gpu_schedule_kernels;
pub mod adaptive_temperature_control;

pub use hamiltonian::InformationHamiltonian;
pub use quantum_consensus::{QuantumConsensusOptimizer, ConsensusState};
pub use gpu_thermodynamic_consensus::{GpuThermodynamicConsensus, LLMModel, ThermodynamicState as GpuThermodynamicState};
pub use thermodynamic_consensus::ThermodynamicConsensus;

// Worker 5: Enhanced consensus with adaptive schedule selection
pub use optimized_thermodynamic_consensus::{
    OptimizedThermodynamicConsensus,
    TemperatureSchedule,
    SchedulePerformance,
};

// Worker 5 exports
pub use advanced_simulated_annealing::{
    SimulatedAnnealingSchedule,
    CoolingType,
    GpuAnnealingScheduler,
};

pub use advanced_parallel_tempering::{
    ParallelTemperingSchedule,
    ReplicaState,
    ExchangeSchedule,
    SwapStatistics,
    GpuParallelTempering,
};

pub use advanced_hmc::{
    HMCSchedule,
    Trajectory,
    GpuHMCScheduler,
};

pub use advanced_bayesian_optimization::{
    BayesianOptimizationSchedule,
    GaussianProcess,
    KernelFunction,
    AcquisitionFunction,
};

pub use advanced_multi_objective::{
    MultiObjectiveSchedule,
    Solution,
    ParetoFrontier,
    Scalarization,
};

pub use advanced_replica_exchange::{
    ReplicaExchange,
    ThermodynamicReplicaState,
    ExchangeProposal,
    IterationStats,
    ExchangeStatistics,
};

pub use gpu_schedule_kernels::{
    GpuScheduleKernels,
    BoltzmannKernel,
    ReplicaSwapKernel,
    LeapfrogKernel,
    GaussianProcessKernel,
    ParetoDominanceKernel,
    BatchTemperatureKernel,
    GPKernelType,
    CoolingStrategy,
};

pub use adaptive_temperature_control::{
    AdaptiveTemperatureController,
    AdaptiveCoolingSchedule,
    AcceptanceMonitor,
    PIDController,
    AdaptiveControllerStats,
    OPTIMAL_ACCEPTANCE_RATE,
};
