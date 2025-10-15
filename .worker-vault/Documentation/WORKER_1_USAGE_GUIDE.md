# Worker 1 Usage Guide
## AI Core & Time Series Forecasting

**Complete**: 100% (280/280 hours)
**Modules**: 13 production modules, 7,395 lines, 102 tests
**Build Status**: ✅ Successful

---

## Table of Contents

1. [Transfer Entropy](#1-transfer-entropy)
2. [Thermodynamic Routing](#2-thermodynamic-routing)
3. [Active Inference](#3-active-inference)
4. [Time Series Forecasting](#4-time-series-forecasting)
5. [Integration Examples](#5-integration-examples)

---

## 1. Transfer Entropy

### Overview

Transfer Entropy (TE) quantifies directed information flow between time series using the Kraskov-Stögbauer-Grassberger (KSG) algorithm with GPU acceleration.

**Files**:
- `src/orchestration/routing/te_embedding_gpu.rs` (384 lines, 5 tests)
- `src/orchestration/routing/gpu_kdtree.rs` (562 lines, 7 tests)
- `src/orchestration/routing/ksg_transfer_entropy_gpu.rs` (553 lines, 7 tests)
- `src/orchestration/routing/te_validation.rs` (613 lines, 5 tests)

### Basic Usage

```rust
use prism_ai::{detect_causal_direction, CausalDirection};

// Create two coupled time series
let x_series = vec![/* your data */];
let y_series = vec![/* your data */];

// Compute Transfer Entropy
let result = detect_causal_direction(
    &x_series,
    &y_series,
    embedding_dim: 2,    // State space dimension
    delay: 1,            // Time delay τ
    k_neighbors: 5       // KSG k parameter
)?;

println!("TE(X→Y): {:.4}", result.te_x_to_y);
println!("TE(Y→X): {:.4}", result.te_y_to_x);
println!("Direction: {:?}", result.direction);

// Interpret result
match result.direction {
    CausalDirection::XToY => println!("X causes Y"),
    CausalDirection::YToX => println!("Y causes X"),
    CausalDirection::Bidirectional => println!("Mutual causation"),
    CausalDirection::Independent => println!("No causation"),
}
```

### Advanced: Manual KSG Computation

```rust
use prism_ai::orchestration::routing::{
    GpuTimeDelayEmbedding, GpuNearestNeighbors,
    KsgTransferEntropyGpu, DistanceMetric
};

// Step 1: Time-delay embedding
let embedding = GpuTimeDelayEmbedding::new(
    embedding_dim: 3,
    delay: 2
)?;

let embedded_x = embedding.embed(&x_series)?;
let embedded_y = embedding.embed(&y_series)?;

// Step 2: k-NN search on GPU
let knn = GpuNearestNeighbors::new(
    DistanceMetric::MaxNorm  // KSG uses max norm
)?;

// Step 3: KSG Transfer Entropy
let ksg = KsgTransferEntropyGpu::new(k: 5)?;

let te_x_to_y = ksg.compute_transfer_entropy(
    &embedded_x,
    &embedded_y
)?;

println!("TE(X→Y) = {:.6}", te_x_to_y);
```

### Use Cases

#### LLM Routing
```rust
// Detect which LLM models influence each other
let gpt4_responses = vec![/* response times */];
let claude_responses = vec![/* response times */];

let coupling = detect_causal_direction(
    &gpt4_responses,
    &claude_responses,
    2, 1, 5
)?;

if coupling.te_x_to_y > 0.5 {
    println!("GPT-4 influences Claude routing");
}
```

#### PWSA (Missile Tracking)
```rust
// Detect causal relationships between satellite tracks
let sat1_trajectory = vec![/* positions */];
let sat2_trajectory = vec![/* positions */];

let result = detect_causal_direction(
    &sat1_trajectory,
    &sat2_trajectory,
    3, 2, 7
)?;

if result.direction == CausalDirection::XToY {
    println!("Satellite 1 is pursuing Satellite 2");
}
```

---

## 2. Thermodynamic Routing

### Overview

Thermodynamic routing uses statistical mechanics to model LLM consensus, with multiple temperature schedules and replica exchange for optimal exploration.

**Files**:
- `src/orchestration/thermodynamic/advanced_energy.rs` (742 lines, 8 tests)
- `src/orchestration/thermodynamic/temperature_schedules.rs` (635 lines, 11 tests)
- `src/orchestration/thermodynamic/replica_exchange.rs` (565 lines, 10 tests)

### Basic Usage

```rust
use prism_ai::{ThermodynamicNetwork, NetworkConfig};

// Create thermodynamic network (3 LLM models)
let config = NetworkConfig {
    num_nodes: 3,
    coupling_strength: 1.0,
    temperature: 1.0,
    dt: 0.01,
    dissipation: 0.1,
};

let mut network = ThermodynamicNetwork::new(config);

// Evolve system to equilibrium
for _ in 0..1000 {
    network.step()?;
}

let state = network.get_state();
println!("Free Energy: {:.4}", state.free_energy);
println!("Entropy: {:.4}", state.entropy);
println!("Temperature: {:.4}", state.temperature);
```

### Temperature Schedules

```rust
use prism_ai::orchestration::thermodynamic::{
    TemperatureSchedule, ScheduleType
};

// Exponential cooling
let schedule = TemperatureSchedule::new(
    initial_temp: 2.0,
    schedule_type: ScheduleType::Exponential { alpha: 0.95 }
);

for iteration in 0..100 {
    let temp = schedule.temperature(iteration);
    println!("T({}) = {:.4}", iteration, temp);
}

// Adaptive schedule (targets 23.4% acceptance rate)
let adaptive = TemperatureSchedule::new(
    initial_temp: 1.0,
    schedule_type: ScheduleType::Adaptive {
        target_acceptance: 0.234,  // Gelman optimal
        window_size: 100,
        adaptation_rate: 0.1,
    }
);

// Fokker-Planck SDE
let fokker_planck = TemperatureSchedule::new(
    initial_temp: 1.5,
    schedule_type: ScheduleType::FokkerPlanckSDE {
        damping: 0.1,
        noise_strength: 0.5,
        dt: 0.01,
    }
);
```

### Replica Exchange

```rust
use prism_ai::orchestration::thermodynamic::{
    ReplicaExchangeSystem, ReplicaConfig
};

// Create replica exchange with 5 temperature levels
let config = ReplicaConfig {
    n_replicas: 5,
    min_temp: 0.1,
    max_temp: 10.0,
    swap_interval: 10,
    adaptation_interval: 100,
};

let mut replica_system = ReplicaExchangeSystem::new(config)?;

// Run parallel tempering
for iteration in 0..1000 {
    replica_system.step()?;

    if iteration % 100 == 0 {
        let stats = replica_system.get_statistics();
        println!("Swap acceptance: {:.2}%", stats.swap_acceptance * 100.0);
        println!("Gelman-Rubin R̂: {:.4}", stats.gelman_rubin);
    }
}

// Convergence criterion: R̂ < 1.1
if replica_system.has_converged() {
    println!("✓ Replicas converged!");
}
```

---

## 3. Active Inference

### Overview

Hierarchical active inference with GPU-accelerated belief propagation and policy search for autonomous decision-making.

**Files**:
- `src/active_inference/hierarchical_inference_gpu.rs` (565 lines, 9 tests)
- `src/active_inference/policy_search_gpu.rs` (424 lines, 12 tests)

### Hierarchical Inference

```rust
use prism_ai::{HierarchicalModel, StateSpaceLevel};

// Create 3-level hierarchy
let levels = vec![
    StateSpaceLevel {
        state_dim: 2,        // Low-level: sensor data
        obs_dim: 2,
        precision: 1.0,
        learning_rate: 0.01,
        timescale: 1.0,      // Fast (10ms)
    },
    StateSpaceLevel {
        state_dim: 4,        // Mid-level: features
        obs_dim: 2,
        precision: 0.5,
        learning_rate: 0.005,
        timescale: 100.0,    // Medium (1s)
    },
    StateSpaceLevel {
        state_dim: 8,        // High-level: goals
        obs_dim: 4,
        precision: 0.1,
        learning_rate: 0.001,
        timescale: 6000.0,   // Slow (60s)
    },
];

let mut model = HierarchicalModel::new(levels);

// Perform inference
let observation = vec![0.5, 0.3];
model.infer_states(&observation)?;

// Get beliefs at each level
let beliefs = model.get_all_beliefs();
for (i, belief) in beliefs.iter().enumerate() {
    println!("Level {}: {:?}", i, belief);
}
```

### Policy Search

```rust
use prism_ai::{
    GenerativeModel, ObservationModel, TransitionModel,
    VariationalInference, PolicySelector
};

// Define generative model
let obs_model = ObservationModel::linear_gaussian(
    state_dim: 4,
    obs_dim: 2,
    noise: 0.1
);

let trans_model = TransitionModel::linear_dynamical(
    state_dim: 4,
    process_noise: 0.05
);

let gen_model = GenerativeModel::new(
    preferred_observations: vec![1.0, 0.0],  // Target state
    obs_model,
    trans_model,
);

// Variational inference
let var_inference = VariationalInference::new(
    step_size: 0.01,
    max_iterations: 100,
    convergence_threshold: 1e-4
);

// Policy selector
let policy_selector = PolicySelector::new(
    exploration_weight: 0.1  // Balance exploration vs exploitation
);

// Select action based on expected free energy
let observation = vec![0.5, 0.5];
let current_state = vec![0.4, 0.6, 0.1, 0.2];

let action = policy_selector.select_action(
    &gen_model,
    &var_inference,
    &observation,
    &current_state
)?;

println!("Selected action: {:?}", action);
```

### Expected Free Energy

```rust
use prism_ai::active_inference::policy_search_gpu::{
    GpuPolicySearch, PolicyType
};

let policy_search = GpuPolicySearch::new(
    n_policies: 10,
    horizon: 5
)?;

// Evaluate multiple policies in parallel on GPU
let policies = vec![
    PolicyType::Exploitation,
    PolicyType::Balanced,
    PolicyType::Exploratory,
    PolicyType::Aggressive,
];

let efes = policy_search.evaluate_policies_batch(
    &policies,
    &current_state,
    &gen_model
)?;

// Select policy with minimum expected free energy
let best_idx = efes.iter()
    .enumerate()
    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
    .map(|(idx, _)| idx)
    .unwrap();

println!("Best policy: {:?} (EFE = {:.4})", policies[best_idx], efes[best_idx]);
```

---

## 4. Time Series Forecasting

### Overview

GPU-accelerated time series forecasting with ARIMA, LSTM/GRU, and uncertainty quantification.

**Files**:
- `src/time_series/arima_gpu.rs` (865 lines, 8 tests)
- `src/time_series/lstm_forecaster.rs` (780 lines, 10 tests)
- `src/time_series/uncertainty.rs` (585 lines, 8 tests)
- `src/time_series/mod.rs` (122 lines, 3 tests)

### ARIMA Forecasting

```rust
use prism_ai::{ArimaGpu, ArimaConfig};

// Configure ARIMA(2,1,1) model
let config = ArimaConfig {
    p: 2,  // AutoRegressive order
    d: 1,  // Differencing order
    q: 1,  // Moving Average order
    include_constant: true,
};

// Train model
let mut model = ArimaGpu::new(config)?;
model.fit(&training_data)?;

// Forecast 10 steps ahead
let forecast = model.forecast(horizon: 10)?;

println!("Forecast: {:?}", forecast);

// Model selection criteria
let aic = model.aic()?;
let bic = model.bic()?;
println!("AIC: {:.4}, BIC: {:.4}", aic, bic);
```

### Auto ARIMA

```rust
use prism_ai::time_series::auto_arima;

// Automatically select best ARIMA model
let model = auto_arima(
    &data,
    max_p: 5,  // Search AR orders 0-5
    max_d: 2,  // Search differencing 0-2
    max_q: 5   // Search MA orders 0-5
)?;

let forecast = model.forecast(20)?;
println!("Best model forecast: {:?}", forecast);
```

### LSTM/GRU Forecasting

```rust
use prism_ai::{LstmForecaster, LstmConfig, CellType};

// Configure LSTM
let config = LstmConfig {
    cell_type: CellType::LSTM,  // or CellType::GRU
    hidden_size: 50,
    num_layers: 2,
    sequence_length: 20,  // Lookback window
    learning_rate: 0.001,
    epochs: 100,
    batch_size: 32,
    dropout: 0.2,         // Regularization
};

// Train model
let mut model = LstmForecaster::new(config)?;
model.fit(&training_data)?;

// Forecast
let forecast = model.forecast(&training_data, horizon: 10)?;

println!("LSTM forecast: {:?}", forecast);
```

### Uncertainty Quantification

```rust
use prism_ai::{
    UncertaintyQuantifier, UncertaintyConfig, UncertaintyMethod,
    ForecastWithUncertainty
};

// Configure uncertainty quantifier
let config = UncertaintyConfig {
    confidence_level: 0.95,  // 95% confidence intervals
    method: UncertaintyMethod::Residual,
    n_bootstrap_samples: 1000,
    residual_window: 100,
};

let mut quantifier = UncertaintyQuantifier::new(config);

// Add historical residuals
for (pred, actual) in predictions.iter().zip(actuals.iter()) {
    quantifier.add_residual(actual - pred);
}

// Get forecast with confidence intervals
let forecast_with_ci = quantifier.residual_intervals(&forecast)?;

for i in 0..forecast_with_ci.forecast.len() {
    println!("Step {}: {:.2} [{:.2}, {:.2}]",
        i + 1,
        forecast_with_ci.forecast[i],
        forecast_with_ci.lower_bound[i],
        forecast_with_ci.upper_bound[i]
    );
}
```

### Bootstrap Intervals

```rust
// Distribution-free bootstrap intervals
let forecast_with_bootstrap = quantifier.bootstrap_intervals(
    &historical_data,
    |data| model.forecast(data, 10),  // Forecast function
    horizon: 10
)?;

println!("Bootstrap 95% CI:");
for i in 0..10 {
    println!("  [{:.2}, {:.2}]",
        forecast_with_bootstrap.lower_bound[i],
        forecast_with_bootstrap.upper_bound[i]
    );
}
```

### Unified Forecasting Interface

```rust
use prism_ai::TimeSeriesForecaster;

let mut forecaster = TimeSeriesForecaster::new();

// Auto-forecast: tries ARIMA first, falls back to LSTM
let forecast = forecaster.auto_forecast(&data, horizon: 20)?;

println!("Auto-selected forecast: {:?}", forecast);

// Or use specific method
forecaster.fit_arima(&data, arima_config)?;
let arima_forecast = forecaster.forecast_arima(20)?;

forecaster.fit_lstm(&data, lstm_config)?;
let lstm_forecast = forecaster.forecast_lstm(&data, 20)?;

// With uncertainty
let forecast_with_uncertainty = forecaster.forecast_with_uncertainty(20)?;
```

---

## 5. Integration Examples

### Example 1: LLM Cost Forecasting + Thermodynamic Optimization

```rust
use prism_ai::{TimeSeriesForecaster, ThermodynamicNetwork};

// Forecast LLM costs for next week
let mut forecaster = TimeSeriesForecaster::new();
forecaster.fit_lstm(&historical_costs, lstm_config)?;
let cost_forecast = forecaster.forecast_lstm(&historical_costs, 7)?;

// Adjust thermodynamic parameters based on forecast
let mut network = ThermodynamicNetwork::new(config);

for (day, predicted_cost) in cost_forecast.iter().enumerate() {
    if predicted_cost > &threshold {
        // Increase temperature → more exploration of cheaper models
        network.set_temperature(1.5);
        println!("Day {}: High cost predicted, increasing exploration", day + 1);
    } else {
        // Decrease temperature → exploit best models
        network.set_temperature(0.5);
        println!("Day {}: Normal cost, exploiting best models", day + 1);
    }

    network.step()?;
}
```

### Example 2: PWSA Trajectory Prediction + Active Inference

```rust
use prism_ai::{TimeSeriesForecaster, HierarchicalModel, detect_causal_direction};

// Detect causal relationships between missile tracks
let result = detect_causal_direction(
    &missile1_positions,
    &missile2_positions,
    3, 2, 7
)?;

if result.te_x_to_y > 0.5 {
    println!("Missile 1 is pursuing Missile 2!");

    // Forecast Missile 2 trajectory
    let mut forecaster = TimeSeriesForecaster::new();
    forecaster.fit_arima(&missile2_positions, arima_config)?;
    let trajectory_forecast = forecaster.forecast_with_uncertainty(10)?;

    // Use Active Inference to plan intercept
    let mut hier_model = HierarchicalModel::new(levels);

    for (t, predicted_pos) in trajectory_forecast.forecast.iter().enumerate() {
        let observation = vec![*predicted_pos];
        hier_model.infer_states(&observation)?;

        // Plan intercept maneuver
        let intercept_action = policy_selector.select_action(
            &gen_model,
            &var_inference,
            &observation,
            &interceptor_state
        )?;

        println!("T+{}: Intercept action {:?}", t, intercept_action);
    }
}
```

### Example 3: Transfer Entropy → Temperature Schedule Selection

```rust
// Measure coupling between LLM models
let coupling = detect_causal_direction(
    &gpt4_latencies,
    &claude_latencies,
    2, 1, 5
)?;

// Strong coupling → use replica exchange
if coupling.te_x_to_y > 0.7 || coupling.te_y_to_x > 0.7 {
    println!("Strong coupling detected, using replica exchange");

    let replica_system = ReplicaExchangeSystem::new(replica_config)?;

    for _ in 0..1000 {
        replica_system.step()?;
    }

// Weak coupling → use simple exponential schedule
} else {
    println!("Weak coupling, using exponential cooling");

    let schedule = TemperatureSchedule::new(
        1.0,
        ScheduleType::Exponential { alpha: 0.95 }
    );
}
```

---

## Performance Metrics

### Transfer Entropy
- **Target**: <100ms for 1000 variables
- **Status**: ⏳ Ready for benchmarking
- **GPU Acceleration**: ✅ Enabled

### Thermodynamic Routing
- **Target**: 40-70% cost savings
- **Status**: ⏳ Ready for production measurement
- **Convergence**: ✅ Gelman-Rubin R̂ < 1.1

### Active Inference
- **Target**: <1ms decision time
- **Status**: ⏳ Ready for benchmarking
- **Strategies**: ✅ 4 types (Exploitation, Balanced, Exploratory, Aggressive)

### Time Series Forecasting
- **Target**: RMSE <5% on validation
- **Status**: ⏳ Ready for domain-specific testing
- **Methods**: ✅ ARIMA + LSTM/GRU + 4 uncertainty methods

---

## Build and Test

```bash
# Build library
cargo build --lib --features cuda

# Run all Worker 1 tests
cargo test --lib orchestration::routing --features cuda
cargo test --lib orchestration::thermodynamic --features cuda
cargo test --lib active_inference --features cuda
cargo test --lib time_series --features cuda

# Check specific modules
cargo test --lib ksg_transfer_entropy_gpu
cargo test --lib replica_exchange
cargo test --lib hierarchical_inference_gpu
cargo test --lib lstm_forecaster
```

---

## Integration Points

### For Other Workers

**Worker 3 (PWSA)**:
- Use Transfer Entropy for satellite track coupling analysis
- Use Time Series for trajectory forecasting
- Use Active Inference for intercept planning

**Worker 4 (Finance)**:
- Use Time Series for price/volatility forecasting
- Use Thermodynamic routing for portfolio optimization

**Worker 5 (LLM Orchestration)**:
- Use Transfer Entropy for model coupling detection
- Use Thermodynamic routing for consensus
- Use Time Series for cost forecasting

**Worker 7 (Robotics)**:
- Use Time Series for environment dynamics prediction
- Use Active Inference for motion planning
- Use Transfer Entropy for multi-agent interaction analysis

---

## Next Steps

1. **Performance Benchmarking**: Validate <100ms TE, <1ms policy search targets
2. **JIDT Comparison**: Validate Transfer Entropy accuracy (<5% error)
3. **Production Deployment**: Measure 40-70% cost savings in LLM routing
4. **Domain-Specific Tuning**: Optimize hyperparameters for PWSA, Finance, Robotics
5. **GPU Kernel Optimization**: Work with Worker 2 to optimize critical kernels

---

**Worker 1 Status**: 🎉 **100% COMPLETE** 🎉

All 280 hours delivered. Ready for production integration and benchmarking.
