# Worker 7 → Worker 2: GPU Kernel Request

**Requesting Worker**: Worker 7 (Robotics & Scientific Discovery)
**Target Worker**: Worker 2 (GPU Infrastructure)
**Priority**: High (Week 1-2 delivery needed for Week 5 integration)
**Label**: `[KERNEL]`

---

## Executive Summary

Worker 7 requires GPU kernels for robotics motion planning and trajectory forecasting. These kernels will enable real-time obstacle prediction, multi-agent coordination, and collision-free path planning at 50Hz control rates.

**Total Kernels Requested**: 6 (3 time series + 3 robotics-specific)

---

## Kernel Requests

### 1. Time Series Kernels (Shared with Worker 1)

#### Kernel 1: `ar_forecast`
**Purpose**: Autoregressive forecasting for short-term obstacle prediction

**Signature**:
```cuda
__global__ void ar_forecast(
    const float* historical_data,    // Input: historical positions [batch_size, history_len, n_features]
    const float* ar_coefficients,    // Input: AR coefficients [order, n_features]
    float* forecast,                 // Output: predicted values [batch_size, horizon, n_features]
    int batch_size,
    int history_len,
    int horizon,
    int n_features,
    int ar_order
);
```

**Use Case** (Worker 7):
```rust
// Predict obstacle position 1 second ahead
let historical_positions = Array2::from_shape_vec((10, 2), obstacle_history)?;
let forecast = gpu_executor.ar_forecast(
    &historical_positions,
    &ar_coeffs,
    horizon_steps=10,  // 1 second at 10Hz
)?;
```

**Performance Requirements**:
- Latency: < 1ms for batch_size=100
- Throughput: > 10,000 predictions/second
- Precision: FP32 (FP16 if tensor cores available)

---

#### Kernel 2: `lstm_cell`
**Purpose**: LSTM cell for complex trajectory pattern learning

**Signature**:
```cuda
__global__ void lstm_cell(
    const float* input,              // Input: [batch_size, input_dim]
    const float* hidden_state,       // Input/Output: [batch_size, hidden_dim]
    const float* cell_state,         // Input/Output: [batch_size, hidden_dim]
    const float* weights_ih,         // Input: input-hidden weights [4*hidden_dim, input_dim]
    const float* weights_hh,         // Input: hidden-hidden weights [4*hidden_dim, hidden_dim]
    const float* bias,               // Input: biases [4*hidden_dim]
    float* output,                   // Output: [batch_size, hidden_dim]
    int batch_size,
    int input_dim,
    int hidden_dim
);
```

**Use Case** (Worker 7):
```rust
// Predict multi-agent trajectories with LSTM
for timestep in 0..horizon {
    gpu_executor.lstm_cell(
        &input_t,
        &mut hidden,
        &mut cell,
        &weights_ih,
        &weights_hh,
        &bias,
        &mut output_t,
    )?;
}
```

**Performance Requirements**:
- Latency: < 2ms for batch_size=50, hidden_dim=128
- Support: hidden_dim up to 512
- Memory: Efficient use of shared memory for weights

---

#### Kernel 3: `uncertainty_propagation`
**Purpose**: Compute prediction uncertainty for safety margins

**Signature**:
```cuda
__global__ void uncertainty_propagation(
    const float* mean_forecast,      // Input: predicted means [batch_size, horizon, n_features]
    const float* covariance,          // Input: covariance matrix [n_features, n_features]
    const float* dynamics_noise,      // Input: process noise [n_features]
    float* uncertainty_bounds,        // Output: ±σ bounds [batch_size, horizon, n_features]
    int batch_size,
    int horizon,
    int n_features
);
```

**Use Case** (Worker 7):
```rust
// Compute safety margins around predicted obstacles
let uncertainty = gpu_executor.uncertainty_propagation(
    &forecast_means,
    &covariance_matrix,
    &process_noise,
)?;

// Add safety margin = 3*sigma
let safe_radius = obstacle_radius + 3.0 * uncertainty;
```

**Performance Requirements**:
- Latency: < 0.5ms
- Precision: FP32 (covariance must be positive definite)

---

### 2. Robotics-Specific Kernels

#### Kernel 4: `collision_detection_batch`
**Purpose**: Parallel collision detection for motion planning

**Signature**:
```cuda
__global__ void collision_detection_batch(
    const float* robot_positions,    // Input: candidate positions [n_candidates, 2]
    const float* obstacle_positions, // Input: obstacle positions [n_obstacles, 2]
    const float* obstacle_radii,     // Input: obstacle radii [n_obstacles]
    float robot_radius,
    bool* collision_flags,           // Output: collision detected [n_candidates]
    float* min_distances,            // Output: minimum distances [n_candidates]
    int n_candidates,
    int n_obstacles
);
```

**Use Case** (Worker 7):
```rust
// Check 1000 candidate waypoints for collisions
let collision_free = gpu_executor.collision_detection_batch(
    &candidate_positions,
    &obstacle_positions,
    &obstacle_radii,
    robot_radius,
)?;

// Filter to collision-free candidates
let valid_candidates: Vec<_> = candidates
    .iter()
    .zip(collision_free)
    .filter(|(_, is_free)| *is_free)
    .map(|(pos, _)| pos)
    .collect();
```

**Performance Requirements**:
- Latency: < 1ms for n_candidates=1000, n_obstacles=100
- Throughput: > 1M collision checks/second

---

#### Kernel 5: `trajectory_cost_evaluation`
**Purpose**: Evaluate cost of trajectory candidates in parallel

**Signature**:
```cuda
__global__ void trajectory_cost_evaluation(
    const float* trajectories,       // Input: candidate trajectories [n_traj, n_steps, 2]
    const float* goal_position,      // Input: goal [2]
    const float* obstacle_positions, // Input: obstacles [n_obstacles, 2]
    const float* obstacle_radii,     // Input: radii [n_obstacles]
    float* costs,                    // Output: trajectory costs [n_traj]
    int n_trajectories,
    int n_steps,
    int n_obstacles,
    float goal_weight,
    float collision_weight,
    float smoothness_weight
);
```

**Cost Function**:
```
cost = goal_weight * distance_to_goal
     + collision_weight * collision_penalty
     + smoothness_weight * trajectory_curvature
```

**Use Case** (Worker 7):
```rust
// Evaluate 100 candidate trajectories
let costs = gpu_executor.trajectory_cost_evaluation(
    &candidate_trajectories,
    &goal_position,
    &obstacles,
    &obstacle_radii,
    goal_weight=1.0,
    collision_weight=1000.0,  // High penalty
    smoothness_weight=0.1,
)?;

// Select minimum cost trajectory
let best_idx = costs.argmin();
let best_trajectory = &candidate_trajectories[best_idx];
```

**Performance Requirements**:
- Latency: < 2ms for n_traj=100, n_steps=50, n_obstacles=100
- Precision: FP32

---

#### Kernel 6: `interaction_matrix_gpu`
**Purpose**: Compute agent-agent interaction matrix for multi-agent forecasting

**Signature**:
```cuda
__global__ void interaction_matrix_gpu(
    const float* agent_positions,    // Input: [n_agents, 2]
    const float* agent_velocities,   // Input: [n_agents, 2]
    float* interaction_matrix,       // Output: [n_agents, n_agents]
    int n_agents,
    float interaction_radius,
    float velocity_alignment_weight
);
```

**Interaction Model**:
```
interaction[i,j] = f(distance, velocity_alignment)
where:
  distance_factor = exp(-distance / interaction_radius)
  velocity_factor = dot(v_i, v_j) / (||v_i|| * ||v_j||)
  interaction = distance_factor * (1 + velocity_alignment_weight * velocity_factor)
```

**Use Case** (Worker 7):
```rust
// Compute how agents influence each other
let interactions = gpu_executor.interaction_matrix_gpu(
    &agent_positions,
    &agent_velocities,
    interaction_radius=5.0,
    velocity_alignment_weight=0.3,
)?;

// Use in LSTM forecasting to predict coordinated motion
let predicted_trajectories = lstm_forecaster.forecast_with_interactions(
    &agents,
    &interactions,
    horizon=3.0,
)?;
```

**Performance Requirements**:
- Latency: < 0.5ms for n_agents=50
- Complexity: O(n_agents²) but parallelized

---

## Integration Requirements

### Memory Management:
- All kernels should work with CudaSlice from cudarc
- Support for pinned memory for faster host↔device transfers
- Reuse buffers where possible to minimize allocations

### Error Handling:
- Check for CUDA errors after kernel launch
- Validate input dimensions
- Return meaningful error messages

### Interface Example:
```rust
// Expected interface in Worker 2's kernel_executor.rs
impl GpuKernelExecutor {
    pub fn ar_forecast(
        &self,
        historical_data: &CudaSlice<f32>,
        ar_coefficients: &CudaSlice<f32>,
        horizon: usize,
    ) -> Result<CudaSlice<f32>> {
        // Launch kernel
        // Return forecast
    }
}
```

---

## Testing Requirements

### Unit Tests (Worker 2 provides):
- [ ] Correctness: Compare GPU results vs CPU reference
- [ ] Performance: Measure latency and throughput
- [ ] Edge cases: Empty inputs, single element, large batches

### Integration Tests (Worker 7 will provide):
- [ ] Real robotics scenario with moving obstacles
- [ ] Multi-agent coordination test
- [ ] Collision detection accuracy

---

## Timeline

**Week 1 (Days 1-3)**: Implement time series kernels (ar_forecast, lstm_cell, uncertainty_propagation)

**Week 1 (Days 4-5)**: Implement robotics kernels (collision_detection_batch, trajectory_cost_evaluation)

**Week 2 (Days 1-2)**: Implement interaction_matrix_gpu

**Week 2 (Days 3-5)**: Testing, optimization, documentation

**Delivery**: End of Week 2 (so Worker 7 can integrate in Week 5)

---

## Performance Targets Summary

| Kernel | Latency Target | Throughput Target |
|--------|---------------|-------------------|
| ar_forecast | < 1ms | > 10k predictions/s |
| lstm_cell | < 2ms | > 5k forward passes/s |
| uncertainty_propagation | < 0.5ms | > 20k propagations/s |
| collision_detection_batch | < 1ms | > 1M checks/s |
| trajectory_cost_evaluation | < 2ms | > 50k trajectories/s |
| interaction_matrix_gpu | < 0.5ms | > 100k pairs/s |

**Overall Target**: 50Hz control loop (20ms total budget)
- Prediction: < 5ms
- Planning: < 10ms
- Execution: < 5ms

---

## Priority Ranking

1. **HIGH**: `collision_detection_batch` (needed for basic motion planning)
2. **HIGH**: `ar_forecast` (short-term prediction)
3. **MEDIUM**: `trajectory_cost_evaluation` (optimization)
4. **MEDIUM**: `lstm_cell` (long-term prediction)
5. **LOW**: `uncertainty_propagation` (safety enhancement)
6. **LOW**: `interaction_matrix_gpu` (multi-agent, advanced feature)

---

## Contact & Questions

**Worker 7 Lead**: Robotics & Scientific Discovery Module
**Questions**: Create GitHub issue with `[KERNEL]` tag
**Status Updates**: Weekly sync on Friday EOD

---

## Acceptance Criteria

- [ ] All 6 kernels implemented and tested
- [ ] Performance targets met
- [ ] Integration tests pass
- [ ] Documentation complete (docstrings + examples)
- [ ] Code review approved
- [ ] Merged to `parallel-development` branch

---

**Thank you, Worker 2!** 🚀

These kernels will enable real-time robotic motion planning with predictive capabilities, a critical component for the PRISM-AI robotics system.
