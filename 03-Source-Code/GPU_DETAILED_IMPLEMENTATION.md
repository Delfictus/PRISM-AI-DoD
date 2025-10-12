# Detailed GPU Migration Implementation Plans

## Task 1: Replace CPU computation in gpu_enabled.rs with kernel_executor

### Current Problem:
- Lines 108-130: Uses CPU computation while printing "GPU-ENABLED mode"
- Lines 136-145: ReLU uses CPU loop
- Lines 149-182: Softmax uses CPU implementation

### Implementation Steps:
1. **Add kernel_executor dependency**
   - Import `GpuKernelExecutor` and `get_global_executor`
   - Store executor reference in `GpuContext`

2. **Replace matmul (lines 108-130)**
   ```rust
   // OLD: CPU computation
   for i in 0..m {
       for j in 0..n {
           for l in 0..k {
               sum += a_data[i * k + l] * b_data[l * n + j];
           }
       }
   }

   // NEW: GPU kernel execution
   let executor = get_global_executor()?;
   let result = executor.lock().unwrap().matrix_multiply(
       a_data, b_data, m, k, n
   )?;
   ```

3. **Replace ReLU (lines 136-145)**
   ```rust
   // OLD: CPU loop
   for x in &mut self.data {
       *x = x.max(0.0);
   }

   // NEW: GPU kernel
   let executor = get_global_executor()?;
   executor.lock().unwrap().relu_inplace(&mut self.data)?;
   ```

4. **Replace Softmax (lines 149-182)**
   ```rust
   // OLD: CPU nested loops
   // NEW: GPU kernel
   let executor = get_global_executor()?;
   executor.lock().unwrap().softmax(
       &mut self.data, batch_size, num_classes
   )?;
   ```

5. **Add proper GPU memory management**
   - Keep data on GPU between operations
   - Minimize host-device transfers
   - Use pinned memory for transfers

### Success Criteria:
- NO CPU loops in computation
- All operations use kernel_executor
- Performance >10x faster than CPU

---

## Task 2: Migrate PWSA Active Inference Classifier to GPU

### Current Problem:
- `src/pwsa/active_inference_classifier.rs`: All computations on CPU
- Variational free energy computed serially
- Belief updates not parallelized

### Implementation Steps:

1. **Create GPU kernels for core operations**
   ```cuda
   __global__ void variational_free_energy_kernel(
       float* beliefs, float* observations,
       float* precision, float* free_energy,
       int batch_size, int state_dim
   );

   __global__ void belief_update_kernel(
       float* prior, float* likelihood,
       float* posterior, float* prediction_error,
       int batch_size, int state_dim
   );

   __global__ void kl_divergence_kernel(
       float* p, float* q, float* kl_div,
       int batch_size, int dim
   );
   ```

2. **Port GpuActiveInferenceClassifier**
   - Store all beliefs/states on GPU
   - Implement parallel policy evaluation
   - Batch process multiple hypotheses

3. **GPU memory layout optimization**
   - Structure-of-Arrays for beliefs
   - Coalesced memory access patterns
   - Shared memory for reductions

4. **Integration points**
   - Replace CPU `compute_free_energy()`
   - Replace CPU `update_beliefs()`
   - Replace CPU `select_action()`

### Success Criteria:
- 100x faster inference
- Support batch size >1000
- <1ms latency for updates

---

## Task 3: Port Neuromorphic modules to GPU kernels

### Current Problem:
- `src/neuromorphic/src/gpu_reservoir.rs`: Placeholder implementation
- Spike generation on CPU
- STDP learning not parallelized

### Implementation Steps:

1. **Spike generation kernel**
   ```cuda
   __global__ void generate_spikes_kernel(
       float* membrane_potential, bool* spikes,
       float threshold, int num_neurons
   );
   ```

2. **Reservoir evolution kernel**
   ```cuda
   __global__ void reservoir_update_kernel(
       float* weights, float* states, bool* spikes,
       float* next_states, int reservoir_size,
       float leak_rate, float spectral_radius
   );
   ```

3. **STDP learning kernel**
   ```cuda
   __global__ void stdp_update_kernel(
       float* weights, bool* pre_spikes, bool* post_spikes,
       float* spike_times, float learning_rate,
       int num_synapses
   );
   ```

4. **Pattern recognition kernel**
   - Parallel template matching
   - GPU-accelerated correlation

### Success Criteria:
- Support 10,000+ neuron reservoirs
- Real-time spike processing
- <0.1ms per timestep

---

## Task 4: Implement GPU Statistical Mechanics

### Current Problem:
- `src/statistical_mechanics/gpu_bindings.rs`: Stub implementation
- Kuramoto model on CPU
- Entropy calculations serial

### Implementation Steps:

1. **Kuramoto oscillator kernel**
   ```cuda
   __global__ void kuramoto_evolution_kernel(
       float* phases, float* frequencies,
       float* coupling_matrix, float* new_phases,
       int n_oscillators, float dt, float coupling_strength
   );
   ```

2. **Entropy production kernel**
   ```cuda
   __global__ void entropy_production_kernel(
       float* states, float* velocities,
       float* entropy_rate, int n_particles,
       float temperature
   );
   ```

3. **Phase synchronization analysis**
   - Order parameter calculation on GPU
   - Parallel phase coherence computation

4. **Optimization**
   - Use texture memory for coupling matrix
   - Shared memory for local interactions

### Success Criteria:
- Simulate 100,000+ oscillators
- Real-time entropy tracking
- 60+ FPS visualization

---

## Task 5: Convert Transfer Entropy calculations to GPU

### Current Problem:
- `src/cma/transfer_entropy_gpu.rs`: CPU implementation
- Mutual information computed serially
- Embedding not parallelized

### Implementation Steps:

1. **Transfer entropy kernel**
   ```cuda
   __global__ void transfer_entropy_kernel(
       float* source, float* target,
       float* te_matrix, int time_series_length,
       int embedding_dim, int tau
   );
   ```

2. **Mutual information kernel**
   ```cuda
   __global__ void mutual_information_kernel(
       float* x, float* y, float* mi,
       int n_samples, int n_bins
   );
   ```

3. **Time-delay embedding kernel**
   - Parallel embedding generation
   - GPU-based correlation computation

4. **Optimization strategies**
   - Block-wise computation for large matrices
   - Warp-level primitives for reductions

### Success Criteria:
- Process 1M+ sample time series
- <10ms for full TE matrix
- Support streaming data

---

## Task 6: Port Quantum simulation to GPU kernels

### Current Problem:
- `src/quantum_mlir/runtime.rs`: CPU state vector operations
- Gate applications serial
- Measurement simulation slow

### Implementation Steps:

1. **Quantum gate kernels**
   ```cuda
   __global__ void apply_hadamard_kernel(
       cuComplex* state_vector, int qubit_idx,
       int state_dim
   );

   __global__ void apply_cnot_kernel(
       cuComplex* state_vector, int control, int target,
       int state_dim
   );
   ```

2. **State evolution kernel**
   ```cuda
   __global__ void quantum_evolution_kernel(
       cuComplex* state, cuComplex* hamiltonian,
       cuComplex* next_state, float dt, int dim
   );
   ```

3. **Measurement kernel**
   - Parallel probability computation
   - GPU-based sampling

4. **Optimization**
   - cuBLAS for matrix operations
   - Tensor cores for complex arithmetic

### Success Criteria:
- Simulate 30+ qubits
- <1ms per gate operation
- Exact numerical precision

---

## Task 7: Migrate Active Inference to GPU

### Current Problem:
- `src/active_inference/gpu_inference.rs`: Partial implementation
- Policy evaluation on CPU
- Expected free energy serial

### Implementation Steps:

1. **Expected free energy kernel**
   ```cuda
   __global__ void expected_free_energy_kernel(
       float* beliefs, float* policies,
       float* efe_values, float* preferences,
       int n_policies, int horizon, int state_dim
   );
   ```

2. **Belief propagation kernel**
   ```cuda
   __global__ void belief_propagation_kernel(
       float* messages, float* beliefs,
       float* observations, float* transition_model,
       int n_states, int n_timesteps
   );
   ```

3. **Policy selection kernel**
   - Parallel policy evaluation
   - GPU-based softmax selection

4. **Precision-weighted prediction**
   - Parallel error computation
   - GPU precision updates

### Success Criteria:
- Evaluate 1000+ policies in parallel
- <5ms decision time
- Support hierarchical models

---

## Task 8: Implement GPU Thermodynamic Consensus

### Current Problem:
- CPU-based entropy optimization
- Serial temperature annealing
- Consensus formation slow

### Implementation Steps:

1. **Entropy optimization kernel**
   ```cuda
   __global__ void entropy_optimization_kernel(
       float* states, float* gradients,
       float* entropy, float temperature,
       int n_agents, int state_dim
   );
   ```

2. **Simulated annealing kernel**
   ```cuda
   __global__ void parallel_annealing_kernel(
       float* solutions, float* energies,
       float* temperatures, curandState* rand_states,
       int n_replicas, int dim
   );
   ```

3. **Consensus formation kernel**
   - Parallel voting aggregation
   - GPU-based weight updates

4. **Boltzmann sampling**
   - Parallel random sampling
   - GPU temperature scheduling

### Success Criteria:
- 1000+ agent consensus
- <100ms convergence
- Provable convergence guarantees

---

## Task 9: Port Quantum Voting to GPU

### Current Problem:
- Quantum state manipulation on CPU
- Interference patterns computed serially
- Measurement collapse slow

### Implementation Steps:

1. **Superposition manipulation kernel**
   ```cuda
   __global__ void superposition_kernel(
       cuComplex* amplitudes, float* phases,
       int* votes, int n_voters, int n_choices
   );
   ```

2. **Interference pattern kernel**
   ```cuda
   __global__ void quantum_interference_kernel(
       cuComplex* state1, cuComplex* state2,
       float* interference_pattern, int dim
   );
   ```

3. **Measurement collapse kernel**
   - Parallel probability computation
   - GPU-based outcome selection

### Success Criteria:
- Handle 10,000+ voters
- <10ms voting round
- Quantum advantage demonstrated

---

## Task 10: Convert Transfer Entropy Router to GPU

### Current Problem:
- Causal flow computation on CPU
- Routing decisions serial
- Information flow not optimized

### Implementation Steps:

1. **Causal flow kernel**
   ```cuda
   __global__ void causal_flow_kernel(
       float* time_series, float* causal_matrix,
       int n_nodes, int time_window, int lag
   );
   ```

2. **Dynamic routing kernel**
   ```cuda
   __global__ void routing_decision_kernel(
       float* causal_strengths, int* routes,
       float* costs, int n_paths, int n_nodes
   );
   ```

3. **Information optimization**
   - Parallel path evaluation
   - GPU-based flow updates

### Success Criteria:
- Route 1000+ information streams
- <1ms routing decisions
- Adaptive to network changes

---

## Task 11: Implement GPU PID Synergy Decomposition

### Current Problem:
- Information decomposition on CPU
- Unique/redundant/synergistic computed serially

### Implementation Steps:

1. **Unique information kernel**
   ```cuda
   __global__ void unique_information_kernel(
       float* joint_dist, float* unique_info,
       int n_vars, int n_samples
   );
   ```

2. **Redundant information kernel**
   ```cuda
   __global__ void redundant_information_kernel(
       float* marginals, float* redundant_info,
       int n_sources, int n_samples
   );
   ```

3. **Synergistic information kernel**
   - Parallel synergy computation
   - GPU-based optimization

### Success Criteria:
- Process 100+ variable systems
- <50ms full decomposition
- Numerical stability maintained

---

## Task 12: Port CMA algorithms to GPU

### Current Problem:
- TSP solver on CPU
- Graph algorithms serial
- Evolution strategies slow

### Implementation Steps:

1. **TSP solver kernel**
   ```cuda
   __global__ void tsp_2opt_kernel(
       float* distances, int* tours,
       float* costs, int n_cities, int n_tours
   );
   ```

2. **Graph coloring kernel**
   ```cuda
   __global__ void graph_coloring_kernel(
       int* adjacency, int* colors,
       int n_vertices, int max_colors
   );
   ```

3. **Evolution strategy kernel**
   - Parallel population evaluation
   - GPU-based selection/mutation

### Success Criteria:
- Solve 1000+ city TSP
- <100ms for graph problems
- 100x speedup vs CPU

---

## Task 13: Remove ALL CPU fallback paths

### Implementation Steps:

1. **Search and destroy**
   ```bash
   # Find all CPU fallbacks
   grep -r "cfg(not(feature = \"cuda\"))" src/
   grep -r "CPU fallback" src/
   grep -r "placeholder" src/
   ```

2. **Make GPU mandatory**
   - Remove all conditional compilation
   - Fail compilation without CUDA
   - No optional GPU features

3. **Verification**
   - Compile with `--no-default-features`
   - Should fail without CUDA
   - No CPU code paths remain

### Success Criteria:
- ZERO CPU fallback code
- Compilation fails without GPU
- All tests require GPU

---

## Task 14: Implement local LLM inference on GPU

### Current Problem:
- Only API calls to external services
- No local model execution
- Latency and cost issues

### Implementation Steps:

1. **Model loading**
   - Load GGUF/ONNX models to GPU
   - Weight quantization (INT8/INT4)
   - Memory mapping optimization

2. **Attention kernel**
   ```cuda
   __global__ void multi_head_attention_kernel(
       float* Q, float* K, float* V,
       float* output, float* mask,
       int seq_len, int d_model, int n_heads
   );
   ```

3. **Token generation**
   - Parallel beam search
   - GPU-based sampling

4. **Integration options**
   - llama.cpp with CUDA backend
   - ONNX Runtime CUDA provider
   - Custom transformer implementation

### Success Criteria:
- 100+ tokens/second
- Support 7B+ parameter models
- <100ms first token latency

---

## Task 15: Create GPU kernel library for novel algorithms

### Implementation Steps:

1. **Information geometry kernels**
   ```cuda
   __global__ void fisher_information_kernel(...);
   __global__ void geodesic_kernel(...);
   __global__ void natural_gradient_kernel(...);
   ```

2. **Variational inference kernels**
   ```cuda
   __global__ void elbo_kernel(...);
   __global__ void reparameterization_kernel(...);
   __global__ void importance_sampling_kernel(...);
   ```

3. **Causal discovery kernels**
   ```cuda
   __global__ void pc_algorithm_kernel(...);
   __global__ void granger_causality_kernel(...);
   __global__ void ccm_kernel(...);
   ```

4. **Manifold optimization kernels**
   ```cuda
   __global__ void retraction_kernel(...);
   __global__ void parallel_transport_kernel(...);
   __global__ void riemannian_gradient_kernel(...);
   ```

### Success Criteria:
- Comprehensive kernel library
- Well-documented APIs
- Unit tests for each kernel

---

## Task 16: Optimize memory transfers and kernel fusion

### Implementation Steps:

1. **Kernel fusion**
   - Combine matmul + activation
   - Fuse normalization + dropout
   - Merge reductions

2. **Memory optimization**
   - Use pinned memory
   - Implement zero-copy where possible
   - Optimize transfer patterns

3. **Stream parallelism**
   - Multiple CUDA streams
   - Overlap compute/transfer
   - Asynchronous execution

4. **Tensor core utilization**
   - Mixed precision computation
   - Tensor core GEMM operations
   - Automatic mixed precision

### Success Criteria:
- >80% GPU utilization
- >1 TFLOPS sustained
- <10% transfer overhead

---

## Task 17: Benchmark and verify GPU acceleration

### Implementation Steps:

1. **Create benchmark suite**
   - Micro-benchmarks per kernel
   - End-to-end performance tests
   - Scaling studies

2. **Verification suite**
   - Correctness vs CPU reference
   - Numerical stability tests
   - Memory leak detection

3. **Performance profiling**
   - Nsight Systems profiling
   - Kernel occupancy analysis
   - Memory bandwidth analysis

4. **Documentation**
   - Performance reports
   - Optimization guide
   - Best practices

### Success Criteria:
- 100-1000x speedup achieved
- All tests pass
- Complete performance documentation

---

## IMMEDIATE IMPLEMENTATION START: Task 1

Let's begin by fixing gpu_enabled.rs...