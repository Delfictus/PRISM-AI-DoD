# Production-Grade Upgrade Plan - Technical Specification

**Objective**: Upgrade all novel algorithms from functional proof-of-concept to production-grade sophisticated implementations

**Estimated Effort**: 200-400 hours
**Priority**: HIGH for commercial deployment
**Target**: Enterprise-ready, maximum performance

---

## PART 1: TRANSFER ENTROPY ROUTER - Full TE Implementation

### Current State: Correlation Proxy (BASIC)
```rust
// Current: multiply + sum = correlation
let product = executor.elementwise_multiply(&domain, &quality)?;
let sum = executor.reduce_sum(&product)?;
let te_proxy = sum / product.len();
```

### Target State: Full Kraskov-Stögbauer-Grassberger TE (SOPHISTICATED)

**Mathematical Foundation**:
```
TE(X→Y) = I(Y_future; X_past | Y_past)
        = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)
        = ψ(k) + ψ(nₓ) - ψ(nₓᵧ) - ψ(N)

where:
- ψ is digamma function
- k is number of nearest neighbors
- nₓ, nₓᵧ are neighbor counts in different spaces
- N is sample size
```

### Implementation Steps (40-60 hours):

#### Step 1.1: Time-Delay Embedding on GPU (8 hours)
**File**: `src/orchestration/routing/te_embedding_gpu.rs`

```rust
pub struct GpuTimeDelayEmbedding {
    kernel: Arc<CudaFunction>,  // time_delayed_embedding kernel
    context: Arc<CudaContext>,
}

impl GpuTimeDelayEmbedding {
    /// Embed time series: [x(t), x(t-τ), x(t-2τ), ..., x(t-(m-1)τ)]
    pub fn embed_gpu(
        &self,
        time_series_gpu: &CudaSlice<f32>,  // Data STAYS on GPU
        embedding_dim: usize,
        tau: usize,  // Time delay
    ) -> Result<CudaSlice<f32>> {
        // Use time_delayed_embedding kernel
        // Output: [n_samples - (m-1)τ, m] matrix ON GPU
    }
}
```

**Tasks**:
- [ ] Implement GpuTimeDelayEmbedding struct
- [ ] Integrate time_delayed_embedding kernel
- [ ] Handle edge cases (insufficient data, wrap-around)
- [ ] Add automatic τ selection (optimize lag)
- [ ] Test with historical query data

#### Step 1.2: KD-Tree on GPU for k-NN Search (12 hours)
**File**: `src/orchestration/routing/gpu_kdtree.rs`

**Challenge**: KD-tree is tree-based (hard to parallelize)
**Solution**: Use GPU-friendly alternatives:

```rust
pub struct GpuNearestNeighbors {
    /// Use brute-force distance computation (parallelizes well)
    /// For N=1000 samples, GPU brute-force faster than CPU KD-tree
}

impl GpuNearestNeighbors {
    /// Find k nearest neighbors using parallel distance computation
    pub fn find_k_neighbors_gpu(
        &self,
        query_points_gpu: &CudaSlice<f32>,  // [n_query, dim]
        reference_points_gpu: &CudaSlice<f32>,  // [n_ref, dim]
        k: usize,
    ) -> Result<(CudaSlice<i32>, CudaSlice<f32>)> {
        // Returns: (indices[n_query, k], distances[n_query, k])
        // GPU kernel: parallel distance computation + top-k selection
    }
}
```

**New GPU Kernel Needed**:
```cuda
__global__ void knn_distances(
    float* query, float* reference, float* distances,
    int n_query, int n_ref, int dim
);

__global__ void select_k_smallest(
    float* distances, int* indices, int* k_indices, float* k_distances,
    int n, int k
);
```

**Tasks**:
- [ ] Implement parallel distance computation kernel
- [ ] Implement top-k selection kernel (heap-based or bitonic sort)
- [ ] Integrate with existing top_k_sampling kernel
- [ ] Optimize for different data sizes (switch CPU/GPU based on N)
- [ ] Benchmark against CPU kdtree

#### Step 1.3: Full KSG Transfer Entropy Computation (15 hours)
**File**: `src/orchestration/routing/ksg_transfer_entropy_gpu.rs`

```rust
pub struct KSGTransferEntropyGpu {
    kdtree: GpuNearestNeighbors,
    embedding: GpuTimeDelayEmbedding,
    k: usize,  // Number of nearest neighbors (typically 4-10)
}

impl KSGTransferEntropyGpu {
    /// Compute TE(X→Y) using Kraskov-Stögbauer-Grassberger estimator
    ///
    /// Full mathematical formulation:
    /// 1. Embed X and Y with time delays
    /// 2. Form joint space [Y_future, Y_past, X_past]
    /// 3. For each point, find k nearest neighbors in joint space
    /// 4. Count neighbors in marginal spaces
    /// 5. Compute TE = ψ(k) + ⟨ψ(nₓ)⟩ - ⟨ψ(nₓᵧ)⟩ - ψ(N)
    pub fn compute_te_full_gpu(
        &self,
        source_gpu: &CudaSlice<f32>,  // X time series on GPU
        target_gpu: &CudaSlice<f32>,  // Y time series on GPU
        embedding_dim_y: usize,
        embedding_dim_x: usize,
        tau: usize,
    ) -> Result<f32> {
        // 1. Time-delay embedding (GPU)
        let y_embedded = self.embedding.embed_gpu(target_gpu, embedding_dim_y, tau)?;
        let x_embedded = self.embedding.embed_gpu(source_gpu, embedding_dim_x, tau)?;

        // 2. Form joint space [Y_future, Y_past, X_past] (GPU concatenation)
        let joint_space = self.concatenate_spaces_gpu(&y_embedded, &x_embedded)?;

        // 3. k-NN search in joint space (GPU)
        let (joint_neighbors, joint_distances) = self.kdtree.find_k_neighbors_gpu(
            &joint_space, &joint_space, self.k
        )?;

        // 4. Count neighbors in marginal spaces (GPU kernel)
        let nx_counts = self.count_marginal_neighbors_gpu(
            &joint_neighbors, &joint_distances, /* marginal X */
        )?;
        let nxy_counts = self.count_marginal_neighbors_gpu(
            &joint_neighbors, &joint_distances, /* marginal XY */
        )?;

        // 5. Digamma computation and TE formula (GPU)
        let te = self.compute_ksg_formula_gpu(&nx_counts, &nxy_counts, self.k)?;

        Ok(te)
    }
}
```

**New GPU Kernels Needed**:
```cuda
__global__ void concatenate_embeddings(
    float* y_future, float* y_past, float* x_past,
    float* joint_space, int n, int dim_y, int dim_x
);

__global__ void count_marginal_neighbors(
    int* joint_neighbors, float* joint_distances,
    float* reference_marginal, float* marginal_distances,
    int n, int k, int marginal_dim
);

__global__ void digamma_vector(float* n, float* psi_n, int length);

__global__ void ksg_te_formula(
    float* nx_counts, float* nxy_counts,
    float psi_k, float psi_n, float* te_out, int n_points
);
```

**Tasks**:
- [ ] Implement time-delay embedding integration
- [ ] Implement k-NN GPU search
- [ ] Implement marginal neighbor counting
- [ ] Add digamma function approximation on GPU
- [ ] Implement full KSG formula
- [ ] Add statistical significance testing (permutation test on GPU)
- [ ] Validate against reference implementations (JIDT, pyinform)
- [ ] Add auto-parameter selection (optimal k, embedding dims)

#### Step 1.4: Advanced Routing Features (5 hours)

**Tasks**:
- [ ] Multi-dimensional domain detection (not just keywords)
- [ ] Continuous learning from feedback
- [ ] Confidence intervals for TE estimates
- [ ] Outlier detection and handling
- [ ] Model drift detection
- [ ] A/B testing framework

---

## PART 2: THERMODYNAMIC CONSENSUS - Advanced Optimization

### Current State: Basic Boltzmann (FUNCTIONAL)
```rust
// Current: Simple energy, exponential cooling
E = cost - quality + latency
T(t+1) = α * T(t)  // α = 0.95
```

### Target State: Advanced Thermodynamic Optimization (SOPHISTICATED)

**Mathematical Foundation**:
```
Multi-objective optimization with thermodynamic principles:
1. Partition function: Z = Σᵢ exp(-βEᵢ)
2. Free energy: F = -kT ln(Z)
3. Entropy production: dS/dt = β(dE/dt) + d(ln Z)/dt
4. Fokker-Planck dynamics for temperature evolution
5. Replica exchange for better exploration
```

### Implementation Steps (35-50 hours):

#### Step 2.1: Advanced Energy Function (6 hours)
**File**: `src/orchestration/thermodynamic/advanced_energy.rs`

```rust
pub struct AdvancedEnergyModel {
    /// Multi-factor energy function
}

impl AdvancedEnergyModel {
    /// Sophisticated energy formulation
    pub fn compute_energy_gpu(&self, model: &LLMModel, context: &QueryContext) -> Result<f32> {
        // E = w₁·cost + w₂·(-quality) + w₃·latency + w₄·uncertainty + w₅·context_mismatch

        // Factor 1: Cost (with usage-based discounts)
        let cost_energy = self.cost_with_volume_discounts(model, context)?;

        // Factor 2: Quality (task-specific, not global)
        let quality_energy = self.task_specific_quality(model, context)?;

        // Factor 3: Latency (with deadline urgency)
        let latency_energy = self.urgency_weighted_latency(model, context)?;

        // Factor 4: Uncertainty (epistemic + aleatoric)
        let uncertainty_energy = self.bayesian_uncertainty(model, context)?;

        // Factor 5: Context length penalty
        let context_energy = self.context_mismatch_penalty(model, context)?;

        // GPU computation: weighted sum on GPU
        let weights_gpu = self.learned_weights_gpu?;  // Learned via gradient descent
        let factors = vec![cost_energy, quality_energy, latency_energy, uncertainty_energy, context_energy];
        let total_energy = self.weighted_sum_gpu(&factors, &weights_gpu)?;

        Ok(total_energy)
    }

    fn bayesian_uncertainty(&self, model: &LLMModel, context: &QueryContext) -> Result<f32> {
        // Epistemic uncertainty: model parameter uncertainty
        // Aleatoric uncertainty: inherent task randomness
        // Compute using GPU kernels
    }
}
```

**GPU Kernels Needed**:
```cuda
__global__ void weighted_energy_sum(
    float* factors, float* weights, float* energy_out, int n_factors
);

__global__ void bayesian_uncertainty(
    float* historical_variance, float* epistemic, float* aleatoric,
    float* total_uncertainty, int n_samples
);
```

**Tasks**:
- [ ] Implement multi-factor energy model
- [ ] Add task-specific quality estimation
- [ ] Implement Bayesian uncertainty quantification
- [ ] Add context-length penalty computation
- [ ] Learn energy weights via gradient descent on GPU
- [ ] Validate energy function correlates with actual cost/quality

#### Step 2.2: Advanced Temperature Schedules (8 hours)
**File**: `src/orchestration/thermodynamic/temperature_schedules.rs`

```rust
pub enum TemperatureSchedule {
    /// Exponential: T(t) = T₀ α^t
    Exponential { T0: f64, alpha: f64 },

    /// Logarithmic: T(t) = T₀ / log(1 + t)
    Logarithmic { T0: f64 },

    /// Adaptive: T based on acceptance rate
    Adaptive {
        target_acceptance: f64,  // 0.234 for optimal MCMC
        adaptation_rate: f64,
    },

    /// Fokker-Planck: dT/dt = -γT + η·√T·ξ(t)
    FokkerPlanck {
        damping: f64,
        noise_strength: f64,
    },

    /// Replica Exchange: Multiple temperatures in parallel
    ReplicaExchange {
        n_replicas: usize,
        T_min: f64,
        T_max: f64,
        exchange_frequency: usize,
    },
}

impl TemperatureSchedule {
    pub fn update_temperature_gpu(&self, current_T: f64, metrics: &AdaptiveMetrics) -> Result<f64> {
        match self {
            Self::Adaptive { .. } => {
                // Adjust T based on acceptance rate
                // If accepting too much: increase T (explore more)
                // If rejecting too much: decrease T (exploit more)
            },
            Self::FokkerPlanck { .. } => {
                // Use cuRAND for stochastic differential equation
                // dT = -γT dt + η√T dW
            },
            _ => // Other schedules
        }
    }
}
```

**Tasks**:
- [ ] Implement 5 temperature schedules
- [ ] Add adaptive schedule with acceptance rate tracking
- [ ] Implement Fokker-Planck SDE on GPU (cuRAND)
- [ ] Add replica exchange with parallel tempering
- [ ] Implement Metropolis-Hastings temperature swaps
- [ ] Add convergence diagnostics (Gelman-Rubin statistic on GPU)

#### Step 2.3: Parallel Tempering / Replica Exchange (12 hours)
**File**: `src/orchestration/thermodynamic/replica_exchange.rs`

```rust
pub struct ReplicaExchangeConsensus {
    n_replicas: usize,
    temperatures: Vec<f64>,  // Geometric ladder: T_i = T₀ * r^i

    // GPU-resident replica states
    replica_energies_gpu: CudaSlice<f32>,  // [n_replicas, n_models]
    replica_selections_gpu: CudaSlice<i32>,  // [n_replicas]
}

impl ReplicaExchangeConsensus {
    /// Run replica exchange on GPU
    ///
    /// Algorithm:
    /// 1. Each replica at temperature Tᵢ independently samples
    /// 2. Periodically attempt swaps between adjacent replicas
    /// 3. Swap probability: min(1, exp(ΔβΔE))
    /// 4. Coldest replica gives final answer
    pub fn select_with_replica_exchange_gpu(
        &mut self,
        energies: &[f32],
        n_steps: usize,
    ) -> Result<usize> {
        // All replicas evolve on GPU in parallel
        // Swap attempts use cuRAND
        // Final selection from T_min replica
    }
}
```

**GPU Kernels Needed**:
```cuda
__global__ void replica_exchange_step(
    float* energies, int* selections, float* temperatures,
    curandState* rng_states, int n_replicas, int n_models
);

__global__ void attempt_replica_swaps(
    int* selections, float* energies, float* temperatures,
    curandState* rng_states, int n_replicas
);
```

**Tasks**:
- [ ] Implement replica data structures
- [ ] Create geometric temperature ladder
- [ ] Implement parallel replica evolution on GPU
- [ ] Add Metropolis swap criterion
- [ ] Implement swap acceptance tracking
- [ ] Add adaptive temperature spacing
- [ ] Validate convergence is faster than single chain

#### Step 2.4: Bayesian Online Learning (9 hours)
**File**: `src/orchestration/thermodynamic/bayesian_learning.rs`

```rust
pub struct BayesianQualityModel {
    /// Prior: Beta distribution over quality
    /// Update: Bayesian updating from feedback

    // GPU-resident parameters
    alpha_gpu: CudaSlice<f32>,  // [n_models, n_domains]
    beta_gpu: CudaSlice<f32>,   // [n_models, n_domains]
}

impl BayesianQualityModel {
    /// Update beliefs after observing outcome
    pub fn bayesian_update_gpu(
        &mut self,
        model_idx: usize,
        domain_idx: usize,
        observed_quality: f32,
    ) -> Result<()> {
        // Beta distribution conjugate update on GPU
        // α_new = α_old + success_count
        // β_new = β_old + failure_count

        // Compute on GPU for all (model, domain) pairs in parallel
    }

    /// Compute expected quality with uncertainty
    pub fn expected_quality_with_ci_gpu(
        &self,
        model_idx: usize,
        domain_idx: usize,
    ) -> Result<(f32, f32, f32)> {
        // Returns: (mean, lower_95ci, upper_95ci)
        // Computed on GPU
    }
}
```

**Tasks**:
- [ ] Implement Beta distribution on GPU
- [ ] Add conjugate Bayesian updates
- [ ] Compute credible intervals on GPU
- [ ] Track per-domain quality (not global)
- [ ] Add concept drift detection
- [ ] Implement Thompson sampling for exploration

---

## PART 3: LOCAL LLM - Production Transformer

### Current State: Random Weights, Basic Attention
### Target State: Full Llama/Mistral Implementation with Trained Weights

### Implementation Steps (80-120 hours):

#### Step 3.1: GGUF Model Loader (20 hours)
**File**: `src/orchestration/local_llm/gguf_loader.rs`

```rust
pub struct GGUFLoader {
    file_path: String,
}

impl GGUFLoader {
    /// Parse GGUF file and extract metadata + tensors
    pub fn load_model(&self) -> Result<ModelWeights> {
        // 1. Parse GGUF header (version, metadata, tensor count)
        // 2. Read tensor metadata (name, shape, dtype)
        // 3. Read tensor data (weights, biases)
        // 4. Construct model architecture from metadata
    }

    /// Upload weights to GPU
    pub fn upload_to_gpu(
        &self,
        weights: &ModelWeights,
        context: &Arc<CudaContext>,
    ) -> Result<GpuModelWeights> {
        // Upload ALL weights to GPU
        // Convert quantized weights (INT4/INT8) to FP16/FP32
        // Organize for efficient GPU access
    }
}

pub struct ModelWeights {
    embeddings: Vec<f32>,
    layers: Vec<LayerWeights>,
    output: Vec<f32>,
    metadata: ModelMetadata,
}

pub struct GpuModelWeights {
    embeddings_gpu: CudaSlice<f32>,
    layers_gpu: Vec<GpuLayerWeights>,
    output_gpu: CudaSlice<f32>,
}

pub struct GpuLayerWeights {
    wq: CudaSlice<f32>,
    wk: CudaSlice<f32>,
    wv: CudaSlice<f32>,
    wo: CudaSlice<f32>,
    w1: CudaSlice<f32>,  // FFN
    w2: CudaSlice<f32>,
    w3: CudaSlice<f32>,  // SwiGLU gate
    ln_gamma: CudaSlice<f32>,
    ln_beta: CudaSlice<f32>,
}
```

**Tasks**:
- [ ] Implement GGUF v3 parser (latest format)
- [ ] Handle quantized weights (INT4, INT8, FP16)
- [ ] Implement dequantization on GPU
- [ ] Support multiple model architectures (Llama, Mistral, Falcon)
- [ ] Add weight validation and checksums
- [ ] Implement streaming load for large models
- [ ] Add memory-mapped file support

#### Step 3.2: Proper Q/K/V Projections (8 hours)

**Current Issue** (line 117):
```rust
// WRONG: Uses input directly as Q, K, V
unsafe {
    stream.launch_builder(kernel)
        .arg(input)  // Should be Q = input @ Wq
        .arg(input)  // Should be K = input @ Wk
        .arg(input)  // Should be V = input @ Wv
```

**Correct Implementation**:
```rust
pub fn compute_qkv_gpu(&self, input: &CudaSlice<f32>) -> Result<(CudaSlice<f32>, CudaSlice<f32>, CudaSlice<f32>)> {
    // Q = input @ Wq (GPU matmul)
    let q = self.matmul_gpu(input, &self.wq)?;

    // K = input @ Wk (GPU matmul)
    let k = self.matmul_gpu(input, &self.wk)?;

    // V = input @ Wv (GPU matmul)
    let v = self.matmul_gpu(input, &self.wv)?;

    // All stay on GPU
    Ok((q, k, v))
}
```

**Tasks**:
- [ ] Implement proper Q/K/V projections using Wq, Wk, Wv
- [ ] Keep Q/K/V on GPU (no download)
- [ ] Validate attention scores are correct
- [ ] Add attention masking (causal mask for decoder)

#### Step 3.3: KV-Cache Implementation (15 hours)
**File**: `src/orchestration/local_llm/kv_cache.rs`

```rust
pub struct KVCache {
    // Cached K and V for all previous positions
    // Stored on GPU to avoid recomputation

    keys_gpu: Vec<CudaSlice<f32>>,    // [n_layers][past_len, d_model]
    values_gpu: Vec<CudaSlice<f32>>,  // [n_layers][past_len, d_model]

    max_cache_len: usize,
}

impl KVCache {
    /// Update cache with new K/V for current token
    pub fn append_gpu(
        &mut self,
        layer_idx: usize,
        new_k: CudaSlice<f32>,  // [1, d_model]
        new_v: CudaSlice<f32>,  // [1, d_model]
    ) -> Result<()> {
        // Append new K/V to cache on GPU (concat operation)
        // When cache full, implement sliding window or eviction
    }

    /// Get full K/V for attention computation
    pub fn get_kv_gpu(&self, layer_idx: usize) -> Result<(&CudaSlice<f32>, &CudaSlice<f32>)> {
        // Return full cached K and V (on GPU)
    }
}
```

**GPU Kernel Needed**:
```cuda
__global__ void concat_cache(
    float* old_cache, float* new_item, float* updated_cache,
    int old_len, int new_len, int d_model
);
```

**Tasks**:
- [ ] Implement GPU tensor concatenation
- [ ] Add cache size management (LRU eviction)
- [ ] Implement sliding window for long sequences
- [ ] Add cache hit rate tracking
- [ ] Optimize memory layout for coalesced access

#### Step 3.4: Eliminate Feed-Forward Downloads (6 hours)

**Current Issue** (lines 155-180): Downloads weights, computes, uploads
**Fix**: Keep everything on GPU

```rust
fn feed_forward_fused_gpu(&self, input: &CudaSlice<f32>) -> Result<CudaSlice<f32>> {
    // Use fused_linear_gelu kernel
    // Input ON GPU, weights ON GPU, output ON GPU
    // ZERO downloads/uploads

    let exec = self.executor.lock().unwrap();
    let kernel = exec.get_kernel("fused_linear_gelu")?;

    // First projection + GELU (FUSED)
    let mut hidden = stream.alloc_zeros()?;
    kernel.launch(&input, &self.w1, &self.bias1, &mut hidden)?;

    // Second projection (stays on GPU)
    let output = self.matmul_gpu(&hidden, &self.w2)?;

    Ok(output)
}
```

**Tasks**:
- [ ] Use fused_linear_gelu kernel
- [ ] Eliminate ALL downloads in forward pass
- [ ] Keep intermediate results on GPU
- [ ] Add SwiGLU activation (Llama uses this, not GELU)

#### Step 3.5: Top-K / Top-P Sampling on GPU (8 hours)

**Current**: Greedy (argmax) - boring outputs
**Target**: Proper sampling with temperature, top-k, top-p

```rust
pub fn sample_with_temperature_gpu(
    &self,
    logits_gpu: &CudaSlice<f32>,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> Result<i32> {
    // 1. Divide by temperature (GPU)
    let scaled_logits = self.scale_logits_gpu(logits_gpu, temperature)?;

    // 2. Top-k filtering (GPU kernel we have)
    let (top_k_indices, top_k_logits) = self.top_k_gpu(&scaled_logits, top_k)?;

    // 3. Softmax on top-k (GPU)
    let top_k_probs = self.softmax_gpu(&top_k_logits)?;

    // 4. Top-p (nucleus) filtering
    let nucleus_probs = self.nucleus_sampling_gpu(&top_k_probs, top_p)?;

    // 5. Sample from filtered distribution (cuRAND)
    let sampled_idx = self.categorical_sample_gpu(&nucleus_probs)?;
    let token = top_k_indices[sampled_idx];

    Ok(token)
}
```

**GPU Kernel Needed**:
```cuda
__global__ void nucleus_filtering(
    float* sorted_probs, int* indices, float* filtered_probs,
    float p_threshold, int vocab_size
);
```

**Tasks**:
- [ ] Implement top-k selection (use existing kernel)
- [ ] Add top-p (nucleus) filtering
- [ ] Implement temperature scaling
- [ ] Add repetition penalty
- [ ] Validate sampling produces diverse outputs

#### Step 3.6: BPE Tokenizer (12 hours)
**File**: `src/orchestration/local_llm/bpe_tokenizer.rs`

```rust
pub struct BPETokenizer {
    vocab: HashMap<Vec<u8>, i32>,
    merges: Vec<(Vec<u8>, Vec<u8>)>,
    special_tokens: HashMap<String, i32>,
}

impl BPETokenizer {
    /// Load from tokenizer.json (HuggingFace format)
    pub fn from_file(path: &str) -> Result<Self> {
        // Parse tokenizer.json
        // Extract vocab, merges, special tokens
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Result<Vec<i32>> {
        // 1. Pre-tokenization (split on whitespace, punctuation)
        // 2. Byte-level BPE encoding
        // 3. Apply merge rules
        // 4. Map to vocab IDs
    }

    /// Decode token IDs to text
    pub fn decode(&self, tokens: &[i32]) -> Result<String> {
        // 1. Map IDs to byte sequences
        // 2. Concatenate and convert to UTF-8
        // 3. Handle special tokens
    }
}
```

**Tasks**:
- [ ] Parse tokenizer.json format
- [ ] Implement BPE merge algorithm
- [ ] Add special token handling (<|endoftext|>, <|im_start|>, etc.)
- [ ] Handle UTF-8 edge cases
- [ ] Optimize for common case (caching)

#### Step 3.7: Mixed Precision (FP16) (10 hours)

```rust
pub struct MixedPrecisionLLM {
    // Store weights in FP16, compute in FP32 where needed
    weights_fp16_gpu: Vec<CudaSlice<half::f16>>,

    // Use Tensor Cores on RTX 5070 (8x faster for FP16 matmul)
}
```

**GPU Kernel for FP16 Matmul**:
```cuda
// Use __half2 for 2x throughput
__global__ void matmul_fp16_tensor_core(
    __half* a, __half* b, __half* c,
    int m, int k, int n
) {
    // Use HMMA instructions (Tensor Core)
    // 8x faster than FP32 on RTX 5070
}
```

**Tasks**:
- [ ] Convert weight storage to FP16
- [ ] Implement FP16 matmul using Tensor Cores
- [ ] Add automatic mixed precision (AMP)
- [ ] Validate accuracy loss is acceptable
- [ ] Benchmark speedup (should be 5-8x)

---

## PART 4: ACTIVE INFERENCE - Advanced Algorithms

### Current State: Basic Free Energy
### Target State: Advanced Hierarchical Active Inference

### Implementation Steps (30-45 hours):

#### Step 4.1: Hierarchical Belief Propagation (12 hours)
**File**: `src/active_inference/hierarchical_inference_gpu.rs`

```rust
pub struct HierarchicalActiveInferenceGpu {
    // Multi-level hierarchy
    levels: Vec<BeliefLevel>,

    // GPU-resident beliefs
    beliefs_gpu: Vec<CudaSlice<f32>>,  // [n_levels][state_dim]
    precisions_gpu: Vec<CudaSlice<f32>>,  // [n_levels][state_dim, state_dim]
}

impl HierarchicalActiveInferenceGpu {
    /// Message passing between levels on GPU
    pub fn belief_propagation_gpu(&mut self, observations: &CudaSlice<f32>) -> Result<()> {
        // Bottom-up pass: prediction errors
        for level in 0..self.levels.len()-1 {
            let error = self.compute_prediction_error_gpu(level)?;
            self.propagate_error_up_gpu(error, level)?;
        }

        // Top-down pass: predictions
        for level in (0..self.levels.len()).rev() {
            let prediction = self.compute_prediction_gpu(level)?;
            self.propagate_prediction_down_gpu(prediction, level)?;
        }

        // All operations on GPU
    }
}
```

**GPU Kernels Needed**:
```cuda
__global__ void prediction_error(
    float* observation, float* prediction, float* precision,
    float* error_weighted, int n
);

__global__ void belief_update(
    float* prior, float* likelihood, float* precision,
    float* posterior, int state_dim
);
```

**Tasks**:
- [ ] Implement multi-level hierarchy
- [ ] Add precision-weighted prediction errors
- [ ] Implement belief propagation on GPU
- [ ] Add variational message passing
- [ ] Validate against theoretical predictions

#### Step 4.2: Advanced Policy Search (10 hours)
**File**: `src/active_inference/policy_search_gpu.rs`

```rust
pub struct PolicySearchGpu {
    n_policies: usize,
    horizon: usize,

    // Parallel policy evaluation on GPU
    policies_gpu: CudaSlice<f32>,  // [n_policies, horizon, action_dim]
    efe_gpu: CudaSlice<f32>,       // [n_policies] - expected free energy
}

impl PolicySearchGpu {
    /// Evaluate N policies in parallel on GPU
    pub fn evaluate_policies_parallel_gpu(
        &self,
        current_beliefs: &CudaSlice<f32>,
        policies: &CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>> {
        // For each policy (in parallel on GPU):
        // 1. Simulate forward (transition model)
        // 2. Compute expected observations
        // 3. Compute expected free energy
        // 4. Return EFE for all policies (on GPU)

        // Select best policy using argmin on GPU
    }
}
```

**Tasks**:
- [ ] Implement parallel policy rollouts
- [ ] Add model-based planning on GPU
- [ ] Compute expected free energy in parallel
- [ ] Add sophisticated action selection (not just argmin)

#### Step 4.3: Sophisticated Generative Models (8 hours)

**Tasks**:
- [ ] Implement non-linear transition models
- [ ] Add attention-based observation models
- [ ] Implement neural network generative models
- [ ] Add online learning of generative model parameters

---

## PART 5: PRODUCTION FEATURES (40-60 hours)

### Step 5.1: Comprehensive Error Handling (8 hours)

**Tasks**:
- [ ] Add detailed error types for each failure mode
- [ ] Implement automatic recovery from GPU errors
- [ ] Add fallback mechanisms (if GPU fails)
- [ ] Comprehensive input validation
- [ ] Add error logging and telemetry

### Step 5.2: Performance Monitoring (6 hours)

```rust
pub struct PerformanceMonitor {
    gpu_utilization: Arc<AtomicF64>,
    memory_usage: Arc<AtomicU64>,
    kernel_times: HashMap<String, Vec<f64>>,
}
```

**Tasks**:
- [ ] Add nvidia-smi integration for GPU monitoring
- [ ] Track kernel execution times
- [ ] Monitor memory usage
- [ ] Add performance profiling hooks
- [ ] Implement automatic performance regression detection

### Step 5.3: Configuration Management (4 hours)

**Tasks**:
- [ ] Add TOML configuration files
- [ ] Environment-based config (dev/prod)
- [ ] Hot-reload configuration
- [ ] Validation of config parameters

### Step 5.4: Comprehensive Testing (12 hours)

**Tasks**:
- [ ] Unit tests for every kernel
- [ ] Integration tests for full pipelines
- [ ] Property-based testing (hypothesis testing)
- [ ] Benchmark suite with regression tracking
- [ ] Stress testing (memory leaks, long-running)

### Step 5.5: Documentation (10 hours)

**Tasks**:
- [ ] API documentation (rustdoc)
- [ ] Mathematical foundations document
- [ ] Deployment guide
- [ ] Performance tuning guide
- [ ] Example notebooks/tutorials

---

## PART 6: ADVANCED GPU OPTIMIZATIONS (30-40 hours)

### Step 6.1: Tensor Core Utilization (12 hours)

**Use RTX 5070 Tensor Cores for 8x speedup**:
```cuda
// Tensor Core WMMA API
#include <mma.h>
using namespace nvcuda::wmma;

__global__ void matmul_tensor_core(
    half* a, half* b, float* c,
    int m, int n, int k
) {
    // Use wmma::fragment for 16x16x16 tiles
    // 8x faster than CUDA cores for FP16
}
```

**Tasks**:
- [ ] Implement Tensor Core matmul
- [ ] Convert critical paths to FP16
- [ ] Validate accuracy acceptable
- [ ] Benchmark speedup (should be 5-8x)

### Step 6.2: Advanced Kernel Fusion (10 hours)

**More Fused Kernels**:
```cuda
// Fuse entire transformer block
__global__ void fused_transformer_block(
    float* input, float* wq, float* wk, float* wv, float* wo,
    float* w1, float* w2, float* ln_params,
    float* output, int seq_len, int d_model
) {
    // LayerNorm + Attention + FFN + Residual
    // ALL in ONE kernel - eliminates 10+ kernel launches
}
```

**Tasks**:
- [ ] Fuse entire transformer block
- [ ] Fuse TE computation pipeline
- [ ] Fuse thermodynamic selection
- [ ] Benchmark kernel launch overhead reduction

### Step 6.3: Multi-Stream Async Execution (8 hours)

```rust
pub struct AsyncGpuExecutor {
    streams: Vec<Arc<CudaStream>>,

    pub fn execute_async(&self, ops: Vec<GpuOp>) -> Result<()> {
        // Distribute ops across streams
        // Overlap computation and memory transfers
        // Use events for synchronization
    }
}
```

**Tasks**:
- [ ] Create multiple CUDA streams
- [ ] Implement async execution
- [ ] Add event-based synchronization
- [ ] Overlap transfers with computation

---

## GRANULAR TODO LIST

### Transfer Entropy (40 hours)
- [ ] 1.1.1: Implement GpuTimeDelayEmbedding struct
- [ ] 1.1.2: Integrate time_delayed_embedding kernel
- [ ] 1.1.3: Add automatic τ selection
- [ ] 1.2.1: Implement GPU k-NN distance kernel
- [ ] 1.2.2: Implement GPU top-k selection kernel
- [ ] 1.2.3: Benchmark vs CPU KDTree
- [ ] 1.3.1: Implement joint space concatenation
- [ ] 1.3.2: Implement marginal neighbor counting
- [ ] 1.3.3: Add digamma GPU kernel
- [ ] 1.3.4: Implement full KSG formula
- [ ] 1.3.5: Add significance testing
- [ ] 1.3.6: Validate against JIDT
- [ ] 1.4.1: Multi-dimensional domain detection
- [ ] 1.4.2: Continuous learning
- [ ] 1.4.3: Add confidence intervals

### Thermodynamic Consensus (35 hours)
- [ ] 2.1.1: Multi-factor energy model
- [ ] 2.1.2: Task-specific quality
- [ ] 2.1.3: Bayesian uncertainty
- [ ] 2.1.4: Learn energy weights
- [ ] 2.2.1: Implement 5 temperature schedules
- [ ] 2.2.2: Adaptive schedule
- [ ] 2.2.3: Fokker-Planck SDE
- [ ] 2.3.1: Replica exchange framework
- [ ] 2.3.2: Parallel replica evolution
- [ ] 2.3.3: Metropolis swaps
- [ ] 2.3.4: Convergence diagnostics
- [ ] 2.4.1: Beta distribution on GPU
- [ ] 2.4.2: Bayesian updates
- [ ] 2.4.3: Credible intervals
- [ ] 2.4.4: Thompson sampling

### Local LLM (80 hours)
- [ ] 3.1.1: GGUF v3 parser
- [ ] 3.1.2: Quantization handling (INT4/INT8)
- [ ] 3.1.3: GPU weight upload
- [ ] 3.1.4: Model architecture detection
- [ ] 3.2.1: Proper Q/K/V projections
- [ ] 3.2.2: Keep QKV on GPU
- [ ] 3.2.3: Attention masking
- [ ] 3.3.1: KV-cache data structures
- [ ] 3.3.2: GPU concatenation
- [ ] 3.3.3: Cache eviction policy
- [ ] 3.3.4: Sliding window
- [ ] 3.4.1: Use fused_linear_gelu
- [ ] 3.4.2: Eliminate FFN downloads
- [ ] 3.4.3: Add SwiGLU activation
- [ ] 3.5.1: Temperature scaling
- [ ] 3.5.2: Top-k filtering
- [ ] 3.5.3: Top-p (nucleus) sampling
- [ ] 3.5.4: Repetition penalty
- [ ] 3.6.1: Parse tokenizer.json
- [ ] 3.6.2: BPE merge algorithm
- [ ] 3.6.3: Special token handling
- [ ] 3.7.1: FP16 weight conversion
- [ ] 3.7.2: Tensor Core matmul
- [ ] 3.7.3: AMP integration
- [ ] 3.7.4: Accuracy validation

### Active Inference (30 hours)
- [ ] 4.1.1: Multi-level hierarchy
- [ ] 4.1.2: Precision-weighted errors
- [ ] 4.1.3: Message passing GPU
- [ ] 4.2.1: Parallel policy evaluation
- [ ] 4.2.2: Model-based planning
- [ ] 4.2.3: Sophisticated action selection
- [ ] 4.3.1: Non-linear transition models
- [ ] 4.3.2: Neural observation models
- [ ] 4.3.3: Online learning

### Production Features (40 hours)
- [ ] 5.1: Error handling
- [ ] 5.2: Performance monitoring
- [ ] 5.3: Configuration management
- [ ] 5.4: Comprehensive testing
- [ ] 5.5: Documentation

### Advanced Optimizations (30 hours)
- [ ] 6.1: Tensor Core implementation
- [ ] 6.2: Advanced kernel fusion
- [ ] 6.3: Multi-stream async

---

## PRIORITIZED EXECUTION PLAN

### Phase 1: High Commercial Value (2-3 weeks)
**Focus**: Thermodynamic Consensus + TE Router to production-grade

**Week 1**:
- Transfer Entropy: Full KSG implementation (Steps 1.1-1.3)
- Thermodynamic: Advanced energy + schedules (Steps 2.1-2.2)

**Week 2**:
- Transfer Entropy: Advanced features (Step 1.4)
- Thermodynamic: Replica exchange (Step 2.3)

**Week 3**:
- Thermodynamic: Bayesian learning (Step 2.4)
- Integration testing and validation

### Phase 2: LLM Production-Ready (3-4 weeks)
**Focus**: Load actual models, proper inference

**Week 4-5**:
- GGUF loader (Step 3.1)
- Proper attention (Step 3.2)
- BPE tokenizer (Step 3.6)

**Week 6**:
- KV-cache (Step 3.3)
- Proper sampling (Step 3.5)

**Week 7**:
- FP16 + Tensor Cores (Step 3.7)
- Feed-forward optimization (Step 3.4)

### Phase 3: Production Hardening (2-3 weeks)
**Week 8-9**:
- Error handling (Step 5.1)
- Testing (Step 5.4)
- Monitoring (Step 5.2)

**Week 10**:
- Documentation (Step 5.5)
- Final optimizations (Part 6)

---

## SUCCESS METRICS

### Transfer Entropy Router:
- [ ] Computes actual TE (not correlation proxy)
- [ ] Uses full histogram->MI->conditional MI pipeline
- [ ] Validates against JIDT reference (< 5% error)
- [ ] Processes 1000+ variable systems
- [ ] < 100ms for full causal network

### Thermodynamic Consensus:
- [ ] Multi-factor energy model
- [ ] 5+ temperature schedules
- [ ] Replica exchange operational
- [ ] Bayesian online learning
- [ ] Demonstrates 40-70% cost savings in simulation

### Local LLM:
- [ ] Loads actual Llama-7B weights
- [ ] Proper BPE tokenization
- [ ] KV-cache working
- [ ] Proper top-p sampling
- [ ] 50-100 tokens/sec on RTX 5070
- [ ] Outputs are coherent and diverse

### Active Inference:
- [ ] Hierarchical inference operational
- [ ] Advanced policy search
- [ ] Online learning of generative models
- [ ] Real-time performance (< 1ms)

---

## ESTIMATED TOTAL EFFORT

**Transfer Entropy**: 40 hours
**Thermodynamic Consensus**: 35 hours
**Local LLM**: 80 hours
**Active Inference**: 30 hours
**Production Features**: 40 hours
**Advanced Optimizations**: 30 hours

**TOTAL**: **255 hours** (6-7 weeks full-time)

**Critical Path**: Local LLM (longest, most complex)
**Quick Wins**: Thermodynamic + TE (highest commercial value, less work)

---

## RECOMMENDATION

**Prioritize for maximum commercial impact**:

1. **Complete Thermodynamic Consensus** (35 hours) - $5M-$20M value
2. **Complete Transfer Entropy Router** (40 hours) - $3M-$15M value
3. **Production features** (40 hours) - Enterprise-ready
4. **Then** Local LLM if needed (80 hours)

**Rationale**: Thermodynamic + TE are novel, patent-worthy, and solve real enterprise problems. LLM is nice-to-have but llama.cpp exists.

**Total to production-grade commercial product**: **115 hours** (3 weeks)

---

This plan transforms proof-of-concept into production-grade sophisticated implementations.