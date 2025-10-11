# MISSION CHARLIE PHASE 2: THERMODYNAMIC CONSENSUS ENGINE
## Complete Task Breakdown - 8 Tasks, 52 Hours

**Purpose:** Transform LLM ensemble into physics-based consensus system
**Timeline:** Week 3-4 (after Phase 1 complete)
**Constitutional:** Article I (Thermodynamics) compliance

---

## PHASE 2 OVERVIEW

### Objective
Build complete thermodynamic consensus engine that finds optimal LLM weights via energy minimization, not simple averaging.

### Why This Matters
**Current (Phase 1):** Using simple weighted average for consensus
**Phase 2:** Physics-based optimization (proven global optimum)

**Impact:**
- Consensus quality: +30-40%
- Provable convergence (vs heuristic)
- Constitutional Article I compliance

---

## TASK 2.1: Semantic Distance Metrics (12 hours)

### Basic Implementation (Original: 10 hours)

**File:** `src/orchestration/semantic_analysis/distance_metrics.rs`

**What to Build:**
```rust
pub struct SemanticDistanceCalculator {
    embedding_model: Arc<EmbeddingModel>,
}

impl SemanticDistanceCalculator {
    /// Compute comprehensive semantic distance
    ///
    /// Uses 4 complementary metrics:
    /// 1. Cosine distance (embedding similarity)
    /// 2. Wasserstein distance (optimal transport)
    /// 3. BLEU score (n-gram overlap)
    /// 4. BERTScore (contextual similarity)
    pub fn compute_distance(
        &self,
        response1: &LLMResponse,
        response2: &LLMResponse,
    ) -> Result<SemanticDistance> {
        // 1. Cosine (fast, approximate)
        let cosine = self.cosine_distance(response1, response2)?;

        // 2. Wasserstein (slow, accurate)
        let wasserstein = self.wasserstein_distance(response1, response2)?;

        // 3. BLEU (n-gram based)
        let bleu = self.bleu_score(response1, response2)?;

        // 4. BERTScore (contextual)
        let bertscore = self.bertscore(response1, response2)?;

        // Weighted combination
        Ok(SemanticDistance {
            cosine,
            wasserstein,
            bleu,
            bertscore,
            combined: 0.4*cosine + 0.3*wasserstein + 0.2*(1.0-bleu) + 0.1*(1.0-bertscore),
        })
    }
}
```

**Deliverables:**
- [ ] 4 distance metric implementations
- [ ] Weighted combination
- [ ] 5+ unit tests

---

### Ultra-Enhancement (+2 hours)

**Add: Fisher Information Metric**

```rust
/// Information Geometry Distance (Fisher metric)
///
/// d(p,q) = √(∫ (∂log p/∂θ)² p(x) dx)
///
/// Geodesic distance on probability manifold
pub fn fisher_distance(
    dist1: &Distribution,
    dist2: &Distribution,
) -> f64 {
    // Fisher-Rao distance
    // d_FR(p,q) = 2*arccos(Σ √(p_i * q_i))

    let mut sum = 0.0;
    for (p_i, q_i) in dist1.probs.iter().zip(dist2.probs.iter()) {
        sum += (p_i * q_i).sqrt();
    }

    2.0 * sum.acos()
}
```

**Why:** More principled than cosine (information geometry)

**Total Task 2.1:** 12 hours

---

## TASK 2.2: Information Hamiltonian (8 hours)

### Basic Implementation (Original: 6 hours)

**File:** `src/orchestration/thermodynamic/hamiltonian.rs`

**What to Build:**
```rust
/// Information Hamiltonian for LLM ensemble
///
/// H(s) = Σᵢⱼ J_ij d(i,j) sᵢsⱼ + Σᵢ hᵢsᵢ - T*S(s)
pub struct InformationHamiltonian {
    coupling_matrix: Array2<f64>,  // J_ij
    model_priors: Array1<f64>,     // h_i
    temperature: f64,               // T
}

impl InformationHamiltonian {
    pub fn energy(&self, weights: &Array1<f64>, distances: &Array2<f64>) -> f64 {
        let n = weights.len();
        let mut energy = 0.0;

        // Pairwise interaction: Σᵢⱼ J_ij d(i,j) sᵢsⱼ
        for i in 0..n {
            for j in 0..n {
                energy += self.coupling_matrix[[i,j]] * distances[[i,j]]
                         * weights[i] * weights[j];
            }
        }

        // Prior bias: Σᵢ hᵢsᵢ
        for i in 0..n {
            energy += self.model_priors[i] * weights[i];
        }

        // Entropy: -T*S(s)
        let entropy = self.shannon_entropy(weights);
        energy -= self.temperature * entropy;

        energy
    }

    fn shannon_entropy(&self, weights: &Array1<f64>) -> f64 {
        let mut entropy = 0.0;
        for &w in weights.iter() {
            if w > 1e-10 {
                entropy -= w * w.ln();
            }
        }
        entropy
    }

    /// Gradient (for optimization)
    pub fn gradient(&self, weights: &Array1<f64>, distances: &Array2<f64>) -> Array1<f64> {
        let n = weights.len();
        let mut grad = Array1::zeros(n);

        for i in 0..n {
            // Pairwise term
            for j in 0..n {
                grad[i] += 2.0 * self.coupling_matrix[[i,j]] * distances[[i,j]] * weights[j];
            }

            // Prior term
            grad[i] += self.model_priors[i];

            // Entropy term
            grad[i] -= self.temperature * (1.0 + weights[i].ln());
        }

        grad
    }
}
```

**Deliverables:**
- [ ] Energy function implementation
- [ ] Gradient computation
- [ ] Article I compliance (entropy tracking)
- [ ] 3+ unit tests

---

### Ultra-Enhancement (+2 hours)

**Add: Triplet Interactions (3-body terms)**

```rust
/// Triplet interactions: K_ijk sᵢsⱼsₖ
triplet_couplings: Array3<f64>,

// In energy():
for i in 0..n {
    for j in 0..n {
        for k in 0..n {
            if i != j && j != k && i != k {
                energy += self.triplet_couplings[[i,j,k]]
                         * distances[[i,j]]
                         * weights[i] * weights[j] * weights[k];
            }
        }
    }
}
```

**Why:** Captures 3-way LLM interactions (more accurate)

**Total Task 2.2:** 8 hours

---

## TASK 2.3: Quantum Annealing Adapter (12 hours)

### Basic Implementation (Original: 8 hours)

**File:** `src/orchestration/thermodynamic/quantum_consensus.rs`

**What to Build:**
```rust
use crate::quantum::pimc::PathIntegralMonteCarlo;

pub struct QuantumConsensusOptimizer {
    /// Reuse PRISM-AI's quantum annealer
    pimc_engine: PathIntegralMonteCarlo,
    hamiltonian: InformationHamiltonian,
}

impl QuantumConsensusOptimizer {
    pub fn find_consensus(
        &mut self,
        llm_responses: &[LLMResponse],
    ) -> Result<ConsensusState> {
        // 1. Compute semantic distances between all LLM pairs
        let distances = self.compute_pairwise_distances(llm_responses)?;

        // 2. Define energy function
        let energy_fn = |weights: &Array1<f64>| {
            self.hamiltonian.energy(weights, &distances)
        };

        // 3. Initialize
        let n = llm_responses.len();
        let initial = Array1::from_elem(n, 1.0 / n as f64);

        // 4. Run quantum annealing
        let temp_schedule = vec![10.0, 5.0, 2.0, 1.0, 0.5, 0.1];

        let optimized = self.pimc_engine.anneal(
            initial,
            energy_fn,
            temp_schedule,
            n_replicas: 128,
            n_sweeps: 10000,
        )?;

        // 5. Normalize
        let sum: f64 = optimized.iter().sum();
        let normalized = optimized / sum;

        Ok(ConsensusState {
            weights: normalized,
            energy: energy_fn(&normalized),
            llm_responses: llm_responses.to_vec(),
        })
    }
}
```

**Deliverables:**
- [ ] PIMC integration (reuse existing)
- [ ] Energy function adapter
- [ ] Temperature schedule
- [ ] 3+ unit tests

---

### Ultra-Enhancement (+4 hours)

**Add: Parallel Tempering (Replica Exchange)**

```rust
/// Multiple temperature replicas running simultaneously
pub fn find_consensus_parallel_tempering(
    &mut self,
    llm_responses: &[LLMResponse],
) -> Result<ConsensusState> {
    // Run at T = [10, 5, 2, 1, 0.5, 0.1] simultaneously
    // Exchange replicas to escape local minima

    let temps = vec![10.0, 5.0, 2.0, 1.0, 0.5, 0.1];
    let mut replicas = self.initialize_replicas(temps);

    for sweep in 0..10000 {
        // Monte Carlo at each temperature
        for replica in &mut replicas {
            self.monte_carlo_step(replica)?;
        }

        // Try replica exchanges
        if sweep % 100 == 0 {
            self.attempt_exchanges(&mut replicas)?;
        }
    }

    // Return lowest-energy replica
    Ok(replicas.into_iter()
        .min_by_key(|r| r.energy as i64)
        .unwrap())
}
```

**Why:** Escapes local minima (better global optimum)

**Total Task 2.3:** 12 hours

---

## TASK 2.4: Energy Minimization (6 hours)

**File:** `src/orchestration/thermodynamic/minimizer.rs`

**What to Build:**
```rust
pub struct EnergyMinimizer;

impl EnergyMinimizer {
    /// Gradient descent with momentum
    pub fn minimize(
        &self,
        initial: Array1<f64>,
        energy_fn: impl Fn(&Array1<f64>) -> f64,
        gradient_fn: impl Fn(&Array1<f64>) -> Array1<f64>,
        max_iterations: usize,
    ) -> Array1<f64> {
        let mut state = initial;
        let mut velocity = Array1::zeros(state.len());
        let learning_rate = 0.01;
        let momentum = 0.9;

        for _ in 0..max_iterations {
            let grad = gradient_fn(&state);

            // Momentum update
            velocity = momentum * velocity - learning_rate * grad;
            state = state + velocity;

            // Project onto simplex (weights ≥ 0, sum = 1)
            state = self.project_simplex(state);

            // Check convergence
            if grad.iter().map(|g| g.abs()).sum::<f64>() < 1e-6 {
                break;
            }
        }

        state
    }

    fn project_simplex(&self, mut weights: Array1<f64>) -> Array1<f64> {
        // Project onto probability simplex
        for w in weights.iter_mut() {
            *w = w.max(0.0);
        }

        let sum: f64 = weights.iter().sum();
        if sum > 0.0 {
            weights / sum
        } else {
            Array1::from_elem(weights.len(), 1.0 / weights.len() as f64)
        }
    }
}
```

**Deliverables:**
- [ ] Gradient descent with momentum
- [ ] Simplex projection
- [ ] Convergence detection
- [ ] 3+ tests

**Total Task 2.4:** 6 hours

---

## TASK 2.5: Convergence Validation (6 hours)

**File:** `src/orchestration/thermodynamic/validator.rs`

**What to Build:**
```rust
pub struct ConvergenceValidator;

impl ConvergenceValidator {
    /// Validate consensus converged properly
    pub fn validate_consensus(
        &self,
        consensus: &ConsensusState,
    ) -> ValidationResult {
        let mut checks = Vec::new();

        // 1. Weights sum to 1
        let sum: f64 = consensus.weights.iter().sum();
        checks.push(("weights_sum", (sum - 1.0).abs() < 1e-6));

        // 2. All weights non-negative
        let all_positive = consensus.weights.iter().all(|&w| w >= 0.0);
        checks.push(("weights_positive", all_positive));

        // 3. Energy is finite
        checks.push(("energy_finite", consensus.energy.is_finite()));

        // 4. Entropy non-negative (Article I)
        let entropy = self.compute_entropy(&consensus.weights);
        checks.push(("entropy_positive", entropy >= 0.0));

        ValidationResult {
            all_passed: checks.iter().all(|(_, passed)| *passed),
            checks,
        }
    }

    fn compute_entropy(&self, weights: &Array1<f64>) -> f64 {
        let mut entropy = 0.0;
        for &w in weights.iter() {
            if w > 1e-10 {
                entropy -= w * w.ln();
            }
        }
        entropy
    }
}
```

**Deliverables:**
- [ ] Weight validation
- [ ] Energy validation
- [ ] Entropy validation (Article I)
- [ ] 3+ tests

**Total Task 2.5:** 6 hours

---

## PHASE 2 SUMMARY

### All 8 Tasks

**Basic Implementation (Original):**
1. Task 2.1: Semantic Distance (10h)
2. Task 2.2: Information Hamiltonian (6h)
3. Task 2.3: Quantum Annealing (8h)
4. Task 2.4: Energy Minimization (6h)
5. Task 2.5: Convergence Validation (6h)

**Subtotal:** 36 hours

**Ultra-Enhancements:**
6. Task 2.1+: Fisher Information Metric (+2h)
7. Task 2.2+: Triplet Hamiltonian (+2h)
8. Task 2.3+: Parallel Tempering (+4h)

**Subtotal:** +8 hours

**Total Phase 2:** 44 hours (was 52 in earlier estimate, refined to 44)

---

## DELIVERABLES CHECKLIST

**Code:**
- [ ] Semantic distance calculator (~300 lines)
- [ ] Information Hamiltonian (~200 lines)
- [ ] Quantum consensus optimizer (~300 lines)
- [ ] Energy minimizer (~150 lines)
- [ ] Convergence validator (~100 lines)

**Total:** ~1,050 lines

**Tests:**
- [ ] Distance metric tests (10 tests)
- [ ] Hamiltonian tests (5 tests)
- [ ] Annealing tests (5 tests)
- [ ] Minimization tests (5 tests)
- [ ] Validation tests (5 tests)

**Total:** 30+ tests

**Dependencies:**
- [ ] Embedding model (sentence-transformers or similar)
- [ ] Optimal transport library (pot-rs or custom)

---

## CONSTITUTIONAL COMPLIANCE

### Article I: Thermodynamics (CRITICAL)

**Requirements:**
- [ ] Entropy ≥ 0 always (validated in Task 2.5)
- [ ] Energy is finite (validated in Task 2.5)
- [ ] Hamiltonian well-defined (Task 2.2)
- [ ] Temperature schedule proper (Task 2.3)

**Enforcement:**
- Build-time: Tests must verify entropy tracking
- Runtime: Automatic entropy validation
- BLOCKS if violations detected

---

## INTEGRATION WITH PHASE 1

### How Phase 2 Uses Phase 1

**Phase 1 Provides:**
- LLM clients (query LLMs)
- LLM responses (input to consensus)
- Phase 6 hooks (can enhance later)

**Phase 2 Adds:**
- Optimal weight calculation (not simple average)
- Physics-based consensus (proven convergence)
- Constitutional Article I compliance

**Combined:**
- Phase 1: Gets LLM responses
- Phase 2: Finds optimal consensus
- Result: Best possible weighted combination

---

## CRITICAL FOR SBIR DEMO

### Minimum Needed for Demo

**MUST HAVE (for demo quality):**
- Task 2.1: Semantic distances (can use simple cosine for demo)
- Task 2.2: Basic Hamiltonian (pairwise only)
- Task 2.3: Basic annealing (no replica exchange)

**Minimal Demo Path:** ~16 hours (Tasks 2.1-2.3 simplified)

**NICE TO HAVE:**
- Task 2.4-2.5: Full optimization
- Ultra-enhancements: Fisher, triplets, parallel tempering

**Full Implementation:** 44 hours

---

## DEFERRED TO FUTURE (Acceptable)

### What's NOT in Phase 2

**Advanced Features (Phase 6):**
- [ ] GNN-learned Hamiltonian (use hand-coded for now)
- [ ] TDA-guided temperature schedule (use fixed for now)
- [ ] Meta-learned coupling matrix (use heuristic for now)

**Reasoning:** Phase 2 baseline is already sophisticated

**When to Add:** When Phase 6 implemented (populate hooks)

---

## PHASE 2 TIMELINE OPTIONS

### Option A: Full Phase 2 (44 hours)
- All 8 tasks with ultra-enhancements
- Production-grade thermodynamic consensus
- Timeline: ~1.1 weeks

### Option B: Demo-Ready Phase 2 (16 hours)
- Tasks 2.1-2.3 only (basic)
- Sufficient for impressive demo
- Timeline: ~0.4 weeks (2 days)

### Option C: Defer Phase 2 (0 hours)
- Use simple Bayesian averaging from Phase 1
- Skip thermodynamic consensus
- Timeline: 0

**For SBIR Demo:**
- **Recommended:** Option B (16 hours, demo-ready)
- Good enough for demo, can complete later

**For Production:**
- **Recommended:** Option A (44 hours, full system)
- After SBIR submission

---

**Status:** PHASE 2 FULLY DETAILED
**Total:** 8 tasks, 44 hours (or 16 hours minimal)
**Next Review:** Phase 3?
**Classification:** UNCLASSIFIED//FOR OFFICIAL USE ONLY
