# GEMINI AUDIT - CRITICAL FILE LIST
**Date**: October 14, 2025
**Purpose**: Prioritized list of source files for Google Gemini code audit
**Base Path**: `/home/diddy/Desktop/PRISM-AI-DoD/03-Source-Code/`

---

## PRIORITY 1: SPACE FORCE SBIR DEMONSTRATION (CRITICAL)

### Graph Coloring Implementation
**File**: `src/quantum/src/gpu_coloring.rs` (701 lines)
**Why Critical**: Core Space Force SBIR requirement - graph coloring for satellite scheduling
**What to Verify**:
- Jones-Plassmann parallel coloring algorithm implementation
- Real GPU kernel launches (not CPU fallback)
- Adjacency matrix GPU construction
- Production-ready vs proof-of-concept

**Associated Tests**: `tests/test_gpu_coloring.rs`

---

### TSP Optimization Implementation
**File**: `src/quantum/src/gpu_tsp.rs` (467 lines)
**Why Critical**: Core Space Force SBIR requirement - mission planning optimization
**What to Verify**:
- GPU-accelerated 2-opt optimization
- Parallel evaluation of O(n²) swap candidates
- Distance matrix GPU computation
- Claimed 50× speedup validity

**Associated Tests**: `tests/test_gpu_tsp.rs`

---

## PRIORITY 2: GPU INFRASTRUCTURE (FOUNDATION)

### GPU Context Management
**File**: `src/gpu/context.rs`
**Why Critical**: Foundation for all GPU operations - if this is fake, everything is fake
**What to Verify**:
- Real CUDA device initialization (`cuInit`, `cuDeviceGet`)
- Actual CUDA context creation (`cuCtxCreate`)
- Not just wrapper around CPU code
- Proper error handling

---

### GPU Memory Management
**File**: `src/gpu/memory.rs`
**Why Critical**: Real GPU acceleration requires real GPU memory operations
**What to Verify**:
- Actual `cuMemAlloc` calls (not Vec<T> in disguise)
- Host-to-Device transfers (`cuMemcpyHtoD`)
- Device-to-Host transfers (`cuMemcpyDtoH`)
- Memory lifecycle management (no leaks)

---

### GPU Module and Kernel Loading
**File**: `src/gpu/module.rs`
**Why Critical**: This is where PTX kernels are loaded - verify kernels exist
**What to Verify**:
- `cuModuleLoad` or `cuModuleLoadData` calls
- PTX source code strings or file paths
- `cuModuleGetFunction` to retrieve kernel handles
- Not empty/stub implementations

---

### GPU Main Module
**File**: `src/gpu/mod.rs`
**Why Critical**: Public API and feature flags
**What to Verify**:
- Check `#[cfg(feature = "cuda")]` conditions
- Verify GPU code isn't always disabled
- Public exports of GPU types

---

## PRIORITY 3: CUDA KERNELS (REALITY CHECK)

### Kernel Source Directory
**Files**: `src/gpu/kernels/*.ptx` or `src/gpu/kernels/*.cu`
**Why Critical**: 61 kernels claimed - need to verify they exist
**What to Verify**:
- Count actual PTX or CUDA source files
- Kernels are not empty stubs
- Kernel count matches claim (61 kernels)

**Expected Kernels by Category**:

#### Graph Coloring Kernels (4+)
- `build_adjacency` - Construct adjacency matrix from coupling matrix
- `init_priorities` - Initialize vertex priorities for Jones-Plassmann
- `find_independent_set` - Find independent set in parallel
- `color_vertices` - Assign colors to independent set

#### TSP Kernels (2+)
- `compute_distances` - Calculate distance matrix from coordinates
- `evaluate_2opt_swaps` - Parallel evaluation of 2-opt swap deltas

#### Transfer Entropy Kernels (8+)
- `compute_joint_distribution` - Build joint probability distributions
- `compute_marginal_distribution` - Marginalize distributions
- `calculate_entropy` - Entropy calculation
- `calculate_conditional_entropy` - Conditional entropy
- `transfer_entropy_kernel` - Full TE computation
- `mutual_information_kernel` - MI computation

#### Active Inference Kernels (6+)
- `compute_free_energy` - Variational free energy
- `update_beliefs` - Belief propagation
- `precision_weighted_prediction_error` - Precision-weighted PE
- `expected_free_energy` - Expected free energy for policy selection

#### Protein Folding Kernels (4+)
- `calculate_forces` - Force calculation for molecular dynamics
- `energy_minimization` - Gradient descent on GPU
- `contact_map_computation` - Residue contact predictions
- `rmsd_calculation` - RMSD between structures

#### Neural Network Kernels (10+)
- `matmul` - Matrix multiplication
- `conv2d` - 2D convolution
- `relu` - ReLU activation
- `softmax` - Softmax activation
- `cross_entropy_loss` - Loss computation
- `adam_update` - Adam optimizer step

---

## PRIORITY 4: TRANSFER ENTROPY (CORE ALGORITHM)

### Transfer Entropy Main Module
**File**: `src/information_theory/transfer_entropy.rs`
**Why Critical**: Core PRISM-AI algorithm - if this is simplified, whole system is compromised
**What to Verify**:
- Actual TE calculation (not placeholder)
- Handling of history length k
- Joint distribution estimation
- Conditional entropy computation

---

### GPU Transfer Entropy
**File**: `src/information_theory/gpu_te.rs`
**Why Critical**: GPU acceleration of TE is key differentiator
**What to Verify**:
- GPU kernel launches for TE computation
- Parallel histogram construction
- GPU-accelerated entropy calculations
- Not just CPU code with "gpu" in filename

---

### Causal Graph Construction
**File**: `src/information_theory/causal_graph.rs`
**Why Critical**: TE builds causal graphs - verify this works
**What to Verify**:
- Uses TE values to construct directed graphs
- Statistical significance testing
- Handles multiple time series
- Produces meaningful causal structure

---

## PRIORITY 5: API SERVER (DEPLOYMENT READINESS)

### API Server Main Module
**File**: `src/api_server/mod.rs`
**Why Critical**: Production deployment requires working API
**What to Verify**:
- Server actually starts and listens
- Routes are registered
- Proper error handling
- Not just scaffold/template

---

### REST Routes
**File**: `src/api_server/routes.rs`
**Why Critical**: REST API claimed as primary interface
**What to Verify**:
- Count actual endpoints (claim: 42+)
- Endpoints call real PRISM-AI functions
- Not mock/stub responses
- Input validation present

---

### GraphQL API
**File**: `src/api_server/graphql.rs`
**Why Critical**: GraphQL claimed as advanced query interface
**What to Verify**:
- Full schema definition
- Resolvers connected to real functions
- Coverage of all 15 domains
- Complex queries actually work

---

### WebSocket Support
**File**: `src/api_server/websocket.rs`
**Why Critical**: Real-time streaming claimed
**What to Verify**:
- WebSocket handlers implemented
- Can stream results incrementally
- Not just HTTP upgrade placeholder

---

## PRIORITY 6: DRUG DISCOVERY (HIGH-VALUE APPLICATION)

### Molecular Docking
**File**: `src/applications/drug_discovery/docking.rs`
**Why Critical**: Drug discovery is major market claim
**What to Verify**:
- Real docking algorithm (not simplified)
- Force field implementation (AMBER/CHARMM or simplified?)
- 3D conformer generation
- RDKit dependency status (missing but needed?)

**Red Flags to Check**:
- Comments like "simplified chemistry"
- Missing RDKit in Cargo.toml
- Using Euclidean distance as energy (too simple)

---

### Scoring Functions
**File**: `src/applications/drug_discovery/scoring.rs`
**Why Critical**: Docking requires proper scoring
**What to Verify**:
- Scoring function matches published methods (Vina, Glide, etc.)
- Not just distance-based scoring
- Terms include: van der Waals, electrostatics, H-bonds, desolvation

---

### Protein Folding
**File**: `src/orchestration/local_llm/gpu_protein_folding.rs`
**Why Critical**: Claimed AlphaFold2 competition capability
**What to Verify**:
- Real energy potentials (not simplified Euclidean)
- Gradient-based optimization on GPU
- Can handle no-homolog proteins
- Competitive with AlphaFold2 or overstated claim?

---

### Molecular Dynamics
**File**: `src/physics/molecular_dynamics.rs`
**Why Critical**: MD simulation for drug discovery and materials science
**What to Verify**:
- Real force field implementation
- Integration scheme (Verlet, Leapfrog?)
- Temperature/pressure control
- Not just particle simulation

---

## PRIORITY 7: APPLICATION DOMAINS (15 CLAIMED)

### Finance Application
**File**: `src/applications/finance/mod.rs`
**What to Verify**:
- Portfolio optimization uses real TE
- Risk assessment algorithms
- GPU acceleration present

---

### Cybersecurity Application
**File**: `src/applications/cybersecurity/mod.rs`
**What to Verify**:
- Anomaly detection implementation
- Causal graph for threat detection
- Real-time capability

---

### Supply Chain Application
**File**: `src/applications/supply_chain/mod.rs`
**What to Verify**:
- Route optimization uses TSP/VRP
- Demand forecasting algorithms
- GPU-accelerated optimization

---

### Healthcare Application
**File**: `src/applications/healthcare/mod.rs`
**What to Verify**:
- Disease progression modeling
- Treatment recommendation system
- Medical time series analysis

---

### Robotics Application
**File**: `src/applications/robotics/mod.rs`
**What to Verify**:
- Real-time control algorithms
- GPU-accelerated planning
- Sensor fusion

---

### Energy Grid Application
**File**: `src/applications/energy_grid/mod.rs`
**What to Verify**:
- Load balancing algorithms
- Fault detection using TE
- Real-time grid analysis

---

### Manufacturing Application
**File**: `src/applications/manufacturing/mod.rs`
**What to Verify**:
- Predictive maintenance
- Quality control algorithms
- Process optimization

---

### Agriculture Application
**File**: `src/applications/agriculture/mod.rs`
**What to Verify**:
- Crop yield prediction
- Resource optimization
- Environmental monitoring

---

### Scientific Computing Application
**File**: `src/applications/scientific_computing/mod.rs`
**What to Verify**:
- Numerical methods implementation
- GPU-accelerated simulations
- Parallel algorithms

---

### Telecommunications Application
**File**: `src/applications/telecommunications/mod.rs`
**What to Verify**:
- Network optimization
- Traffic prediction
- Resource allocation

---

## PRIORITY 8: TESTING AND VALIDATION

### Integration Tests Directory
**Path**: `tests/`
**Why Critical**: 95.54% pass rate - verify test quality
**What to Verify**:
- Count tests (claimed: 539)
- Tests are substantive (not just "doesn't panic")
- GPU tests actually run on GPU
- Edge cases covered

**Key Test Files**:
- `tests/test_gpu_coloring.rs` - Graph coloring validation
- `tests/test_gpu_tsp.rs` - TSP optimization validation
- `tests/test_transfer_entropy.rs` - TE correctness
- `tests/test_api_server.rs` - API integration tests
- `tests/test_protein_folding.rs` - Protein folding validation

**Test Quality Metrics**:
- Average lines per test (>10 is substantive)
- Assertion count (multiple assertions per test)
- Input scale (large enough to stress GPU)
- Ground truth comparisons (known correct answers)

---

### Benchmark Suite
**Path**: `benches/`
**Why Critical**: Performance claims need benchmark evidence
**What to Verify**:
- CPU vs GPU comparisons exist
- Realistic problem sizes
- Speedup measurements include transfer costs
- Benchmarks for claimed 50-100× speedup

**Expected Benchmarks**:
- `benches/graph_coloring_bench.rs` - 20× claimed
- `benches/tsp_bench.rs` - 50× claimed
- `benches/transfer_entropy_bench.rs` - 100× claimed
- `benches/matrix_ops_bench.rs` - 10-50× claimed

---

## PRIORITY 9: BUILD AND DEPLOYMENT

### Cargo Configuration
**File**: `Cargo.toml`
**Why Critical**: Dependencies reveal true capabilities
**What to Verify**:
- CUDA bindings present (cust/cudarc/cuda-sys)
- Dependencies are stable versions (not git repos)
- Optional features properly configured
- RDKit presence/absence

**Critical Dependencies to Find**:
```toml
[dependencies]
cust = "0.3"  # or cudarc or cuda-sys
ndarray = "0.15"
tokio = "1.35"
axum = "0.7"  # or actix-web
async-graphql = "6.0"
```

**Missing Dependencies to Note**:
- RDKit bindings (if absent, drug discovery limited)
- ONNX Runtime (if absent, some AI features limited)

---

### Build Script
**File**: `build.rs`
**Why Critical**: Shows if CUDA kernels are compiled from source
**What to Verify**:
- PTX compilation from .cu files
- NVCC invocation
- Kernel embedding into binary
- Feature flag handling

---

### Docker Configuration
**File**: `Dockerfile` or `deployment/docker/Dockerfile`
**Why Critical**: Container deployment claimed
**What to Verify**:
- NVIDIA CUDA base image
- GPU runtime support
- Production-ready configuration
- Not just development container

---

### Deployment Documentation
**File**: `docs/DEPLOYMENT.md` or similar
**Why Critical**: Production deployment requires documentation
**What to Verify**:
- Clear deployment steps
- GPU requirements specified
- Configuration options documented
- Health checks defined

---

## PRIORITY 10: DOCUMENTATION VS REALITY

### Production Readiness Report
**File**: `PRODUCTION_READINESS_REPORT.md`
**Why Critical**: Compare claims to actual code
**What to Verify**:
- Are claimed features actually implemented?
- Are test pass rates accurate?
- Are performance metrics backed by benchmarks?
- Are limitations honestly disclosed?

---

### Performance Metrics
**File**: `PERFORMANCE_METRICS.txt`
**Why Critical**: Verify performance claims
**What to Check**:
- Speedup measurements exist
- Realistic workloads used
- Statistical significance
- Repeatability

---

### README and User Documentation
**File**: `README.md`, `docs/*.md`
**Why Critical**: Marketing vs reality check
**What to Verify**:
- Features described are implemented
- No "coming soon" features listed as ready
- Installation instructions work
- Examples are runnable

---

## AUDIT EXECUTION ORDER

Execute audit in this order for maximum efficiency:

1. **Start with GPU Infrastructure** (Priority 2)
   - If GPU is fake, rest doesn't matter
   - Files: `src/gpu/*.rs`
   - Time: 2 hours

2. **Verify CUDA Kernels Exist** (Priority 3)
   - If no kernels, GPU acceleration is fake
   - Files: `src/gpu/kernels/*`
   - Time: 2 hours

3. **Space Force SBIR Critical** (Priority 1)
   - Graph coloring and TSP - must work
   - Files: `src/quantum/src/gpu_coloring.rs`, `src/quantum/src/gpu_tsp.rs`
   - Time: 3 hours

4. **Transfer Entropy Core** (Priority 4)
   - Core algorithm must be solid
   - Files: `src/information_theory/*.rs`
   - Time: 2 hours

5. **API Server Deployment** (Priority 5)
   - Production requires working API
   - Files: `src/api_server/*.rs`
   - Time: 1.5 hours

6. **Drug Discovery and Protein Folding** (Priority 6)
   - High-value applications
   - Files: `src/applications/drug_discovery/*.rs`, `src/orchestration/local_llm/gpu_protein_folding.rs`
   - Time: 2 hours

7. **Application Domains** (Priority 7)
   - 15 domains - sample audit
   - Files: `src/applications/*/mod.rs`
   - Time: 2 hours

8. **Testing and Validation** (Priority 8)
   - Verify test quality and benchmarks
   - Files: `tests/*.rs`, `benches/*.rs`
   - Time: 1.5 hours

9. **Build and Deployment** (Priority 9)
   - Verify buildability and deployment readiness
   - Files: `Cargo.toml`, `build.rs`, `Dockerfile`
   - Time: 1 hour

10. **Documentation Reality Check** (Priority 10)
    - Compare claims to code
    - Files: `*.md`
    - Time: 1 hour

**Total Audit Time**: 18 hours (2-3 days for thorough review)

---

## SUMMARY STATISTICS

**Total Files to Audit**: ~80-100 files
**Critical Path Files**: 15 files (Priorities 1-3)
**Code to Review**: ~50,000-70,000 lines of Rust
**Test Files**: ~50 test files (539 tests)
**Benchmark Files**: ~10 benchmark files
**Documentation**: ~20 markdown files

---

**END OF FILE LIST**
