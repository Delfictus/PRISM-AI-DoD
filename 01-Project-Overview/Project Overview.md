# Project Overview

## What is PRISM-AI?

**PRISM: Predictive Reasoning via Information-theoretic Statistical Manifolds**

PRISM-AI is a GPU-accelerated hybrid computing platform that combines:
- **Neuromorphic computing** analogues
- **Quantum computing** analogues
- **Active Inference** framework
- **Information-theoretic coupling**

All implemented in pure software (Rust + CUDA), no special hardware required beyond an NVIDIA GPU.

---

## Key Features

### 1. **Information Theory**
- Transfer entropy analysis
- Causal direction detection
- KSG (Kozachenko-Leonenko) estimator
- GPU-accelerated computation

### 2. **Active Inference**
- Hierarchical generative models
- Variational inference
- Policy selection
- Free energy minimization

### 3. **Causal Manifold Annealing (CMA)**
- Phase 6 precision refinement
- Ensemble generation
- Quantum annealing simulation
- Neural network integration
- Mathematical guarantees (PAC-Bayes, Conformal Prediction, ZKP)

### 4. **Resilience & Performance**
- Circuit breakers for fault tolerance
- Health monitoring
- Checkpoint/restore functionality
- GPU kernel optimization
- Performance auto-tuning

---

## Architecture Phases

### ✅ Phase 0: Governance Infrastructure (100%)
- Validation framework
- Git hooks
- CI/CD integration

### ✅ Phase 1: Mathematical Foundations (100%)
- Transfer entropy
- Thermodynamic consistency
- GPU validation (647x speedup)

### ✅ Phase 2: Active Inference (100%)
- Hierarchical models
- Recognition and observation models
- Policy controllers

### ✅ Phase 3: Integration Architecture (100%)
- Cross-domain bridges
- Information channels
- 8-phase pipeline

### ✅ Phase 4: Production Hardening (100%)
- Error recovery
- Performance optimization
- Circuit breakers

### 🔄 Phase 5: Validation (Ready)
- Scientific validation suite
- DARPA demonstrations

### ✅ Phase 6: CMA Framework (100%)
- 4-week implementation complete
- 6,000+ lines production code
- Full GPU integration

---

## Technology Stack

### Languages
- **Rust** - Main implementation (62,861 lines)
- **CUDA** - GPU kernels (2,696 lines)
- **Markdown** - Documentation (40,272 lines)

### Key Dependencies
- `tokio` - Async runtime
- `ndarray` - Numerical arrays
- `cudarc` - CUDA bindings
- `candle-core` - Neural network operations
- `nalgebra` - Linear algebra
- `statrs` - Statistics

### Hardware Requirements
- **GPU:** NVIDIA RTX 3060+ (Compute Capability 8.0+)
- **CUDA:** Version 12.0+
- **RAM:** 8GB+ recommended
- **Storage:** 2GB for source + build

---

## Use Cases

1. **Financial Market Analysis**
   - Causal discovery in time series
   - Prediction with uncertainty

2. **Scientific Computing**
   - Information flow analysis
   - Complex system modeling

3. **Optimization Problems**
   - TSP solving
   - Graph coloring
   - Combinatorial optimization

4. **Materials Discovery** (Planned)
   - Property prediction
   - Synthesis planning

5. **Drug Design** (Planned)
   - Binding affinity prediction
   - Molecular optimization

---

## Performance Highlights

- **647x speedup:** Thermodynamic evolution (GPU vs CPU)
- **40-180x speedup:** TSP solving vs classical algorithms
- **<2ms latency:** Active inference decisions
- **23 GPU kernels:** Custom CUDA implementations
- **218 tests:** All passing, 100% success rate

---

## Project Status

**Current State:** Functional but needs polish

### What Works ✅
- Compiles without errors
- All tests pass
- GPU acceleration operational
- Core algorithms implemented
- Library can be used via git dependency

### What Needs Work ⚠️
- 109 compiler warnings (mostly unused code)
- 10 example files have broken imports
- Documentation gaps
- 4 incomplete GPU features
- Not published to crates.io

### Next Steps 🎯
1. Clean up remaining warnings
2. Fix example imports
3. Add comprehensive documentation
4. Publish to crates.io
5. Create demo applications

---

## Team & Contacts

- **Technical Lead:** Benjamin Vaccaro - BV@Delfictus.com
- **Scientific Advisor:** Ididia Serfaty - IS@Delfictus.com
- **Repository:** https://github.com/Delfictus/PRISM-AI

---

## Related Documents

- [[Architecture Overview]] - Technical architecture
- [[Current Status]] - Detailed status tracking
- [[Module Reference]] - Module documentation
- [[Getting Started]] - Development setup

---

*Last Updated: 2025-10-04*
