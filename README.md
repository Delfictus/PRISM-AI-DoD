# PRISM-AI DoD
## Unified Platform: World Record + PWSA SBIR + LLM Orchestration

**Status:** Release v1.0.0
**Hardware:** 8× H200 GPUs
**Timeline:** Production Ready

---

## Triple Mission

### 1. Graph Coloring World Record (Alpha)
- **Target:** ≤82 colors on DSJC1000-5
- **Current:** 130 colors
- **Plan:** [plans/ULTRA_TARGETED_WORLD_RECORD_PLAN.md](docs/plans/ULTRA_TARGETED_WORLD_RECORD_PLAN.md)

### 2. PWSA Data Fusion SBIR (Bravo)
- **Target:** $1.5-2M Phase II funding
- **Requirement:** <5ms fusion latency
- **Plan:** [development/rapid-implementation/](docs/development/rapid-implementation/)

### 3. Thermodynamic LLM Orchestration (Charlie)
- **Target:** Patent-worthy consensus system
- **Innovation:** Physics-based multi-LLM orchestration
- **Plan:** [plans/THERMODYNAMIC_LLM_INTEGRATION.md](docs/plans/THERMODYNAMIC_LLM_INTEGRATION.md)

---

## Repository Structure

```
DoD/
├── docs/                              # Centralized documentation
│   ├── getting-started/               # Quick start guides
│   ├── architecture/                  # System design & missions
│   ├── governance/                    # Constitutional enforcement
│   ├── development/                   # Contributing & testing
│   └── plans/                         # Strategic planning docs
│
├── src/                               # Latest working PRISM-AI code
│   ├── Cargo.toml
│   └── src/
│       ├── core/                      # Shared components (ALL missions)
│       ├── quantum/                   # Quantum annealing
│       ├── neuromorphic/              # Spiking networks
│       └── integration/               # Transfer entropy, active inference
│
└── examples/                          # Example implementations
```

---

## Quick Start

See [Getting Started Guide](docs/getting-started/quick-start.md) for detailed instructions.

### Build & Test
```bash
cd src
cargo build --release --features constitutional_validation
cargo test --all-features
```

### Deploy Missions
```bash
# Alpha: World Record
cargo run --release --bin world_record_attempt

# Bravo: PWSA Demo  
cargo run --release --bin pwsa_demo

# Charlie: LLM Orchestration
cargo run --release --bin llm_consensus
```

---

## Governance Enforcement

### Build-Time
- Mandatory trait implementation
- 95% test coverage requirement
- Memory limit validation

### Runtime
- Entropy non-decreasing check
- Convergence monitoring
- GPU utilization >80%
- Automatic violation handling

### Deployment Gates
- Performance benchmarks must pass
- Security audit required
- Constraint validation mandatory

---

## Shared Infrastructure

All three missions leverage:
- **Transfer Entropy** - Information flow analysis
- **Active Inference** - Free energy minimization
- **Neuromorphic Computing** - Spike-based processing
- **GPU Acceleration** - H200 optimization
- **Constitutional Constraints** - Thermodynamic laws

---

## Documentation

📚 **[Complete Documentation](docs/index.md)** - Comprehensive docs in the `docs/` directory

---

**Classification:** UNCLASSIFIED // FOR OFFICIAL USE ONLY  
**Distribution:** PRISM-AI Development Team