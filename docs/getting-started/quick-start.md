# Quick Start Guide

## Prerequisites

- Rust 1.70+ with GPU support
- 8× H200 GPUs (for full performance)
- CUDA 12.0+

## Build & Test

```bash
cd src-new
cargo build --release --features constitutional_validation
cargo test --all-features
```

## Run Governance Checks

```bash
cargo run --bin governance_validator
```

## Deploy Missions

### Alpha: World Record
```bash
cargo run --release --bin world_record_attempt
```

### Bravo: PWSA Demo
```bash
cargo run --release --bin pwsa_demo
```

### Charlie: LLM Orchestration
```bash
cargo run --release --bin llm_consensus
```

## Next Steps

- See [Installation Guide](installation.md) for detailed setup
- Read [Architecture Overview](../architecture/overview.md) for system design
- Check [Development Guide](../development/contributing.md) for contribution guidelines