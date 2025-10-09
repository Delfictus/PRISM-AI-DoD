# Getting Started

**Quick start guide for PRISM-AI development**

---

## Prerequisites

### Required
- **Rust:** 1.75+ (stable)
- **CUDA Toolkit:** 12.0+ (12.8 tested)
- **NVIDIA GPU:** RTX 3060+ (Compute Capability 8.0+)
- **NVIDIA Driver:** 525.60.13+ (Linux) or 527.41+ (Windows)
- **Git:** 2.0+

### Recommended
- **RAM:** 8GB+ (16GB for large problems)
- **Storage:** 2GB for source + build
- **OS:** Linux (Ubuntu 22.04 tested)

---

## Installation

### 1. Install CUDA Toolkit

**Ubuntu/Debian:**
```bash
# Add NVIDIA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install CUDA
sudo apt-get install -y cuda-toolkit-12-8

# Add to PATH
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

# Verify
nvcc --version
nvidia-smi
```

### 2. Install Rust

```bash
# Install rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Verify
rustc --version
cargo --version
```

### 3. Clone Repository

```bash
git clone https://github.com/Delfictus/PRISM-AI.git
cd PRISM-AI
```

---

## Building

### Standard Build
```bash
# Debug build (faster compilation)
cargo build

# Release build (optimized)
cargo build --release
```

### Build Output
```
Compiling prism-ai v0.1.0
warning: ALL CUDA kernels compiled successfully
warning: 23 kernels ready
Finished in 15s
```

### What Gets Built
- Main library: `target/release/libprism_ai.rlib`
- CUDA kernels: `target/ptx/*.ptx` (23 files)
- Dependencies: `target/release/deps/`

---

## Testing

### Run All Tests
```bash
# All tests (takes ~27 seconds)
cargo test --lib --release

# Expected output:
# test result: ok. 218 passed; 0 failed
```

### Run Specific Tests
```bash
# Module tests
cargo test --lib information_theory
cargo test --lib active_inference
cargo test --lib cma

# Single test
cargo test --lib test_transfer_entropy
```

### Run Integration Tests
```bash
cargo test --test transfer_entropy_tests
```

---

## Documentation

### Generate Docs
```bash
# Build and open docs
cargo doc --lib --no-deps --open

# Docs location: target/doc/prism_ai/index.html
```

### View Existing Docs
```bash
# Open in browser
firefox target/doc/prism_ai/index.html

# Or
xdg-open target/doc/prism_ai/index.html
```

---

## Development Workflow

### 1. Make Changes
Edit files in `src/`

### 2. Check Compilation
```bash
# Quick check (no build)
cargo check

# With warnings
cargo build --release 2>&1 | grep "warning:"
```

### 3. Run Tests
```bash
# Affected tests only
cargo test --lib your_module_name

# All tests
cargo test --lib --release
```

### 4. Format Code
```bash
# Format
cargo fmt

# Check formatting
cargo fmt -- --check
```

### 5. Lint
```bash
# Run clippy
cargo clippy --lib

# Auto-fix where possible
cargo clippy --fix --lib --allow-dirty
```

### 6. Commit
```bash
git add .
git commit -m "feat: your change description"
git push
```

---

## Common Commands

### Building
```bash
cargo build                  # Debug build
cargo build --release        # Release build
cargo clean                  # Clean build artifacts
```

### Testing
```bash
cargo test --lib            # All lib tests
cargo test --lib --release  # Optimized tests
cargo test <pattern>        # Tests matching pattern
```

### Quality
```bash
cargo check                 # Check compilation
cargo fmt                   # Format code
cargo clippy                # Lint code
cargo fix --lib            # Auto-fix issues
```

### Documentation
```bash
cargo doc                   # Generate docs
cargo doc --open            # Generate and open docs
cargo doc --no-deps         # Skip dependencies
```

---

## Project Structure

```
PRISM-AI/
├── src/
│   ├── lib.rs                  # Main library entry
│   ├── mathematics/            # Math foundations
│   ├── information_theory/     # Transfer entropy
│   ├── statistical_mechanics/  # Thermodynamics
│   ├── active_inference/       # Active inference
│   ├── integration/            # Cross-domain
│   ├── resilience/             # Fault tolerance
│   ├── optimization/           # Performance
│   ├── cma/                    # Phase 6 CMA
│   ├── neuromorphic/           # Neuromorphic engine
│   ├── quantum/                # Quantum engine
│   ├── foundation/             # Platform foundation
│   ├── shared-types/           # Common types
│   ├── prct-core/              # PRCT algorithm
│   ├── prct-adapters/          # Adapters
│   └── mathematics/            # Math utils
├── examples/                   # Demo applications
├── tests/                      # Integration tests
├── benches/                    # Benchmarks
├── benchmarks/                 # Benchmark data
├── cuda/                       # CUDA kernel source
├── docs/                       # Documentation
└── target/                     # Build output
```

---

## GPU Setup

### Verify GPU
```bash
# Check GPU
nvidia-smi

# Should show:
# - GPU name (e.g., RTX 5070)
# - Driver version
# - CUDA version
```

### Test GPU
```bash
# Run GPU test (when examples are fixed)
cargo run --example test_gpu_detection --release
```

### GPU Features
```bash
# Enable CUDA (default)
cargo build --features cuda

# CPU simulation (deprecated)
cargo build --features simulation
```

---

## Troubleshooting

### CUDA Not Found
```bash
# Add to ~/.bashrc
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

# Reload
source ~/.bashrc
```

### Compilation Errors
```bash
# Clean and rebuild
cargo clean
cargo build --release
```

### Tests Failing
```bash
# Run verbose
cargo test --lib -- --nocapture

# Run single test
cargo test --lib test_name -- --nocapture
```

### Out of Memory
```bash
# Reduce parallel jobs
cargo build -j 2

# Or use debug build
cargo build
```

---

## Using as a Library

### In Another Project

**Cargo.toml:**
```toml
[dependencies]
prism-ai = { git = "https://github.com/Delfictus/PRISM-AI.git" }
```

**Example Usage:**
```rust
use prism_ai::{
    TransferEntropy,
    HierarchicalModel,
    CircuitBreaker,
};

fn main() {
    let model = HierarchicalModel::new();
    println!("Model created!");
}
```

---

## Next Steps

After setup:
1. ✅ [[Development Workflow]] - Learn the development process
2. ✅ [[Module Reference]] - Understand the modules
3. ✅ [[API Documentation]] - API reference
4. ✅ [[Active Issues]] - See what needs work
5. ✅ [[Testing Guide]] - Write and run tests

---

## Getting Help

- **Documentation:** `cargo doc --open`
- **Issues:** [[Active Issues]]
- **Contact:** BV@Delfictus.com, IS@Delfictus.com
- **Repository:** https://github.com/Delfictus/PRISM-AI

---

*Last Updated: 2025-10-04*
