# PRISM-AI Runtime Testing Guide

## 🎯 Overview

Now that PRISM-AI compiles successfully (0 errors), runtime testing validates that all systems work correctly during execution. This guide provides a systematic approach to testing the entire platform.

---

## 📋 Testing Phases

### Phase 1: Basic Smoke Tests (5-10 minutes)
Verify core systems can initialize and run basic operations

### Phase 2: Component Testing (30-60 minutes)
Test individual subsystems in isolation

### Phase 3: Integration Testing (1-2 hours)
Test interactions between multiple systems

### Phase 4: Performance Testing (1-2 hours)
Validate performance characteristics and GPU acceleration

### Phase 5: Stress Testing (2-4 hours)
Test under load and edge cases

---

## 🚀 Phase 1: Basic Smoke Tests

### 1.1 Build and Run Main Binary

```bash
# Build the main binary
cargo build --release --features cuda --bin prism

# Run the interactive CLI
./target/release/prism
```

**Expected Output:**
```
╔═══════════════════════════════════════════╗
║          PRISM-AI Interactive CLI         ║
║   Quantum-Neuromorphic Computing System   ║
╚═══════════════════════════════════════════╝

Type 'help' for available commands, 'quit' to exit

PRISM> 
```

**Test Commands:**
```bash
PRISM> status          # Check system status
PRISM> help            # Show available commands
PRISM> te 100          # Test transfer entropy
PRISM> causal 100      # Test causal detection
PRISM> thermo 50       # Test thermodynamic network
PRISM> infer           # Test active inference
PRISM> quit            # Exit
```

**Success Criteria:**
- ✅ Binary runs without panicking
- ✅ All commands execute without errors
- ✅ GPU status displayed correctly
- ✅ Results are numerically reasonable

---

### 1.2 Run Simple Examples

```bash
# Test basic GPU functionality
cargo run --release --features cuda --example test_gpu_simple

# Test information theory
cargo run --release --example enhanced_it_demo

# Test active inference
cargo run --release --example active_inference_robotics_demo
```

**Success Criteria:**
- ✅ Examples compile and run
- ✅ No runtime panics
- ✅ Output shows expected results

---

## 🧪 Phase 2: Component Testing

### 2.1 Information Theory Systems

```bash
# Transfer entropy with financial data
cargo run --release --example transfer_entropy_finance_demo

# Enhanced information theory demo
cargo run --release --example enhanced_it_demo
```

**What to Check:**
- Transfer entropy values are finite and reasonable (0-5 bits typical)
- P-values are between 0 and 1
- Causal direction detection works
- No NaN or Inf values

---

### 2.2 Active Inference & Neuromorphic

```bash
# Active inference for robotics
cargo run --release --example active_inference_robotics_demo

# GPU neuromorphic processing
cargo run --release --features cuda --example gpu_neuromorphic_demo
```

**What to Check:**
- Free energy decreases over time
- Prediction errors converge
- Beliefs update correctly
- GPU kernels execute (if CUDA enabled)

---

### 2.3 Quantum Systems

```bash
# Quantum circuit operations
cargo test --release --lib quantum --features cuda

# QUBO solver
cargo run --release --example universal_solver_demo
```

**What to Check:**
- Quantum gates apply correctly
- State vectors normalized
- Entanglement measures valid
- Optimization converges

---

### 2.4 Thermodynamic Network

```bash
# Run thermodynamic consensus
cargo run --release --example finance_consensus_demo
```

**What to Check:**
- Entropy never decreases (2nd law)
- Boltzmann distribution satisfied
- Phase transitions occur
- Consensus emerges

---

### 2.5 LLM & Transformer Systems

```bash
# Test tokenizer
cargo run --release --example test_bpe_tokenizer

# Test KV cache
cargo run --release --example test_kv_cache

# Test GGUF loader
cargo run --release --example test_gguf_loader

# Full LLM pipeline
cargo run --release --example test_complete_llm_pipeline
```

**What to Check:**
- Tokenization works correctly
- KV cache stores/retrieves properly
- GGUF files load (if available)
- Transformer forward pass completes

---

### 2.6 Time Series Forecasting

```bash
# LSTM forecasting
cargo run --release --example lstm_time_series_complete

# ARIMA forecasting
cargo run --release --example arima_forecasting_tutorial

# Finance forecasting
cargo run --release --example finance_forecast_demo
```

**What to Check:**
- Models train without errors
- Predictions are reasonable
- Loss decreases during training
- Forecasts are continuous

---

### 2.7 GPU Acceleration (if CUDA available)

```bash
# GPU monitoring
cargo run --release --features cuda --example gpu_monitoring_demo

# GPU kernel validation
cargo run --release --features cuda --example gpu_kernel_validation

# Tensor core benchmarks
cargo run --release --features cuda --example tensor_core_performance_benchmark
```

**What to Check:**
- GPU detected and initialized
- Kernels compile and execute
- Memory management works
- Performance improvements over CPU

---

## 🔗 Phase 3: Integration Testing

### 3.1 Cross-Domain Integration

```bash
# Unified platform test
cargo run --release --example prism_ai_unified

# PRISM assistant integration
cargo run --release --example test_prism_assistant
```

**What to Check:**
- Multiple systems work together
- Data flows between components
- No deadlocks or race conditions
- Consensus mechanisms converge

---

### 3.2 Domain-Specific Workflows

```bash
# Healthcare
cargo run --release --example healthcare_trajectory_demo

# Finance
cargo run --release --example portfolio_optimization_demo

# Cybersecurity
cargo run --release --example cybersecurity_threat_demo

# Drug discovery
cargo run --release --example drug_discovery_demo

# Robotics
cargo run --release --example robotics_demo
```

**What to Check:**
- Domain logic executes correctly
- Results make sense for the domain
- No domain-specific errors
- Performance is acceptable

---

### 3.3 Worker Systems (if available)

```bash
# Worker 4 complete demo
cargo run --release --example worker4_complete_demo

# Worker 7 drug discovery
cargo run --release --example worker7_drug_discovery_workflow

# Worker 7 robotics
cargo run --release --example worker7_robotics_motion_planning
```

---

### 3.4 Run Integration Test Suite

```bash
# Core integration tests
cargo test --release --test integration_tests

# Phase 3 integration
cargo test --release --test phase3_integration

# Cross-worker integration
cargo test --release --test cross_worker_integration

# LLM integration
cargo test --release --test llm_integration_test
```

**Success Criteria:**
- ✅ All tests pass
- ✅ No panics or crashes
- ✅ Assertions validate correctly

---

## ⚡ Phase 4: Performance Testing

### 4.1 Benchmark Key Operations

```bash
# LLM performance
cargo run --release --example benchmark_llm_performance

# Tensor core benchmarks
cargo run --release --features cuda --example tensor_core_performance_benchmark

# GPU profiling
cargo run --release --features cuda --example gpu_production_profiler
```

**What to Measure:**
- Operations per second
- Memory usage
- GPU utilization (if applicable)
- Latency and throughput

---

### 4.2 Scalability Tests

Test with increasing problem sizes:

```bash
# Small (baseline)
./target/release/prism
PRISM> thermo 10

# Medium
PRISM> thermo 100

# Large
PRISM> thermo 1000

# Very large (stress test)
PRISM> thermo 10000
```

**What to Check:**
- Performance scales reasonably
- No memory leaks
- No crashes at large scales
- Linear or sub-linear scaling

---

## 🔥 Phase 5: Stress Testing

### 5.1 Long-Running Tests

```bash
# Run overnight
cargo run --release --example prism_ai_unified -- --duration 8h

# Or use the CLI with many iterations
./target/release/prism
PRISM> thermo 1000  # Repeat 100+ times
```

**What to Check:**
- No memory leaks over time
- Performance doesn't degrade
- No deadlocks or hangs
- System remains responsive

---

### 5.2 Edge Cases & Error Handling

Test error conditions:

```bash
# Test with invalid inputs
PRISM> te 0           # Zero samples
PRISM> thermo -1      # Negative size
PRISM> color 0        # Empty graph

# Test with extreme values
PRISM> te 1000000     # Very large dataset
PRISM> thermo 50000   # Massive network
```

**What to Check:**
- Graceful error handling
- Informative error messages
- No panics on invalid input
- System recovers properly

---

## 📊 Success Metrics

### Minimum Requirements (Basic Functionality)
- ✅ Main binary runs without crashing
- ✅ Basic commands work (te, thermo, infer)
- ✅ At least 80% of examples run successfully
- ✅ Core unit tests pass

### Good Status (Production Ready)
- ✅ All commands work correctly
- ✅ 95%+ examples run successfully
- ✅ All integration tests pass
- ✅ Performance acceptable for use cases
- ✅ GPU acceleration works (if applicable)

### Excellent Status (Deployment Ready)
- ✅ 100% examples work
- ✅ All tests pass (unit + integration)
- ✅ Performance benchmarks meet targets
- ✅ Stress tests pass
- ✅ Error handling comprehensive
- ✅ Documentation validated

---

## 🐛 Common Issues & Solutions

### Issue 1: GPU Not Detected
```bash
# Check CUDA installation
nvidia-smi

# Verify cudarc can find CUDA
cargo build --features cuda --example test_gpu_simple
```

**Solution:** Ensure CUDA toolkit installed and LD_LIBRARY_PATH set

### Issue 2: Panic on Division by Zero
**Likely Cause:** Uninitialized data or edge case  
**Solution:** Add validation checks, initialize variables properly

### Issue 3: NaN or Inf Values
**Likely Cause:** Numerical instability  
**Solution:** Add bounds checking, use stable algorithms

### Issue 4: Memory Leaks
```bash
# Use valgrind to detect
valgrind --leak-check=full ./target/release/prism
```

**Solution:** Ensure proper Arc/Rc management, no circular references

### Issue 5: Deadlocks
**Symptoms:** Process hangs, no output  
**Solution:** Review lock ordering, use timeout mechanisms

---

## 📝 Testing Checklist

### Pre-Testing
- [ ] Build completes successfully: `cargo build --release --features cuda`
- [ ] All dependencies installed
- [ ] GPU available (if testing CUDA features)
- [ ] Sufficient disk space and memory

### Basic Tests
- [ ] Main binary runs: `./target/release/prism`
- [ ] Help command works
- [ ] Status command shows system info
- [ ] At least 3 basic commands work (te, thermo, infer)

### Component Tests
- [ ] Information theory examples work
- [ ] Active inference examples work
- [ ] Quantum examples work
- [ ] LLM examples work (or gracefully fail if no models)
- [ ] Time series examples work

### Integration Tests
- [ ] Unified platform example works
- [ ] At least 3 domain demos work
- [ ] Integration test suite passes: `cargo test --release`

### Performance Tests
- [ ] Benchmarks run successfully
- [ ] Performance acceptable for intended use
- [ ] GPU acceleration works (if available)

### Stress Tests
- [ ] System handles large inputs
- [ ] Error conditions handled gracefully
- [ ] Long-running tests complete

---

## 🎯 Quick Start Testing (5 minutes)

If you have limited time, run these essential tests:

```bash
# 1. Build
cargo build --release --features cuda --bin prism

# 2. Run main binary
./target/release/prism
# In PRISM CLI:
status
te 100
thermo 50
quit

# 3. Run a simple example
cargo run --release --example enhanced_it_demo

# 4. Run basic tests
cargo test --release --lib -- --test-threads=1

# 5. Check for panics
echo "✅ If you got here without panics, basic functionality works!"
```

---

## 📈 Next Steps After Testing

### If All Tests Pass ✅
1. Deploy to staging environment
2. Run user acceptance testing
3. Performance optimization (if needed)
4. Production deployment planning

### If Some Tests Fail ⚠️
1. Document failed tests
2. Prioritize critical failures
3. Fix issues systematically
4. Re-test after fixes

### If Many Tests Fail ❌
1. Review compilation warnings (620 warnings to address)
2. Fix critical runtime issues first
3. Improve error handling
4. Consider reverting recent changes

---

## 📚 Additional Resources

- **Examples Directory:** `/src-new/examples/` - 48+ working examples
- **Test Directory:** `/src-new/tests/` - Comprehensive test suite
- **Documentation:** See `VICTORY_REPORT.md` for system overview
- **GPU Guides:** `GPU_QUICK_START.md`, `GPU_KERNEL_INTEGRATION_GUIDE.md`

---

## 🎉 Success Indicators

**Your system is runtime-ready when:**
- ✅ Main binary runs interactive CLI
- ✅ Core algorithms execute correctly
- ✅ No unexpected panics or crashes
- ✅ Results are numerically valid
- ✅ Performance is acceptable
- ✅ GPU acceleration works (if enabled)
- ✅ Error handling is graceful

---

**Good luck with runtime testing! You've already achieved 100% compilation - runtime success is the final step!** 🚀
