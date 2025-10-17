# Mission Charlie: Production Deployment Guide

**Classification:** UNCLASSIFIED//FOR OFFICIAL USE ONLY
**Date:** January 9, 2025
**Version:** 1.0

---

## Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Deployment](#deployment)
6. [Monitoring](#monitoring)
7. [Troubleshooting](#troubleshooting)
8. [Security](#security)
9. [Performance Tuning](#performance-tuning)
10. [API Reference](#api-reference)

---

## Overview

Mission Charlie is a thermodynamic LLM intelligence fusion system that orchestrates multiple Large Language Models (GPT-4, Claude, Gemini, Grok) using physics-inspired consensus algorithms.

### Key Features

- **Multi-LLM Orchestration:** Intelligent routing across 4 major LLM providers
- **Quantum-Inspired Caching:** O(√N) semantic cache with LSH and amplitude amplification
- **Thermodynamic Consensus:** Physics-based agreement from multiple LLM responses
- **Constitutional AI:** Governed by Articles I, III, and IV of PRISM-AI Constitution
- **Production-Grade:** Circuit breakers, graceful degradation, comprehensive logging
- **Mission Bravo Integration:** Seamless fusion with PWSA sensor data

### World-First Algorithms (12 Total)

**Tier 1 (Fully Realized):**
1. Quantum Approximate NN Caching
2. MDL Prompt Optimization
3. PWSA-LLM Integration Bridge

**Tier 2 (Functional Frameworks):**
4. Quantum Voting Consensus
5. PID Synergy Decomposition
6. Hierarchical Active Inference
7. Transfer Entropy Routing

**Tier 3 (Conceptual):**
8. Unified Neuromorphic Processing
9. Bidirectional Causality Analysis
10. Joint Active Inference
11. Geometric Manifold Optimization
12. Quantum Entanglement Measures

---

## System Requirements

### Hardware

- **GPU:** NVIDIA RTX GPU with CUDA Compute Capability ≥ 7.5
  - Recommended: RTX 4090, RTX 5070, A100, H100
  - Minimum: RTX 3070
- **RAM:** 32GB minimum, 64GB recommended
- **Storage:** 100GB free space
- **CPU:** 8+ cores recommended

### Software

- **OS:** Linux (Ubuntu 20.04+, RHEL 8+, or compatible)
- **CUDA:** 12.0 or higher (tested with 12.8)
- **Rust:** 1.70 or higher
- **Network:** Internet access for LLM API calls

### API Keys Required

You must have API keys for at least one of:
- OpenAI (GPT-4)
- Anthropic (Claude)
- Google (Gemini)
- xAI (Grok)

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-org/prism-ai-dod.git
cd prism-ai-dod/src
```

### 2. Set Environment Variables

```bash
# LLM API Keys (set at least one)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="..."
export XAI_API_KEY="xai-..."

# Optional: Mission Charlie configuration overrides
export MISSION_CHARLIE_ENABLE_GPU="true"
export MISSION_CHARLIE_LOG_LEVEL="info"
export MISSION_CHARLIE_CACHE_SIZE="10000"
```

### 3. Build System

```bash
# Build with Mission Charlie + Mission Bravo (full system)
cargo build --release --features mission_charlie,pwsa

# Build Mission Charlie only
cargo build --release --features mission_charlie
```

### 4. Verify Installation

```bash
# Run basic tests
cargo test --features mission_charlie

# Run benchmarks (optional)
cargo bench --features mission_charlie
```

---

## Configuration

### Configuration Files

Mission Charlie supports two configuration formats:

1. **TOML** (Recommended): `mission_charlie_config.toml`
2. **JSON**: `mission_charlie_config.json`

### Creating Configuration

```bash
# Copy example configuration
cp mission_charlie_config.example.toml mission_charlie_config.toml

# Edit with your settings
vim mission_charlie_config.toml
```

### Configuration Sections

#### 1. LLM Configuration

```toml
[llm_config]
global_timeout_secs = 60        # Timeout for LLM requests
max_retries = 3                  # Retry failed requests
enable_cost_tracking = true      # Track API costs

[llm_config.openai]
enabled = true                   # Enable/disable this LLM
model_name = "gpt-4"            # Model to use
temperature = 0.7                # Sampling temperature
max_tokens = 4096                # Maximum response tokens
rate_limit_rpm = 500             # Rate limit (requests/min)
initial_quality = 0.8            # Initial bandit quality score
```

#### 2. Cache Configuration

```toml
[cache_config]
enable_semantic_cache = true     # Enable quantum semantic cache
similarity_threshold = 0.85      # Cache hit threshold (0.0-1.0)
max_cache_size = 10000          # Maximum cached entries
cache_ttl_secs = 3600           # Time-to-live (seconds)
lsh_hash_count = 5              # LSH hash functions
lsh_bucket_count = 100          # LSH buckets
```

#### 3. Consensus Configuration

```toml
[consensus_config]
enable_quantum_voting = true     # Quantum voting algorithm
enable_thermodynamic = true      # Thermodynamic consensus
enable_neuromorphic = false      # Neuromorphic (experimental)
min_llms_for_consensus = 3       # Minimum LLMs to query
confidence_threshold = 0.7       # Consensus acceptance threshold
temperature = 1.0                # Thermodynamic temperature
```

#### 4. Error Handling Configuration

```toml
[error_config]
circuit_breaker_threshold = 5     # Failures before opening circuit
circuit_breaker_timeout_secs = 60 # Timeout before retry
enable_graceful_degradation = true
enable_heuristic_fallback = true
max_recovery_attempts = 3
```

#### 5. Logging Configuration

```toml
[logging_config]
min_level = "info"                # trace, debug, info, warn, error, critical
enable_structured_logging = true
enable_json = false               # JSON logs (for log aggregators)
buffer_size = 1000
enable_request_tracing = true     # Trace requests across system
```

#### 6. Performance Configuration

```toml
[performance_config]
enable_gpu = true                 # GPU acceleration
worker_threads = 4                # Async worker threads
enable_prompt_compression = true  # MDL compression
compression_level = 3             # 0-10 (higher = more compression)
enable_parallel_queries = true    # Query LLMs in parallel
max_parallel_queries = 4
```

### Loading Configuration

**From File:**
```rust
use prism_ai::orchestration::production::MissionCharlieConfig;

let config = MissionCharlieConfig::from_toml_file("mission_charlie_config.toml")?;
```

**From Environment:**
```rust
let config = MissionCharlieConfig::from_env()?;
```

**Programmatically:**
```rust
use prism_ai::orchestration::production::ConfigBuilder;

let config = ConfigBuilder::new()
    .with_openai_key(env::var("OPENAI_API_KEY")?)
    .with_claude_key(env::var("ANTHROPIC_API_KEY")?)
    .enable_gpu(true)
    .with_cache_size(10000)
    .build()?;
```

---

## Deployment

### Standalone Deployment

```rust
use prism_ai::orchestration::LLMOrchestrator;
use prism_ai::orchestration::production::MissionCharlieConfig;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load configuration
    let config = MissionCharlieConfig::from_toml_file("mission_charlie_config.toml")?;

    // Initialize orchestrator
    let orchestrator = LLMOrchestrator::new(config).await?;

    // Query with consensus
    let response = orchestrator.query_with_consensus("What is transfer entropy?").await?;

    println!("Response: {}", response);
    Ok(())
}
```

### Integrated with Mission Bravo

```rust
use prism_ai::orchestration::integration::PwsaLLMFusionPlatform;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize fusion platform
    let platform = PwsaLLMFusionPlatform::new().await?;

    // Process sensor data + LLM context
    let intelligence = platform.fuse_sensor_and_llm_intelligence().await?;

    println!("Threat Level: {:.2}", intelligence.threat_level);
    println!("AI Context: {}", intelligence.llm_context);

    Ok(())
}
```

### Docker Deployment (Recommended)

```dockerfile
FROM nvidia/cuda:12.0-devel-ubuntu22.04

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy source
COPY . /app
WORKDIR /app

# Build
RUN cargo build --release --features mission_charlie,pwsa

# Run
CMD ["./target/release/mission_charlie_server"]
```

Build and run:
```bash
docker build -t mission-charlie:latest .
docker run --gpus all -e OPENAI_API_KEY=$OPENAI_API_KEY mission-charlie:latest
```

---

## Monitoring

### Metrics Collection

Mission Charlie exposes metrics for monitoring:

```rust
use prism_ai::orchestration::production::ProductionMonitoring;

let monitoring = ProductionMonitoring::new();

// Record metrics
monitoring.record_metric("llm_latency_ms", 250.5);
monitoring.increment_counter("llm_requests_total");
monitoring.record_latency("consensus_algorithm", duration);

// Get metrics
let summary = monitoring.get_summary();
println!("Total counters: {}", summary.total_counters);
```

### Logging

Mission Charlie uses structured logging:

```rust
use prism_ai::orchestration::production::{ProductionLogger, LogConfig, LogLevel};

let logger = ProductionLogger::new(LogConfig::default());

// Set request context
logger.set_request_id("req-12345");
logger.set_operation("query_with_consensus");

// Log events
logger.info("Processing LLM query");
logger.warn("Cache miss, querying OpenAI");
logger.error("Rate limit exceeded");

// Get logs
let error_count = logger.get_error_count();
```

### Health Checks

```rust
// Check circuit breaker status
let status = error_handler.get_circuit_breaker_status("openai");
match status {
    CircuitBreakerStatus::Closed => println!("OpenAI: Operational"),
    CircuitBreakerStatus::Open => println!("OpenAI: Circuit Open (degraded)"),
    CircuitBreakerStatus::HalfOpen => println!("OpenAI: Testing recovery"),
}

// Check enabled LLMs
let enabled = config.enabled_llm_names();
println!("Active LLMs: {:?}", enabled);
```

---

## Troubleshooting

### Common Issues

#### 1. "No API key provided"

**Cause:** Missing environment variables or configuration
**Solution:**
```bash
export OPENAI_API_KEY="sk-..."
# Or edit mission_charlie_config.toml
```

#### 2. "Circuit breaker open"

**Cause:** Too many failures to an LLM
**Solution:**
- Check API key validity
- Check rate limits
- Reset circuit breaker:
  ```rust
  error_handler.reset_circuit_breaker("openai");
  ```

#### 3. "CUDA out of memory"

**Cause:** GPU memory exhausted
**Solution:**
- Reduce batch size
- Reduce cache size in config
- Disable GPU: `enable_gpu = false`

#### 4. "Rate limit exceeded"

**Cause:** Too many requests to LLM provider
**Solution:**
- Increase `rate_limit_rpm` in config
- Enable caching: `enable_semantic_cache = true`
- Reduce `max_parallel_queries`

#### 5. "Consensus confidence too low"

**Cause:** LLMs disagree significantly
**Solution:**
- Lower `confidence_threshold` in config
- Increase `min_llms_for_consensus`
- Check if query is ambiguous

### Debug Mode

Enable verbose logging:
```bash
export MISSION_CHARLIE_LOG_LEVEL="debug"
export RUST_LOG="prism_ai::orchestration=trace"
```

---

## Security

### API Key Management

**Never hardcode API keys in source code!**

Use environment variables:
```bash
# ~/.bashrc or /etc/environment
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

Or use a secrets manager (recommended for production):
- AWS Secrets Manager
- HashiCorp Vault
- Kubernetes Secrets

### Data Encryption

Mission Charlie supports encryption for sensitive data:

```rust
use prism_ai::pwsa::vendor_sandbox::KeyManager;

let key_manager = KeyManager::new()?;
let encrypted = key_manager.encrypt(b"sensitive data")?;
let decrypted = key_manager.decrypt(&encrypted)?;
```

### Network Security

- Use HTTPS for all LLM API calls (default)
- Configure firewall to allow only necessary ports
- Use VPN for secure communication in classified environments

### Classification Handling

Mission Charlie is **UNCLASSIFIED**. For classified deployments:
- Deploy on approved classified networks
- Disable logging of sensitive queries
- Use Mission Bravo encryption for sensor data
- Follow DoD security guidelines for AI systems

---

## Performance Tuning

### GPU Optimization

```toml
[performance_config]
enable_gpu = true                 # CRITICAL: Enable for best performance
worker_threads = 8                # Match CPU core count
```

Expected performance:
- **Cache hit:** <1ms latency
- **Single LLM query:** 200-500ms
- **Consensus (3 LLMs):** 500-1500ms

### Cache Tuning

```toml
[cache_config]
similarity_threshold = 0.90       # Higher = fewer hits, more accurate
max_cache_size = 50000           # Larger = more memory, more hits
lsh_hash_count = 7               # More = better accuracy, slower
```

Cache hit rate monitoring:
```rust
let hit_rate = cache.get_hit_rate();
println!("Cache hit rate: {:.1}%", hit_rate * 100.0);
```

### Compression Tuning

```toml
[performance_config]
enable_prompt_compression = true
compression_level = 5             # 0-10, higher = slower but cheaper
```

Compression effectiveness:
- Level 3: ~30% token reduction
- Level 5: ~50% token reduction
- Level 7: ~70% token reduction

### Parallel Queries

```toml
[performance_config]
enable_parallel_queries = true
max_parallel_queries = 4          # Query all LLMs simultaneously
```

Latency improvement:
- Sequential: 4 × 400ms = 1600ms
- Parallel (4): max(400ms) = 400ms
- **4x speedup!**

---

## API Reference

### Core Types

```rust
// Query types
pub enum QueryType {
    Reasoning,      // Complex reasoning tasks
    Factual,        // Factual information retrieval
    Creative,       // Creative generation
    Technical,      // Technical/code tasks
}

// LLM client trait
#[async_trait]
pub trait LLMClient: Send + Sync {
    async fn query(&self, prompt: &str, query_type: QueryType) -> Result<String>;
    fn name(&self) -> &str;
    fn cost_per_token(&self) -> f64;
}
```

### Main Functions

```rust
// Single LLM query
let response = orchestrator.query_single("openai", "What is PRISM-AI?").await?;

// Multi-LLM consensus
let response = orchestrator.query_with_consensus("Explain transfer entropy").await?;

// Thermodynamic consensus
let response = orchestrator.query_with_thermodynamic_consensus(prompt).await?;

// Quantum voting consensus
let response = orchestrator.query_with_quantum_consensus(prompt).await?;
```

### Error Handling

```rust
use prism_ai::orchestration::production::{ProductionErrorHandler, RecoveryAction};

let handler = ProductionErrorHandler::new();

match handler.handle_llm_failure("openai", "rate_limit exceeded")? {
    RecoveryAction::RetryAfterDelay(duration) => {
        tokio::time::sleep(duration).await;
        // Retry query
    }
    RecoveryAction::UseAlternateLLM => {
        // Switch to Claude or Gemini
    }
    RecoveryAction::UseCachedResponse => {
        // Return cached response
    }
    _ => {}
}
```

---

## Support

### Documentation

- **PRISM-AI Constitution:** `/00-Constitution/`
- **Implementation Status:** `/00-Constitution/HONEST-IMPLEMENTATION-STATUS-ANALYSIS.md`
- **Integration Verification:** `/00-Constitution/INTEGRATION-VERIFICATION.md`
- **Source Code:** `/src/src/orchestration/`

### Contact

For support, contact:
- Technical Lead: [classified]
- SBIR Program Manager: [classified]
- Issue Tracker: [internal only]

---

**Classification:** UNCLASSIFIED//FOR OFFICIAL USE ONLY
**Distribution:** Authorized personnel only
**Version:** 1.0
**Last Updated:** January 9, 2025
