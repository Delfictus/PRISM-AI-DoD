# LLM API Connection Setup Guide

## Overview
PRISM-AI's Mission Charlie integrates 4 LLM providers for quantum-consensus intelligence. This guide shows exactly what environment variables and configuration you need to connect the actual APIs.

---

## Required Environment Variables

### 1. OpenAI (GPT-4)
```bash
export OPENAI_API_KEY="sk-..."
```
- Get your API key from: https://platform.openai.com/api-keys
- Model: `gpt-4` (configurable)
- Rate limit: 500 requests/min (configurable)

### 2. Anthropic Claude
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```
- Get your API key from: https://console.anthropic.com/settings/keys
- Model: `claude-3-5-sonnet-20241022` (configurable)
- Rate limit: 500 requests/min (configurable)

### 3. Google Gemini
```bash
export GEMINI_API_KEY="..."
```
- Get your API key from: https://aistudio.google.com/app/apikey
- Model: `gemini-2.0-flash-exp` (configurable)
- Rate limit: 500 requests/min (configurable)

### 4. xAI Grok
```bash
export XAI_API_KEY="xai-..."
```
- Get your API key from: https://console.x.ai/
- Model: `grok-2-1212` (configurable)
- Rate limit: 500 requests/min (configurable)

---

## Optional Environment Variables

### Performance Tuning
```bash
# Enable/disable GPU acceleration
export MISSION_CHARLIE_ENABLE_GPU=true

# Set log level (debug, info, warn, error)
export MISSION_CHARLIE_LOG_LEVEL=info

# Set cache size (number of entries)
export MISSION_CHARLIE_CACHE_SIZE=10000
```

---

## Configuration Methods

### Method 1: Environment Variables Only (Simplest)
```bash
# Set environment variables
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="..."
export XAI_API_KEY="xai-..."

# Run PRISM-AI (will load keys from environment)
cargo run --bin prism
```

### Method 2: Configuration File + Environment
Create `config/mission_charlie.toml`:

```toml
[llm_config]
global_timeout_secs = 60
max_retries = 3
enable_cost_tracking = true

[llm_config.openai]
enabled = true
api_key_env = "OPENAI_API_KEY"
model_name = "gpt-4"
temperature = 0.7
max_tokens = 4096
rate_limit_rpm = 500
initial_quality = 0.8

[llm_config.claude]
enabled = true
api_key_env = "ANTHROPIC_API_KEY"
model_name = "claude-3-5-sonnet-20241022"
temperature = 0.7
max_tokens = 4096
rate_limit_rpm = 500
initial_quality = 0.85

[llm_config.gemini]
enabled = true
api_key_env = "GEMINI_API_KEY"
model_name = "gemini-2.0-flash-exp"
temperature = 0.7
max_tokens = 4096
rate_limit_rpm = 500
initial_quality = 0.75

[llm_config.grok]
enabled = true
api_key_env = "XAI_API_KEY"
model_name = "grok-2-1212"
temperature = 0.7
max_tokens = 4096
rate_limit_rpm = 500
initial_quality = 0.7

[cache_config]
enable_semantic_cache = true
similarity_threshold = 0.85
max_cache_size = 10000
cache_ttl_secs = 3600
lsh_hash_count = 5
lsh_bucket_count = 100

[consensus_config]
enable_quantum_voting = true
enable_thermodynamic = true
enable_neuromorphic = false
min_llms_for_consensus = 3
confidence_threshold = 0.7
temperature = 1.0

[error_config]
circuit_breaker_threshold = 5
circuit_breaker_timeout_secs = 60
enable_graceful_degradation = true
enable_heuristic_fallback = true
max_recovery_attempts = 3

[logging_config]
min_level = "info"
enable_structured_logging = true
enable_json = false
buffer_size = 1000
enable_request_tracing = true

[performance_config]
enable_gpu = true
worker_threads = 4
enable_prompt_compression = true
compression_level = 3
enable_parallel_queries = true
max_parallel_queries = 4
```

Then load it in code:
```rust
use prism_ai::orchestration::production::config::MissionCharlieConfig;

let config = MissionCharlieConfig::from_toml_file_with_env("config/mission_charlie.toml")?;
```

### Method 3: Programmatic Configuration
```rust
use prism_ai::orchestration::production::config::ConfigBuilder;

let config = ConfigBuilder::new()
    .with_openai_key("sk-...".to_string())
    .with_claude_key("sk-ant-...".to_string())
    .with_gemini_key("...".to_string())
    .with_grok_key("xai-...".to_string())
    .enable_gpu(true)
    .with_cache_size(10000)
    .with_log_level("info".to_string())
    .build()?;
```

---

## Using the LLM Clients

### Basic Usage
```rust
use prism_ai::orchestration::llm_clients::{
    OpenAIClient, ClaudeClient, GeminiClient, GrokClient, LLMClient
};

// Initialize clients with API keys
let openai = OpenAIClient::new(
    std::env::var("OPENAI_API_KEY")?,
    "gpt-4".to_string()
);

let claude = ClaudeClient::new(
    std::env::var("ANTHROPIC_API_KEY")?,
    "claude-3-5-sonnet-20250110".to_string()
);

// Generate response
let response = openai.generate("Explain quantum entanglement", 0.7).await?;
println!("Response: {}", response.text);
println!("Tokens: {}", response.usage.total_tokens);
println!("Cost: ${:.4}", response.cost_usd);
```

### Using the LLM Orchestrator (Multi-LLM Consensus)
```rust
use prism_ai::orchestration::llm_clients::LLMOrchestrator;
use prism_ai::orchestration::production::config::MissionCharlieConfig;

// Load configuration with API keys from environment
let config = MissionCharlieConfig::from_env()?;

// Create orchestrator
let mut orchestrator = LLMOrchestrator::new(config).await?;

// Query with optimal model selection (bandit algorithm)
let response = orchestrator.query_optimal(
    "What is the capital of France?",
    0.7
).await?;

println!("Best response: {}", response.response.text);
println!("Selected model: {}", response.selected_model);
println!("Confidence: {:.2}%", response.confidence * 100.0);

// Query with consensus (multiple LLMs vote)
let consensus = orchestrator.query_consensus(
    "Explain dark matter",
    0.7
).await?;

println!("Consensus text: {}", consensus.consensus_text);
println!("Participating models: {:?}", consensus.participating_models);
println!("Agreement score: {:.2}%", consensus.agreement_score * 100.0);
```

---

## Validation

### Check Configuration is Valid
```rust
use prism_ai::orchestration::production::config::MissionCharlieConfig;

match MissionCharlieConfig::from_env() {
    Ok(config) => {
        println!("✅ Configuration valid");
        println!("   Enabled LLMs: {:?}", config.enabled_llm_names());
        println!("   Can use consensus: {}", config.can_use_consensus());
        println!("   GPU enabled: {}", config.performance_config.enable_gpu);
    }
    Err(e) => {
        eprintln!("❌ Configuration error: {}", e);
    }
}
```

### Test API Connection
```bash
# Test OpenAI
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"

# Test Anthropic
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{"model":"claude-3-5-sonnet-20250110","max_tokens":1024,"messages":[{"role":"user","content":"Hello"}]}'

# Test Gemini
curl "https://generativelanguage.googleapis.com/v1beta/models?key=$GEMINI_API_KEY"

# Test Grok (xAI)
curl https://api.x.ai/v1/chat/completions \
  -H "Authorization: Bearer $XAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"grok-2-1212","messages":[{"role":"user","content":"Hello"}]}'
```

---

## Configuration Validation Rules

The system enforces these validation rules:

1. **At least 1 LLM must be enabled** - You can disable specific LLMs, but at least one must be active
2. **Enabled LLMs must have API keys** - If enabled=true, api_key must be set
3. **Consensus requires 2+ LLMs** - min_llms_for_consensus must be 2-4
4. **Confidence threshold: 0.0-1.0** - Must be valid probability
5. **Cache similarity: 0.0-1.0** - Must be valid similarity score
6. **Compression level: 0-10** - Higher = more compression

---

## Features Used

### Cargo Dependencies
The following dependencies are already configured in `Cargo.toml`:

```toml
# HTTP client for API calls
reqwest = { version = "0.11", features = ["json"] }

# Rate limiting
governor = "0.6"

# Cost tracking
rust_decimal = "1.33"

# Async trait support
async-trait = "0.1"

# Async utilities
futures = "0.3"
tokio = { version = "1", features = ["full"] }

# Configuration file support
toml = "0.8"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Kolmogorov complexity estimation (for prompt compression)
zstd = "0.13"
```

---

## Architecture Overview

### LLM Client Structure
```
src/orchestration/llm_clients/
├── mod.rs                  - Trait definitions
├── openai_client.rs        - OpenAI GPT-4 client
├── claude_client.rs        - Anthropic Claude client
├── gemini_client.rs        - Google Gemini client
├── grok_client.rs          - xAI Grok client
└── ensemble.rs             - Multi-LLM orchestration
```

### Configuration System
```
src/orchestration/production/
├── config.rs               - MissionCharlieConfig
├── error_handling.rs       - Circuit breakers, graceful degradation
└── logging.rs              - Structured logging
```

### Integration Points
```
src/orchestration/integration/
├── mission_charlie_integration.rs  - Full Mission Charlie system
├── pwsa_llm_bridge.rs              - Sensor fusion + LLM intelligence
└── prism_ai_integration.rs         - PRISM core integration
```

---

## Quick Start Example

```bash
# 1. Set environment variables (minimum required)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# 2. Run PRISM-AI CLI
cargo run --bin prism

# 3. System will automatically:
#    - Load API keys from environment
#    - Initialize enabled LLM clients
#    - Set up consensus mechanisms
#    - Enable GPU acceleration (if available)
```

---

## Troubleshooting

### "At least one LLM must be enabled"
- Ensure at least one API key is set in environment
- Check that corresponding LLM is enabled=true in config

### "OpenAI enabled but no API key provided"
- Set `OPENAI_API_KEY` environment variable
- Or disable OpenAI: `enabled = false` in config

### Rate Limit Errors
- Adjust `rate_limit_rpm` in config (default: 500)
- Enable caching to reduce redundant requests
- Use bandit algorithm for optimal model selection

### Timeout Errors
- Increase `global_timeout_secs` (default: 60)
- Check network connectivity
- Verify API endpoint is accessible

### Cost Concerns
- Enable `enable_cost_tracking = true` to monitor spending
- Use caching to reduce API calls
- Adjust `max_tokens` to limit response size
- Use cheaper models for non-critical queries

---

## Security Best Practices

1. **Never commit API keys to git**
   - Use environment variables
   - Add `.env` to `.gitignore`

2. **Use separate keys for dev/prod**
   - Different keys per environment
   - Easier to rotate and monitor

3. **Rotate keys regularly**
   - Generate new keys periodically
   - Revoke old keys after rotation

4. **Monitor usage**
   - Enable cost tracking
   - Set up alerts for unusual spending
   - Review logs regularly

5. **Restrict key permissions**
   - Use read-only keys where possible
   - Limit rate/usage per key

---

## Additional Resources

- Config file location: `src/orchestration/production/config.rs:14-491`
- LLM client trait: `src/orchestration/llm_clients/mod.rs:29-42`
- Orchestrator: `src/orchestration/llm_clients/ensemble.rs`
- PWSA integration: `src/orchestration/integration/pwsa_llm_bridge.rs:24-81`

---

## Summary

**Minimum Required:**
```bash
export OPENAI_API_KEY="sk-..."        # OR
export ANTHROPIC_API_KEY="sk-ant-..."  # OR
export GEMINI_API_KEY="..."            # OR
export XAI_API_KEY="xai-..."           # (at least one)
```

**Recommended:**
```bash
# All 4 providers for best consensus results
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="..."
export XAI_API_KEY="xai-..."

# Optional tuning
export MISSION_CHARLIE_ENABLE_GPU=true
export MISSION_CHARLIE_LOG_LEVEL=info
export MISSION_CHARLIE_CACHE_SIZE=10000
```

The system will automatically validate configuration on startup and provide clear error messages if anything is missing.
