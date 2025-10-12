# PRISM-AI LLM API Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PRISM-AI Mission Charlie                        │
│                    Quantum-Neuromorphic LLM Fusion                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Configuration Layer                             │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  MissionCharlieConfig (config.rs:14-491)                         │  │
│  │  • Load from .env file                                           │  │
│  │  • Load from mission_charlie.toml                                │  │
│  │  • Programmatic ConfigBuilder                                    │  │
│  │  • Validation & error checking                                   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         LLM Orchestrator                                 │
│                    (ensemble.rs - LLMOrchestrator)                       │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Routing Strategies:                                            │   │
│  │  • Bandit Algorithm (UCB1) - Optimal model selection            │   │
│  │  • Quantum Voting Consensus - Multi-LLM agreement               │   │
│  │  • Transfer Entropy Router - Causal analysis                    │   │
│  │  • Thermodynamic Balancer - Load distribution                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Caching:                                                        │   │
│  │  • Quantum Semantic Cache (LSH hashing)                          │   │
│  │  • Similarity threshold: 0.85                                    │   │
│  │  • Max cache size: 10,000 entries                                │   │
│  │  • TTL: 3600 seconds                                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Error Handling:                                                 │   │
│  │  • Circuit Breaker (5 failures → open)                           │   │
│  │  • Graceful Degradation                                          │   │
│  │  • Heuristic Fallback                                            │   │
│  │  • Retry Logic (max 3 attempts)                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
            ▼                       ▼                       ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│   OpenAI Client     │  │   Claude Client     │  │   Gemini Client     │
│  (openai_client.rs) │  │ (claude_client.rs)  │  │ (gemini_client.rs)  │
│                     │  │                     │  │                     │
│  Model: gpt-4       │  │  Model: claude-3.5  │  │  Model: gemini-2.0  │
│  Quality: 0.80      │  │  Quality: 0.85      │  │  Quality: 0.75      │
│  RPM: 500           │  │  RPM: 500           │  │  RPM: 500           │
└─────────┬───────────┘  └─────────┬───────────┘  └─────────┬───────────┘
          │                        │                        │
          │                        │                        │
          ▼                        ▼                        ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│   Grok Client       │  │   Rate Limiter      │  │   Cost Tracker      │
│  (grok_client.rs)   │  │    (governor)       │  │  (rust_decimal)     │
│                     │  │                     │  │                     │
│  Model: grok-2      │  │  Token bucket       │  │  Track $/request    │
│  Quality: 0.70      │  │  Per-client limits  │  │  Total spend        │
│  RPM: 500           │  │  Backpressure       │  │  Budget alerts      │
└─────────┬───────────┘  └─────────┬───────────┘  └─────────┬───────────┘
          │                        │                        │
          └────────────────────────┼────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         HTTP Client Layer                                │
│                    (reqwest with JSON support)                           │
│                                                                          │
│  Features:                                                               │
│  • TLS encryption                                                        │
│  • Timeout handling (60s default)                                       │
│  • Retry with exponential backoff                                       │
│  • Connection pooling                                                    │
└──────────────────────────────────────────────────────────────────────────┘
                                   │
            ┌──────────────────────┼──────────────────────┐
            │                      │                      │
            ▼                      ▼                      ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│   OpenAI API        │  │   Anthropic API     │  │   Google API        │
│  api.openai.com     │  │  api.anthropic.com  │  │  generative         │
│                     │  │                     │  │  language.google    │
│  Endpoint:          │  │  Endpoint:          │  │  .com               │
│  /v1/chat/          │  │  /v1/messages       │  │                     │
│  completions        │  │                     │  │  Endpoint:          │
│                     │  │  Auth:              │  │  /v1beta/models/    │
│  Auth:              │  │  x-api-key header   │  │  generate           │
│  Bearer token       │  │  anthropic-version  │  │                     │
└─────────────────────┘  └─────────────────────┘  │  Auth:              │
                                                   │  ?key=API_KEY       │
                                                   └─────────────────────┘
            │
            ▼
┌─────────────────────┐
│   xAI API           │
│  api.x.ai           │
│                     │
│  Endpoint:          │
│  /v1/chat/          │
│  completions        │
│                     │
│  Auth:              │
│  Bearer token       │
└─────────────────────┘
```

---

## Data Flow: Query → Response

```
1. User Query
   "What is quantum entanglement?"
        │
        ▼
2. LLMOrchestrator.query_optimal()
   • Check semantic cache (LSH similarity)
   • If cache hit → return cached response (0ms)
   • If cache miss → proceed to model selection
        │
        ▼
3. Model Selection (Bandit Algorithm - UCB1)
   • Calculate upper confidence bound for each LLM
   • UCB = quality + sqrt(2 * ln(total_queries) / model_queries)
   • Select model with highest UCB
   • Example: Claude selected (UCB=0.92)
        │
        ▼
4. Rate Limiting
   • Check token bucket for selected model
   • If limit exceeded → wait or select alternate
   • If available → proceed
        │
        ▼
5. API Request
   • Construct HTTP request with:
     - Model name
     - Temperature (0.7)
     - Max tokens (4096)
     - System/user messages
   • Send POST to api.anthropic.com/v1/messages
   • Include x-api-key header
        │
        ▼
6. Response Processing
   • Parse JSON response
   • Extract text, usage, cost
   • LLMResponse {
       text: "Quantum entanglement is...",
       usage: Usage { prompt: 12, completion: 156, total: 168 },
       cost_usd: 0.0042,
       model: "claude-3-5-sonnet-20250110"
     }
        │
        ▼
7. Quality Update (Bandit Learning)
   • Calculate response quality metrics
   • Update model's quality score
   • quality_new = quality_old * 0.9 + response_quality * 0.1
        │
        ▼
8. Cache Update
   • Store response in semantic cache
   • LSH hash of query → response mapping
   • Set TTL = 3600 seconds
        │
        ▼
9. Cost Tracking
   • Add to total cost counter
   • Log: Model, tokens, cost
   • Check budget alerts
        │
        ▼
10. Return to User
    BanditResponse {
      response: LLMResponse { ... },
      selected_model: "claude",
      confidence: 0.92,
      exploration_factor: 0.15
    }
```

---

## Consensus Query Flow

```
1. User Query (requires high confidence)
   "Should we launch the nuclear deterrent?"
        │
        ▼
2. LLMOrchestrator.query_consensus()
   • Parallel query to ALL enabled LLMs
   • No cache (too critical)
        │
        ├─────────────┬─────────────┬─────────────┐
        │             │             │             │
        ▼             ▼             ▼             ▼
   OpenAI        Claude        Gemini         Grok
   Response      Response      Response      Response
        │             │             │             │
        └─────────────┴─────────────┴─────────────┘
                        │
                        ▼
3. Quantum Voting Consensus
   • Convert text responses to vectors
   • Compute pairwise similarity matrix
   • Calculate quantum coherence
   • Weight by model quality scores
   • Agreement = Σ(similarities × weights) / n
        │
        ▼
4. Consensus Result
   BayesianConsensusResponse {
     consensus_text: "No, absolutely not safe",
     participating_models: ["openai", "claude", "gemini", "grok"],
     agreement_score: 0.95,      // 95% agreement
     individual_confidences: [0.92, 0.97, 0.91, 0.89],
     quantum_coherence: 0.94,
     total_cost_usd: 0.0168      // 4 LLMs queried
   }
        │
        ▼
5. Decision Logic
   if agreement_score > 0.7 {
     // High consensus - trust result
     return consensus_text
   } else {
     // Low consensus - flag for human review
     return "CONSENSUS_FAILED: Human review required"
   }
```

---

## Environment Variable Loading Sequence

```
1. System Environment
   $ export OPENAI_API_KEY="sk-..."
        │
        ▼
2. .env File (if present)
   # Loaded by shell or direnv
   source .env
        │
        ▼
3. MissionCharlieConfig::from_env()
   • Read OPENAI_API_KEY env var
   • Read ANTHROPIC_API_KEY env var
   • Read GEMINI_API_KEY env var
   • Read XAI_API_KEY env var
   • Read MISSION_CHARLIE_* env vars
        │
        ▼
4. Validation
   • At least 1 LLM has api_key?
   • All enabled LLMs have keys?
   • Config values in valid ranges?
        │
        ├─── Valid ────────────▶ Continue
        │
        └─── Invalid ──────────▶ Error with clear message
                                 "Claude enabled but no API key"
```

---

## API Client Implementation

### OpenAI Client
**File:** `src/orchestration/llm_clients/openai_client.rs`

```rust
pub struct OpenAIClient {
    api_key: String,
    model: String,
    client: reqwest::Client,
    total_cost: Arc<Mutex<f64>>,
    total_tokens: Arc<Mutex<usize>>,
}

#[async_trait]
impl LLMClient for OpenAIClient {
    async fn generate(&self, prompt: &str, temperature: f32)
        -> Result<LLMResponse>
    {
        let response = self.client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&json!({
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": 4096
            }))
            .send()
            .await?;

        // Parse and return LLMResponse
    }
}
```

### Claude Client
**File:** `src/orchestration/llm_clients/claude_client.rs`

```rust
pub struct ClaudeClient {
    api_key: String,
    model: String,
    client: reqwest::Client,
    total_cost: Arc<Mutex<f64>>,
    total_tokens: Arc<Mutex<usize>>,
}

#[async_trait]
impl LLMClient for ClaudeClient {
    async fn generate(&self, prompt: &str, temperature: f32)
        -> Result<LLMResponse>
    {
        let response = self.client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&json!({
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": 4096
            }))
            .send()
            .await?;

        // Parse and return LLMResponse
    }
}
```

---

## Integration Points

### PWSA Sensor Fusion + LLM Intelligence
**File:** `src/orchestration/integration/pwsa_llm_bridge.rs:24-81`

```rust
pub struct PwsaLLMFusionPlatform {
    pwsa_platform: Arc<Mutex<PwsaFusionPlatform>>,  // Sensor fusion
    llm_orchestrator: Option<Arc<Mutex<LLMOrchestrator>>>,  // AI analysis
}

impl PwsaLLMFusionPlatform {
    pub async fn fuse_complete_intelligence(&self, sensor_data: &SensorInput)
        -> Result<CompleteIntelligence>
    {
        // Phase 1: Sensor fusion (Mission Bravo)
        let sensor_assessment = self.pwsa_platform.lock()
            .fuse_mission_data(&sensor_data)?;

        // Phase 2: AI intelligence (Mission Charlie)
        let ai_context = if let Some(ref llm_orch) = self.llm_orchestrator {
            let prompt = format!("Analyze threat: {:?}", sensor_assessment);
            let response = llm_orch.lock()
                .query_optimal(&prompt, 0.7)
                .await?;
            Some(response.response.text)
        } else {
            None
        };

        // Combined intelligence
        Ok(CompleteIntelligence {
            sensor_assessment,
            ai_context,
            combined_confidence: 0.95,
        })
    }
}
```

---

## Cost Estimation

### Per-Request Costs (approximate)
| Provider | Model | Input ($/1M tokens) | Output ($/1M tokens) | Avg Cost/Query |
|----------|-------|---------------------|----------------------|----------------|
| OpenAI | GPT-4 | $5.00 | $15.00 | $0.020 |
| Anthropic | Claude 3.5 Sonnet | $3.00 | $15.00 | $0.015 |
| Google | Gemini 2.0 Flash | $0.075 | $0.30 | $0.0004 |
| xAI | Grok-2 | $5.00 | $15.00 | $0.020 |

### Optimization Strategies
1. **Semantic Caching** - 60-80% cache hit rate → 5x cost reduction
2. **Bandit Algorithm** - Routes to cheaper models when quality sufficient
3. **Prompt Compression** - MDL optimization reduces token count by 30%
4. **Selective Consensus** - Only critical queries use all 4 LLMs

**Example Monthly Cost (1000 queries/day):**
- Without optimization: $600/month
- With caching (70% hit): $180/month
- With bandit routing: $120/month
- With compression: $84/month

---

## Security Considerations

### API Key Protection
1. **Never in code** - Always from environment
2. **Never in git** - .env in .gitignore
3. **Rotate regularly** - 90-day rotation policy
4. **Separate keys** - Dev/staging/prod isolation
5. **Monitor usage** - Alert on anomalies

### Request Validation
1. **Rate limiting** - Prevent abuse
2. **Input sanitization** - Prevent injection
3. **Output filtering** - Remove sensitive data
4. **Audit logging** - Track all requests

### Network Security
1. **TLS 1.3** - Encrypted transport
2. **Certificate pinning** - Prevent MITM
3. **Timeout protection** - 60s max
4. **Circuit breakers** - Fail fast on errors

---

## Testing & Validation

### Configuration Validation
```bash
# Test config loading
cargo test --test config_tests

# Validate API keys
cargo run --example validate_config
```

### API Connection Testing
```bash
# Test individual clients
cargo test openai_client_test --features openai
cargo test claude_client_test --features claude

# Test orchestrator
cargo test llm_orchestrator_test --features mission_charlie
```

### Load Testing
```bash
# Stress test with 100 concurrent requests
cargo run --example stress_test_llm --release
```

---

## Monitoring & Observability

### Metrics Tracked
- **Request latency** - P50, P95, P99
- **Success rate** - Per model, per endpoint
- **Cost per query** - Real-time tracking
- **Cache hit rate** - Effectiveness of semantic cache
- **Model quality** - Bandit algorithm scores
- **Consensus agreement** - Multi-LLM alignment

### Logging
```rust
// Structured logging with context
log::info!(
    "LLM request completed";
    "model" => "claude",
    "latency_ms" => 234,
    "tokens" => 168,
    "cost_usd" => 0.0042,
    "cache_hit" => false
);
```

---

## Future Enhancements

1. **Streaming responses** - For real-time UI
2. **Function calling** - Tool use for LLMs
3. **Multimodal support** - Images, audio
4. **Fine-tuned models** - Domain-specific training
5. **Local LLM support** - llama.cpp, vLLM
6. **Cost budgets** - Hard limits per user/day
7. **A/B testing** - Experiment with prompts/models

---

## Quick Reference

### Files
- **Config**: `src/orchestration/production/config.rs:14-491`
- **Orchestrator**: `src/orchestration/llm_clients/ensemble.rs`
- **OpenAI**: `src/orchestration/llm_clients/openai_client.rs`
- **Claude**: `src/orchestration/llm_clients/claude_client.rs`
- **Gemini**: `src/orchestration/llm_clients/gemini_client.rs`
- **Grok**: `src/orchestration/llm_clients/grok_client.rs`

### Commands
```bash
# Setup
cp .env.example .env
# Edit .env with your API keys
source .env

# Build
cargo build --features mission_charlie

# Run
cargo run --bin prism

# Test
cargo test --features mission_charlie
```

### Environment Variables
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
XAI_API_KEY=xai-...
```
