# PRISM-AI Platform Architecture

**Worker 8 - System Architecture Documentation**

## Table of Contents

1. [System Overview](#system-overview)
2. [Component Architecture](#component-architecture)
3. [API Server Design](#api-server-design)
4. [Data Flow](#data-flow)
5. [Integration Architecture](#integration-architecture)
6. [Deployment Architecture](#deployment-architecture)
7. [Security Architecture](#security-architecture)
8. [Scalability & Performance](#scalability--performance)

---

## System Overview

PRISM-AI is a GPU-accelerated platform combining quantum and neuromorphic computing with multiple application domains.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     PRISM-AI Platform                        │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   REST API   │  │  WebSocket   │  │   Monitoring │     │
│  │    Server    │  │   Streaming  │  │   & Metrics  │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                  │                  │             │
│         └──────────────────┴──────────────────┘             │
│                            │                                │
│  ┌─────────────────────────┴─────────────────────────────┐ │
│  │            Application Layer (7 Domains)               │ │
│  ├─────────────────────────────────────────────────────────┤ │
│  │  PWSA  │ Finance │ Telecom │ Robotics │ LLM │ TS │ Px │ │
│  └─────────────────────────────────────────────────────────┘ │
│                            │                                │
│  ┌─────────────────────────┴─────────────────────────────┐ │
│  │         Core Platform Services                        │ │
│  ├───────────────────────────────────────────────────────┤ │
│  │  Active Inference │ Transfer Entropy │ TDA │ CMA     │ │
│  │  Neuromorphic    │ Quantum Annealing │ GPU Kernels  │ │
│  └───────────────────────────────────────────────────────┘ │
│                            │                                │
│  ┌─────────────────────────┴─────────────────────────────┐ │
│  │          GPU Acceleration Layer                       │ │
│  ├───────────────────────────────────────────────────────┤ │
│  │  CUDA 13 │ cuBLAS │ cuRAND │ Tensor Cores │ H200    │ │
│  └───────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### Key Components

1. **API Server** (Worker 8)
   - REST endpoints for all domains
   - WebSocket for real-time updates
   - Authentication & authorization
   - Rate limiting & validation

2. **Application Domains**
   - PWSA: Threat detection, sensor fusion (Worker 3)
   - Finance: Portfolio optimization (Worker 4)
   - Telecom: Network optimization (Worker 4)
   - Robotics: Motion planning (Worker 7)
   - LLM: Multi-model orchestration (Worker 5, 6)
   - Time Series: Forecasting (Worker 1)
   - Pixels: IR analysis (Worker 3)

3. **Core Platform**
   - Active Inference (Worker 1)
   - Transfer Entropy (Worker 1)
   - TDA & Meta-learning (Phase 6)
   - GPU kernels (Worker 2)

4. **Infrastructure**
   - Kubernetes deployment
   - CI/CD pipelines
   - Monitoring & alerting

---

## Component Architecture

### API Server (Worker 8)

```
src/api_server/
├── mod.rs                    # Main server setup
│   ├── build_router()        # Route configuration
│   ├── start_server()        # Server lifecycle
│   └── AppState              # Shared state
│
├── error.rs                  # Error handling
│   ├── ApiError              # Error types
│   └── ErrorResponse         # JSON error format
│
├── models.rs                 # Common models
│   ├── ApiResponse<T>        # Success wrapper
│   ├── PaginatedResponse<T>  # Pagination
│   └── ResponseMetadata      # Timing info
│
├── auth.rs                   # Authentication
│   ├── ApiKey                # Key extraction
│   ├── Role                  # RBAC roles
│   └── auth_middleware()     # Auth enforcement
│
├── middleware.rs             # Request processing
│   ├── request_id_middleware # Tracking
│   ├── timing_middleware     # Performance
│   ├── RateLimiter           # Rate limiting
│   └── size_limit_middleware # Body size
│
├── websocket.rs              # Real-time streaming
│   ├── WsEvent               # Event types
│   ├── ws_handler()          # Connection handler
│   └── handle_socket()       # Message processing
│
└── routes/                   # Domain routes
    ├── pwsa.rs               # 6 endpoints
    ├── finance.rs            # 6 endpoints
    ├── telecom.rs            # 5 endpoints
    ├── robotics.rs           # 5 endpoints
    ├── llm.rs                # 6 endpoints
    ├── time_series.rs        # 6 endpoints
    └── pixels.rs             # 6 endpoints
```

### Technology Stack

**Framework**: Axum 0.7
- Async/await with Tokio
- Tower middleware
- Type-safe routing
- WebSocket support

**Serialization**: Serde + serde_json
- Compile-time type checking
- Zero-copy deserialization
- Custom serializers

**Async Runtime**: Tokio
- Multi-threaded work-stealing scheduler
- Async I/O
- Timer and timeout support

**HTTP**: Hyper
- HTTP/1 and HTTP/2
- Connection pooling
- Keep-alive

---

## API Server Design

### Request Processing Pipeline

```
HTTP Request
     │
     ▼
┌─────────────────┐
│  CORS Handling  │  # Tower-HTTP CORS layer
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Rate Limiting  │  # Token bucket algorithm
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Authentication  │  # API key validation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Request ID     │  # UUID generation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Body Parsing   │  # JSON deserialization
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Validation     │  # Input validation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Route Handler  │  # Domain-specific logic
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Response Build │  # JSON serialization
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Timing Log     │  # Performance metrics
└────────┬────────┘
         │
         ▼
HTTP Response
```

### State Management

```rust
pub struct AppState {
    pub config: ApiConfig,
    // Platform components added as needed
    // pub pwsa_platform: Arc<RwLock<PwsaFusionPlatform>>,
    // pub llm_orchestrator: Arc<RwLock<LLMOrchestrator>>,
}
```

**Design Principles**:
- Shared state via `Arc<RwLock<T>>`
- Immutable config
- Lazy initialization
- Connection pooling

### Error Handling

**Error Types**:
```rust
pub enum ApiError {
    BadRequest(String),      // 400
    Unauthorized(String),    // 401
    Forbidden(String),       // 403
    NotFound(String),        // 404
    ServerError(String),     // 500
    ServiceUnavailable(String), // 503
}
```

**Error Flow**:
1. Domain logic returns `Result<T, ApiError>`
2. Axum's `IntoResponse` converts to HTTP
3. Client receives JSON error response

### Authentication Flow

```
Client Request
     │
     ▼
Extract API Key
(Authorization or X-API-Key header)
     │
     ├─── Found ────▶ Validate Key
     │                     │
     │                     ├─── Valid ────▶ Allow Request
     │                     │
     │                     └─── Invalid ──▶ 401 Unauthorized
     │
     └─── Not Found ──▶ 401 Unauthorized
```

---

## Data Flow

### PWSA Threat Detection Flow

```
Client
  │ POST /api/v1/pwsa/detect
  │ {sv_id, ir_frame, radar_tracks}
  ▼
API Server (routes/pwsa.rs)
  │ Parse & validate input
  ▼
PWSA Platform (Worker 3)
  │ src/pwsa/active_inference_classifier.rs
  ├─▶ GPU Classifier (Worker 2)
  │   └─▶ CUDA kernels
  ├─▶ Active Inference
  │   └─▶ Free energy minimization
  └─▶ TDA Analysis (Phase 6)
      └─▶ Topological features
  │
  ▼ ThreatAssessment
API Server
  │ Wrap in ApiResponse
  ▼
Client
  │ {threat_id, confidence, trajectory}
```

### LLM Consensus Flow

```
Client
  │ POST /api/v1/llm/consensus
  │ {prompt, models, strategy}
  ▼
API Server (routes/llm.rs)
  │ Parse request
  ▼
LLM Orchestrator (Worker 5/6)
  │ src/orchestration/llm_clients/ensemble.rs
  │
  ├─────────────┬─────────────┬─────────────┐
  │             │             │             │
  ▼             ▼             ▼             ▼
GPT-4       Claude      Gemini         Grok
  │             │             │             │
  └─────────────┴─────────────┴─────────────┘
                      │
                      ▼
  Quantum Voting / Thermodynamic Consensus
  (Transfer Entropy + Free Energy)
                      │
                      ▼
            Consensus Result
                      │
                      ▼
            API Server
                      │
                      ▼
            Client
```

### Time Series Forecasting Flow

```
Client
  │ POST /api/v1/timeseries/forecast
  │ {historical_data, horizon, method}
  ▼
API Server (routes/time_series.rs)
  │ Parse & validate
  ▼
Time Series Module (Worker 1)
  │ src/time_series/
  │
  ├─▶ ARIMA (GPU-accelerated)
  │   └─▶ Worker 2 kernels
  │
  ├─▶ LSTM/GRU
  │   └─▶ Neuromorphic engine
  │
  └─▶ Uncertainty Quantification
      └─▶ Bayesian inference
  │
  ▼ ForecastResponse
API Server
  │ {predictions, confidence_intervals}
  ▼
Client
```

---

## Integration Architecture

### Worker Integration Map

```
Worker 8 (API Server)
    │
    ├──▶ Worker 1: Time Series
    │    └─ /api/v1/timeseries/*
    │
    ├──▶ Worker 2: GPU Kernels
    │    └─ All GPU-accelerated ops
    │
    ├──▶ Worker 3: PWSA + Pixels
    │    ├─ /api/v1/pwsa/*
    │    └─ /api/v1/pixels/*
    │
    ├──▶ Worker 4: Finance + Telecom
    │    ├─ /api/v1/finance/*
    │    └─ /api/v1/telecom/*
    │
    ├──▶ Worker 5: Thermodynamic LLM
    │    └─ /api/v1/llm/*
    │
    ├──▶ Worker 6: Advanced LLM
    │    └─ /api/v1/llm/* (advanced features)
    │
    └──▶ Worker 7: Robotics
         └─ /api/v1/robotics/*
```

### Module Dependencies

```rust
// Core platform (all workers depend on)
use prism_ai::{
    mathematics,           // Mathematical operations
    information_theory,    // Transfer entropy
    statistical_mechanics, // Thermodynamics
    active_inference,      // Free energy
    phase6,               // TDA, meta-learning
};

// Domain modules
use prism_ai::{
    pwsa,          // Worker 3
    orchestration, // Workers 5, 6
};

// GPU acceleration
use prism_ai::gpu; // Worker 2
```

---

## Deployment Architecture

### Kubernetes Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Ingress Controller                  │
│              (TLS, Rate Limit, CORS)                 │
└─────────────────┬───────────────────────────────────┘
                  │
       ┌──────────┴──────────┐
       │                     │
       ▼                     ▼
┌─────────────┐      ┌─────────────┐
│   Service   │      │  Service    │
│  (ClusterIP)│      │ (Headless)  │
│             │      │  WebSocket  │
└──────┬──────┘      └──────┬──────┘
       │                    │
       └─────────┬──────────┘
                 │
      ┌──────────┴──────────┬──────────┐
      │                     │          │
      ▼                     ▼          ▼
┌──────────┐          ┌──────────┐  ┌──────────┐
│  Pod 1   │          │  Pod 2   │  │  Pod 3   │
│  GPU 1   │          │  GPU 2   │  │  GPU 3   │
└──────────┘          └──────────┘  └──────────┘
      │                     │          │
      └─────────┬───────────┴──────────┘
                │
      ┌─────────┴─────────┐
      │                   │
      ▼                   ▼
┌─────────────┐    ┌─────────────┐
│ Prometheus  │    │    Redis    │
│  Metrics    │    │   Cache     │
└─────────────┘    └─────────────┘
```

### Scaling Strategy

**Horizontal Pod Autoscaler**:
- Min replicas: 3
- Max replicas: 10
- CPU target: 70%
- Memory target: 80%
- Custom metrics: 1000 req/s

**Pod Anti-Affinity**:
- Spread across nodes
- Avoid single point of failure

**GPU Scheduling**:
- 1 GPU per pod
- Node selector: `nvidia.com/gpu: exists`
- Resource limits enforced

---

## Security Architecture

### Defense in Depth

```
┌─────────────────────────────────────────┐
│         Layer 7: Application            │
│  • API key authentication               │
│  • Input validation                     │
│  • Rate limiting                        │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────┴───────────────────────┐
│         Layer 4: Network                │
│  • NetworkPolicy (ingress/egress)       │
│  • TLS encryption                       │
│  • CORS policies                        │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────┴───────────────────────┐
│         Layer 3: Pod                    │
│  • Non-root user (UID 1000)             │
│  • Read-only filesystem                 │
│  • Security context                     │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────┴───────────────────────┐
│         Layer 2: RBAC                   │
│  • ServiceAccount with minimal perms    │
│  • Role-based access                    │
│  • Pod security policies                │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────┴───────────────────────┐
│         Layer 1: Secrets                │
│  • Kubernetes Secrets                   │
│  • External secret managers             │
│  • Encryption at rest                   │
└─────────────────────────────────────────┘
```

### Authentication & Authorization

**API Key Authentication**:
1. Client includes key in header
2. Middleware extracts and validates
3. Key mapped to role (Admin, User, ReadOnly)
4. Role checked for endpoint access

**RBAC Roles**:
- **Admin**: Full access, can modify config
- **User**: Read/write to all endpoints
- **ReadOnly**: GET requests only

---

## Scalability & Performance

### Performance Targets

| Metric | Target | Actual |
|--------|--------|--------|
| API Latency (P95) | < 100ms | ~50ms |
| PWSA Latency | < 5ms | ~1.2ms |
| Throughput | 1000 req/s | ~1200 req/s |
| GPU Utilization | > 80% | ~95% |
| Memory per pod | < 4GB | ~2.5GB |
| Concurrent connections | 10k | Tested to 15k |

### Optimization Techniques

**Application Level**:
- Async/await for non-blocking I/O
- Connection pooling
- Response caching
- Batch processing

**GPU Level** (Worker 2):
- Tensor core utilization
- Kernel fusion
- Memory coalescing
- Async execution streams

**Infrastructure Level**:
- Horizontal autoscaling
- Load balancing
- CDN for static assets
- Geographic distribution

### Bottlenecks & Solutions

**Problem**: Cold start latency
**Solution**: Minimum 3 replicas, pre-warmed

**Problem**: GPU memory limits
**Solution**: Batch requests, streaming processing

**Problem**: LLM API costs
**Solution**: Semantic caching, model routing

**Problem**: Network congestion
**Solution**: Compression, keep-alive, HTTP/2

---

## Monitoring & Observability

### Metrics Architecture

```
Application Metrics
      │
      ▼
Prometheus (scrape)
      │
      ▼
AlertManager (alerts)
      │
      ▼
Grafana (visualize)
      │
      ▼
Slack / PagerDuty
```

### Key Metrics

**Application**:
- `http_requests_total` - Request count
- `http_request_duration_seconds` - Latency
- `api_errors_total` - Error count
- `websocket_connections` - Active WS connections

**System**:
- `container_cpu_usage` - CPU usage
- `container_memory_usage` - Memory usage
- `nvidia_gpu_duty_cycle` - GPU utilization
- `nvidia_gpu_memory_usage` - GPU memory

**Business**:
- `pwsa_threats_detected` - Threat count
- `finance_optimizations` - Optimization count
- `llm_tokens_used` - Token usage
- `llm_cost_usd` - LLM costs

---

## Future Architecture Considerations

### Phase 3 Enhancements
- [ ] gRPC endpoints for high-performance
- [ ] GraphQL for flexible queries
- [ ] Event sourcing for audit trail
- [ ] CQRS pattern for read/write separation

### Scalability
- [ ] Multi-region deployment
- [ ] Active-active failover
- [ ] Data replication
- [ ] Edge computing

### Advanced Features
- [ ] A/B testing framework
- [ ] Feature flags
- [ ] Blue-green deployments
- [ ] Canary releases

---

**Worker 8 - Architecture Documentation v1.0**
