# PRISM-AI Dual API Deployment Guide

**Worker 8 - Phase 2 Deliverable**
**Date**: October 13, 2025
**API Version**: 1.0
**Status**: Production-Ready

---

## Overview

This guide covers deployment of the PRISM-AI Dual API Server, which provides both REST and GraphQL interfaces for all PRISM-AI capabilities:

- **REST API**: Traditional HTTP endpoints at `/api/v1/*`
- **GraphQL API**: Flexible query interface at `/graphql`
- **Dual Access**: Single server, both APIs simultaneously
- **Worker Coverage**: APIs for Workers 1-8 capabilities

---

## Architecture

```
PRISM-AI Dual API Server (Port 8080)
├── REST API
│   ├── /api/v1/pwsa              - PWSA threat detection (Worker 1)
│   ├── /api/v1/finance           - Basic finance (Worker 3)
│   ├── /api/v1/finance_advanced  - Advanced finance (Worker 4)
│   ├── /api/v1/applications      - Application domains (Worker 3)
│   ├── /api/v1/telecom           - Telecom optimization
│   ├── /api/v1/robotics          - Motion planning (Worker 7)
│   ├── /api/v1/llm               - LLM orchestration (Worker 6)
│   ├── /api/v1/timeseries        - Time series forecasting (Worker 1)
│   ├── /api/v1/pixels            - Pixel-level IR processing
│   └── /api/v1/gpu               - GPU monitoring (Worker 2)
│
├── GraphQL API
│   ├── /graphql                  - GraphQL endpoint
│   ├── /graphql/schema           - Schema introspection
│   └── /graphql (browser)        - GraphQL Playground UI
│
└── Utilities
    ├── /health                   - Health check
    ├── /                         - API info
    └── /ws                       - WebSocket streaming
```

---

## Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS, Windows
- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: NVIDIA GPU with CUDA 12.0+ (optional, for GPU acceleration)
- **Rust**: 1.70+ (stable)

### Dependencies
```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# System dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y build-essential pkg-config libssl-dev

# CUDA Toolkit (optional, for GPU features)
# Follow instructions at: https://developer.nvidia.com/cuda-downloads
```

---

## Deployment Options

### Option 1: Development Mode (Quick Start)

**For local testing and development:**

```bash
# Navigate to source code directory
cd 03-Source-Code

# Run in development mode
cargo run --bin api_server

# Server starts on http://localhost:8080
# GraphQL Playground: http://localhost:8080/graphql
```

**Features enabled:**
- Hot reloading (with `cargo watch`)
- Debug logging
- CORS enabled (permissive)
- No authentication

---

### Option 2: Production Mode (Recommended)

**For production deployment with optimizations:**

```bash
# Build optimized release binary
cd 03-Source-Code
cargo build --release --bin api_server

# Binary location: target/release/api_server

# Run with production config
./target/release/api_server --config config/production.toml
```

**Features enabled:**
- Compiler optimizations (-O3)
- GPU acceleration (if available)
- Authentication/authorization
- Rate limiting
- Request logging
- Performance monitoring

---

### Option 3: Docker Deployment

**For containerized deployment:**

```bash
# Build Docker image
cd 03-Source-Code
docker build -t prism-ai-api:latest -f Dockerfile .

# Run container
docker run -d \
  --name prism-ai-api \
  -p 8080:8080 \
  --gpus all \  # Optional: Enable GPU access
  -v $(pwd)/config:/app/config \
  prism-ai-api:latest

# Check logs
docker logs -f prism-ai-api
```

---

## Configuration

### Default Configuration

```toml
# config/default.toml
[api_server]
host = "0.0.0.0"
port = 8080
cors_enabled = true
auth_enabled = false  # Disable for development
max_body_size_mb = 10
timeout_secs = 60

[gpu]
enable_gpu = true
fallback_to_cpu = true
gpu_device_id = 0

[logging]
level = "info"  # Options: trace, debug, info, warn, error
format = "json"  # Options: json, pretty
```

### Production Configuration

```toml
# config/production.toml
[api_server]
host = "0.0.0.0"
port = 8080
cors_enabled = true
auth_enabled = true  # Enable authentication
api_key = "${PRISM_API_KEY}"  # From environment variable
max_body_size_mb = 10
timeout_secs = 60

[rate_limiting]
enabled = true
requests_per_minute = 100
burst_size = 20

[gpu]
enable_gpu = true
fallback_to_cpu = true
gpu_device_id = 0

[logging]
level = "info"
format = "json"
output = "/var/log/prism-ai/api.log"
```

### Environment Variables

```bash
# Required for production
export PRISM_API_KEY="your-secure-api-key-here"

# Optional: GPU configuration
export CUDA_VISIBLE_DEVICES=0

# Optional: Logging configuration
export RUST_LOG=info

# Optional: Performance tuning
export RAYON_NUM_THREADS=8
```

---

## Testing the Deployment

### 1. Health Check

```bash
curl http://localhost:8080/health
# Expected: "PRISM-AI API Server - Healthy"
```

### 2. REST API Test

```bash
# Time series forecast
curl -X POST http://localhost:8080/api/v1/timeseries/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "data": [100.0, 102.0, 104.0, 106.0],
    "horizon": 5,
    "method": "ARIMA"
  }'

# Advanced portfolio optimization
curl -X POST http://localhost:8080/api/v1/finance_advanced/optimize_advanced \
  -H "Content-Type: application/json" \
  -d '{
    "assets": [
      {"symbol": "AAPL", "expected_return": 0.12, "volatility": 0.20},
      {"symbol": "GOOGL", "expected_return": 0.15, "volatility": 0.25}
    ],
    "strategy": "maximize_sharpe",
    "use_gnn": false,
    "risk_free_rate": 0.03
  }'

# Healthcare risk prediction
curl -X POST http://localhost:8080/api/v1/applications/healthcare/predict_risk \
  -H "Content-Type: application/json" \
  -d '{
    "historical_metrics": [0.2, 0.25, 0.3, 0.28, 0.32],
    "horizon": 5,
    "risk_factors": ["age", "bmi"]
  }'
```

### 3. GraphQL API Test

**Using GraphQL Playground:**
1. Open browser: http://localhost:8080/graphql
2. Try example queries in the playground

**Using curl:**

```bash
# Health query
curl -X POST http://localhost:8080/graphql \
  -H "Content-Type: application/json" \
  -d '{
    "query": "query { health { status version uptimeSeconds } }"
  }'

# Time series forecast
curl -X POST http://localhost:8080/graphql \
  -H "Content-Type: application/json" \
  -d '{
    "query": "query { forecastTimeSeries(input: { historicalData: [100.0, 102.0, 104.0], horizon: 5, method: \"ARIMA\" }) { predictions method horizon } }"
  }'

# Portfolio optimization
curl -X POST http://localhost:8080/graphql \
  -H "Content-Type: application/json" \
  -d '{
    "query": "query { optimizePortfolio(input: { assets: [{ symbol: \"AAPL\", expectedReturn: 0.12, volatility: 0.20 }], objective: \"MaximizeSharpe\" }) { weights { symbol weight } expectedReturn sharpeRatio } }"
  }'
```

### 4. Run Integration Test Suite

```bash
# Run all API integration tests
cd 03-Source-Code
cargo test --test '*' --features api_server

# Run specific test
cargo test test_healthcare_risk_prediction

# Run with output
cargo test -- --nocapture
```

### 5. Run GraphQL Test Script

```bash
# Make test script executable
chmod +x test_graphql_api.sh

# Run comprehensive GraphQL tests
./test_graphql_api.sh
```

---

## Performance Benchmarking

### Load Testing

```bash
# Install Apache Bench (if not already installed)
sudo apt-get install apache2-utils

# Simple load test (1000 requests, 10 concurrent)
ab -n 1000 -c 10 \
  -p payload.json \
  -T application/json \
  http://localhost:8080/api/v1/timeseries/forecast

# Expected results:
# - Requests per second: 100-500 (depending on hardware)
# - Mean response time: 10-50ms (CPU only)
# - Mean response time: 1-10ms (with GPU acceleration)
```

### GPU Performance Validation

```bash
# Run GPU-accelerated time series forecast
curl -X POST http://localhost:8080/api/v1/timeseries/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "data": [100.0, 102.0, 104.0, 106.0, 108.0],
    "horizon": 50,
    "method": "LSTM",
    "use_gpu": true
  }'

# Expected speedup (GPU vs CPU):
# - ARIMA: 15-25x
# - LSTM: 50-100x (with Tensor Cores)
# - GNN: 10-50x
```

---

## Monitoring & Observability

### Health Monitoring

```bash
# Health check endpoint
curl http://localhost:8080/health

# GPU status
curl http://localhost:8080/api/v1/gpu/status

# Performance metrics
curl http://localhost:8080/graphql \
  -H "Content-Type: application/json" \
  -d '{"query": "query { performanceMetrics { endpoint avgResponseTimeMs requestsPerSecond errorRate } }"}'
```

### Logging

```bash
# View logs (development)
cargo run --bin api_server 2>&1 | tee api.log

# View logs (production with systemd)
sudo journalctl -u prism-ai-api -f

# View logs (Docker)
docker logs -f prism-ai-api
```

### Metrics Collection

**Prometheus metrics available at** `/metrics` (when enabled):
- Request count by endpoint
- Response time percentiles (p50, p95, p99)
- Error rate
- GPU utilization
- Memory usage

---

## Troubleshooting

### Issue: Port 8080 Already in Use

```bash
# Find process using port 8080
sudo lsof -i :8080

# Kill process
sudo kill -9 <PID>

# Or use different port
./api_server --port 8081
```

### Issue: GPU Not Detected

```bash
# Check GPU availability
nvidia-smi

# Verify CUDA installation
nvcc --version

# Check GPU visibility
echo $CUDA_VISIBLE_DEVICES

# Test GPU in Rust
cargo test test_gpu_availability -- --nocapture
```

### Issue: Compilation Errors

```bash
# Update Rust toolchain
rustup update stable

# Clean build cache
cargo clean

# Rebuild
cargo build --release
```

### Issue: High Memory Usage

```bash
# Monitor memory
htop

# Reduce max_body_size in config
# Reduce concurrent requests
# Enable request streaming
```

---

## Security Considerations

### Authentication

```rust
// Enable API key authentication
auth_enabled = true
api_key = "your-secure-key"

// Add API key to requests
curl -H "Authorization: Bearer your-secure-key" \
  http://localhost:8080/api/v1/timeseries/forecast
```

### HTTPS/TLS

```bash
# Generate self-signed certificate (development)
openssl req -x509 -newkey rsa:4096 \
  -keyout key.pem -out cert.pem -days 365 -nodes

# Configure TLS in config.toml
[tls]
enabled = true
cert_path = "cert.pem"
key_path = "key.pem"
```

### Rate Limiting

```toml
[rate_limiting]
enabled = true
requests_per_minute = 100
burst_size = 20
```

---

## Production Checklist

- [ ] Built in release mode (`--release`)
- [ ] Authentication enabled (`auth_enabled = true`)
- [ ] API key configured securely (environment variable)
- [ ] Rate limiting enabled
- [ ] HTTPS/TLS configured (for external access)
- [ ] Logging configured (JSON format, persistent storage)
- [ ] Health check endpoint verified
- [ ] GPU acceleration tested (if applicable)
- [ ] Load testing completed
- [ ] Monitoring configured (Prometheus/Grafana)
- [ ] Backup/restore procedures documented
- [ ] Firewall rules configured
- [ ] Service manager configured (systemd/Docker)

---

## Support & Resources

### Documentation
- REST API Reference: `docs/REST_API_REFERENCE.md`
- GraphQL Schema: http://localhost:8080/graphql/schema
- Dual API Guide: `docs/DUAL_API_GUIDE.md`

### Testing
- Integration Tests: `tests/dual_api_integration.rs`
- GraphQL Test Script: `test_graphql_api.sh`
- Test Queries: `graphql_test_queries.json`

### Contact
- **Integration Lead**: Worker 8
- **QA Lead**: Worker 7
- **Strategic Oversight**: Worker 0-Alpha

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Oct 13, 2025 | Initial production release |
| - | - | - Dual API (REST + GraphQL) |
| - | - | - Worker 3/4 application APIs |
| - | - | - GPU acceleration support |

---

**Deployment Status**: ✅ Production-Ready
**Last Updated**: October 13, 2025
**Maintained By**: Worker 8 (Integration Lead)
