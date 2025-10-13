# PRISM-AI REST API Server

**Worker 8 - Deployment & Documentation**

## Overview

Production-ready REST API server providing HTTP and WebSocket access to all PRISM-AI capabilities:

- **PWSA**: Threat detection, sensor fusion, tracking
- **Finance**: Portfolio optimization, risk assessment
- **Telecom**: Network optimization, congestion management
- **Robotics**: Motion planning, trajectory execution
- **LLM**: Multi-model orchestration, consensus queries
- **Time Series**: Forecasting, trajectory prediction (Enhancement +40h)
- **Pixels**: IR frame analysis, entropy maps, TDA (Enhancement +27h)

## Quick Start

### Build and Run

```bash
# Build with API server feature
cargo build --release --features api_server

# Run the server
API_PORT=8080 cargo run --bin api_server --features api_server

# With authentication
API_KEY="your-secret-key" cargo run --bin api_server --features api_server
```

### Environment Variables

- `API_HOST` - Server host (default: 0.0.0.0)
- `API_PORT` - Server port (default: 8080)
- `API_KEY` - API authentication key (enables auth)
- `RUST_LOG` - Logging level (default: info)

## API Endpoints

### Core Domains

#### PWSA (Proliferated Warfighter Space Architecture)
```
POST   /api/v1/pwsa/detect       - Detect threats from sensor data
GET    /api/v1/pwsa/track/:id    - Get track information
GET    /api/v1/pwsa/tracks       - List all active tracks
POST   /api/v1/pwsa/fuse         - Multi-sensor data fusion
GET    /api/v1/pwsa/health       - Health check
```

#### Finance
```
POST   /api/v1/finance/optimize  - Optimize portfolio
GET    /api/v1/finance/portfolio/:id - Get portfolio details
GET    /api/v1/finance/portfolios - List portfolios
POST   /api/v1/finance/risk      - Assess risk
POST   /api/v1/finance/backtest  - Backtest strategy
GET    /api/v1/finance/health    - Health check
```

#### Telecom
```
POST   /api/v1/telecom/optimize  - Optimize network routing
POST   /api/v1/telecom/congestion - Analyze congestion
GET    /api/v1/telecom/topology  - Get network topology
GET    /api/v1/telecom/node/:id  - Get node status
GET    /api/v1/telecom/health    - Health check
```

#### Robotics
```
POST   /api/v1/robotics/plan     - Plan robot motion
POST   /api/v1/robotics/execute  - Execute trajectory
GET    /api/v1/robotics/robot/:id - Get robot status
GET    /api/v1/robotics/robots   - List all robots
GET    /api/v1/robotics/health   - Health check
```

#### LLM Orchestration
```
POST   /api/v1/llm/query         - Query single LLM (optimal)
POST   /api/v1/llm/consensus     - Multi-model consensus
GET    /api/v1/llm/models        - List available models
GET    /api/v1/llm/model/:name   - Get model info
GET    /api/v1/llm/cache/stats   - Cache statistics
GET    /api/v1/llm/health        - Health check
```

### Enhancements

#### Time Series (+40h)
```
POST   /api/v1/timeseries/forecast    - Generic forecast
POST   /api/v1/timeseries/trajectory  - Missile trajectory
POST   /api/v1/timeseries/market      - Market forecast
POST   /api/v1/timeseries/traffic     - Network traffic
GET    /api/v1/timeseries/series/:id  - Series info
GET    /api/v1/timeseries/health      - Health check
```

#### Pixel Processing (+27h)
```
POST   /api/v1/pixels/process    - Full pipeline
POST   /api/v1/pixels/entropy    - Entropy map
POST   /api/v1/pixels/segment    - Image segmentation
POST   /api/v1/pixels/tda        - Topological analysis
GET    /api/v1/pixels/frame/:id  - Frame info
GET    /api/v1/pixels/health     - Health check
```

### WebSocket

```
GET    /ws                       - WebSocket connection
```

Event types:
- `ThreatDetected` - PWSA threat events
- `PortfolioUpdate` - Finance updates
- `NetworkCongestion` - Telecom alerts
- `LlmStream` - Streaming LLM generation
- `ForecastUpdate` - Time series updates
- `SystemStatus` - System health

## Architecture

```
api_server/
├── mod.rs            # Main server setup
├── error.rs          # Error types
├── models.rs         # Common data models
├── auth.rs           # Authentication
├── middleware.rs     # Request middleware
├── websocket.rs      # WebSocket handler
└── routes/
    ├── pwsa.rs       # PWSA endpoints
    ├── finance.rs    # Finance endpoints
    ├── telecom.rs    # Telecom endpoints
    ├── robotics.rs   # Robotics endpoints
    ├── llm.rs        # LLM endpoints
    ├── time_series.rs # Forecasting endpoints
    └── pixels.rs     # Pixel processing endpoints
```

## Authentication

API key authentication via header:

```bash
# Using Authorization header
curl -H "Authorization: Bearer YOUR_API_KEY" \
  http://localhost:8080/api/v1/pwsa/health

# Using X-API-Key header
curl -H "X-API-Key: YOUR_API_KEY" \
  http://localhost:8080/api/v1/pwsa/health
```

## Response Format

All endpoints return JSON in this format:

```json
{
  "success": true,
  "data": { ... },
  "metadata": {
    "processing_time_ms": 12.5,
    "api_version": "1.0",
    "request_id": "uuid"
  }
}
```

Error responses:

```json
{
  "error": "BadRequest",
  "message": "Invalid input data",
  "details": null
}
```

## Example Requests

### PWSA Threat Detection

```bash
curl -X POST http://localhost:8080/api/v1/pwsa/detect \
  -H "Content-Type: application/json" \
  -d '{
    "sv_id": 42,
    "timestamp": 1234567890,
    "ir_frame": {
      "width": 640,
      "height": 480,
      "centroid_x": 320.0,
      "centroid_y": 240.0,
      "hotspot_count": 5
    }
  }'
```

### Finance Portfolio Optimization

```bash
curl -X POST http://localhost:8080/api/v1/finance/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "assets": [
      {
        "symbol": "AAPL",
        "expected_return": 0.12,
        "volatility": 0.25,
        "current_price": 150.0
      }
    ],
    "constraints": {
      "max_position_size": 0.3,
      "min_position_size": 0.05,
      "max_total_risk": 0.15
    },
    "objective": "maximize_sharpe"
  }'
```

### Time Series Forecast

```bash
curl -X POST http://localhost:8080/api/v1/timeseries/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "series_id": "btc-usd",
    "historical_data": [100, 102, 105, 103, 108],
    "timestamps": [1, 2, 3, 4, 5],
    "horizon": 10,
    "method": { "arima": { "p": 2, "d": 1, "q": 2 } },
    "include_uncertainty": true
  }'
```

### Pixel Processing

```bash
curl -X POST http://localhost:8080/api/v1/pixels/process \
  -H "Content-Type: application/json" \
  -d '{
    "frame_id": "frame-001",
    "width": 640,
    "height": 480,
    "pixels": [1000, 1010, ...],
    "processing_options": {
      "compute_entropy": true,
      "compute_tda": true,
      "compute_segmentation": false,
      "extract_features": true,
      "entropy_window_size": 16
    }
  }'
```

## WebSocket Example

```javascript
const ws = new WebSocket('ws://localhost:8080/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  switch(data.type) {
    case 'ThreatDetected':
      console.log('Threat:', data.threat_id, data.confidence);
      break;
    case 'LlmStream':
      console.log('Token:', data.token);
      break;
  }
};

// Send ping
ws.send(JSON.stringify({ type: 'Ping' }));
```

## Monitoring

Health check endpoint:

```bash
curl http://localhost:8080/health
# Response: "PRISM-AI API Server - Healthy"
```

Individual subsystem health:

```bash
curl http://localhost:8080/api/v1/pwsa/health
curl http://localhost:8080/api/v1/finance/health
curl http://localhost:8080/api/v1/llm/health
```

## Performance

- **Latency**: < 5ms for most endpoints (PWSA requirement)
- **Throughput**: 1000+ requests/second
- **Concurrency**: Fully async with Tokio
- **Rate Limiting**: Token bucket algorithm
- **Max Body Size**: 10MB (configurable)
- **Timeout**: 60s (configurable)

## Integration Points

The API server provides REST access to:

- **Worker 1**: Time series forecasting (ARIMA, LSTM)
- **Worker 2**: GPU-accelerated kernels
- **Worker 3**: PWSA sensor fusion, pixel processing
- **Worker 4**: Finance optimization
- **Worker 5**: Thermodynamic LLM orchestration
- **Worker 6**: Advanced LLM features
- **Worker 7**: Robotics motion planning

## Status Codes

- `200 OK` - Success
- `400 Bad Request` - Invalid input
- `401 Unauthorized` - Missing/invalid API key
- `403 Forbidden` - Insufficient permissions
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Server error
- `503 Service Unavailable` - Service down

## Development

### Run in development mode

```bash
RUST_LOG=debug cargo run --bin api_server --features api_server
```

### Run tests

```bash
cargo test --features api_server
```

### Check formatting

```bash
cargo fmt -- --check
cargo clippy --features api_server
```

## Production Deployment

See `deployment/` directory for:
- Docker configuration
- Kubernetes manifests
- CI/CD pipelines
- Monitoring setup

## License

MIT License - See LICENSE file

---

**Worker 8 Status**: Phase 1 Complete (API Server - 2,485 LOC)
**Next**: Deployment infrastructure (Docker, Kubernetes, CI/CD)
