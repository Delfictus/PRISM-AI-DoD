# PRISM-AI REST API Documentation

**Version**: 1.0.0
**Base URL**: `http://localhost:8080` (development) or `https://api.prism-ai.example.com` (production)
**Authentication**: API Key via `Authorization: Bearer <token>` or `X-API-Key: <token>`

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Common Patterns](#common-patterns)
4. [Endpoints by Domain](#endpoints-by-domain)
   - [PWSA](#pwsa-endpoints)
   - [Finance](#finance-endpoints)
   - [Telecom](#telecom-endpoints)
   - [Robotics](#robotics-endpoints)
   - [LLM](#llm-endpoints)
   - [Time Series](#time-series-endpoints)
   - [Pixels](#pixel-processing-endpoints)
5. [WebSocket](#websocket)
6. [Error Handling](#error-handling)
7. [Rate Limiting](#rate-limiting)
8. [Examples](#examples)

---

## Overview

The PRISM-AI REST API provides programmatic access to all platform capabilities:

- **PWSA**: Missile threat detection, sensor fusion, tracking
- **Finance**: Portfolio optimization, risk assessment, backtesting
- **Telecom**: Network optimization, congestion management
- **Robotics**: Motion planning, trajectory execution
- **LLM**: Multi-model orchestration, consensus queries
- **Time Series**: ARIMA/LSTM forecasting, trajectory prediction
- **Pixels**: IR frame analysis, entropy maps, TDA features

### Key Features

- ✅ RESTful design with JSON
- ✅ Async/non-blocking operations
- ✅ WebSocket for real-time updates
- ✅ Comprehensive error responses
- ✅ Request/response validation
- ✅ Rate limiting
- ✅ CORS support

---

## Authentication

All API requests require authentication via API key.

### Methods

#### Bearer Token (Recommended)
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
  https://api.prism-ai.example.com/api/v1/pwsa/health
```

#### X-API-Key Header
```bash
curl -H "X-API-Key: YOUR_API_KEY" \
  https://api.prism-ai.example.com/api/v1/pwsa/health
```

### Obtaining API Key

Contact your administrator or set via environment variable:
```bash
export API_KEY="your-secret-key"
```

---

## Common Patterns

### Response Format

All successful responses follow this structure:

```json
{
  "success": true,
  "data": { /* response data */ },
  "metadata": {
    "processing_time_ms": 12.5,
    "api_version": "1.0",
    "request_id": "uuid-here"
  }
}
```

### Error Format

Error responses:

```json
{
  "error": "BadRequest",
  "message": "Invalid input data",
  "details": "Field 'sv_id' is required"
}
```

### HTTP Status Codes

- `200 OK` - Success
- `400 Bad Request` - Invalid input
- `401 Unauthorized` - Missing/invalid API key
- `403 Forbidden` - Insufficient permissions
- `404 Not Found` - Resource not found
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error
- `503 Service Unavailable` - Service down

---

## PWSA Endpoints

Proliferated Warfighter Space Architecture - Threat detection and sensor fusion.

### POST /api/v1/pwsa/detect

Detect threats from sensor data.

**Request Body:**
```json
{
  "sv_id": 42,
  "timestamp": 1234567890,
  "ir_frame": {
    "width": 640,
    "height": 480,
    "centroid_x": 320.0,
    "centroid_y": 240.0,
    "hotspot_count": 5,
    "pixels": [1000, 1010, ...]  // Optional
  },
  "radar_tracks": [  // Optional
    {
      "track_id": "track-001",
      "position": [100.0, 200.0, 250.0],
      "velocity": [1500.0, 200.0, -50.0],
      "rcs": 0.5
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "threat_id": "threat-uuid",
    "threat_type": "ballistic_missile",
    "confidence": 0.92,
    "position": [320.0, 240.0, 250.0],
    "velocity": [1500.0, 200.0, -50.0],
    "estimated_trajectory": [
      [100.0, 100.0, 250.0],
      [200.0, 150.0, 200.0],
      [300.0, 200.0, 150.0]
    ],
    "time_to_impact": 120.0,
    "recommended_action": "Activate defense system"
  }
}
```

### GET /api/v1/pwsa/track/:track_id

Get specific track information.

**Parameters:**
- `track_id` (path) - Track identifier

**Response:**
```json
{
  "success": true,
  "data": {
    "track_id": "track-001",
    "last_updated": 1234567890,
    "position": [150.0, 200.0, 180.0],
    "velocity": [1200.0, 150.0, -30.0],
    "status": "active"
  }
}
```

### GET /api/v1/pwsa/tracks

List all active tracks.

**Query Parameters:**
- `limit` (optional) - Max results (default: 20)
- `offset` (optional) - Pagination offset (default: 0)

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "track_id": "track-001",
      "last_updated": 1234567890,
      "position": [150.0, 200.0, 180.0],
      "velocity": [1200.0, 150.0, -30.0],
      "status": "active"
    }
  ]
}
```

### POST /api/v1/pwsa/fuse

Multi-sensor data fusion.

**Request Body:**
```json
[
  {
    "sv_id": 42,
    "timestamp": 1234567890,
    "ir_frame": { /* IR data */ }
  },
  {
    "sv_id": 43,
    "timestamp": 1234567891,
    "ir_frame": { /* IR data */ }
  }
]
```

**Response:**
```json
{
  "success": true,
  "data": {
    "fused_tracks": [ /* track data */ ],
    "processing_time_ms": 2.5,
    "sensors_used": 2,
    "confidence": 0.95
  }
}
```

### GET /api/v1/pwsa/health

PWSA subsystem health check.

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "sensors_online": 12,
    "processing_latency_ms": 1.2
  }
}
```

---

## Finance Endpoints

Portfolio optimization and risk assessment.

### POST /api/v1/finance/optimize

Optimize portfolio allocation.

**Request Body:**
```json
{
  "assets": [
    {
      "symbol": "AAPL",
      "expected_return": 0.12,
      "volatility": 0.25,
      "current_price": 150.0
    },
    {
      "symbol": "GOOGL",
      "expected_return": 0.15,
      "volatility": 0.30,
      "current_price": 2800.0
    }
  ],
  "constraints": {
    "max_position_size": 0.3,
    "min_position_size": 0.05,
    "max_total_risk": 0.15,
    "sector_limits": [
      {
        "sector": "Technology",
        "max_allocation": 0.5
      }
    ]
  },
  "objective": "maximize_sharpe"
}
```

**Objective Options:**
- `maximize_sharpe` - Maximize Sharpe ratio
- `minimize_risk` - Minimize portfolio risk
- `maximize_return` - Maximize expected return
- `custom` - Custom risk aversion: `{"custom": {"risk_aversion": 2.0}}`

**Response:**
```json
{
  "success": true,
  "data": {
    "weights": [
      {"symbol": "AAPL", "weight": 0.4},
      {"symbol": "GOOGL", "weight": 0.6}
    ],
    "expected_return": 0.138,
    "expected_risk": 0.14,
    "sharpe_ratio": 0.98,
    "optimization_time_ms": 5.2
  }
}
```

### POST /api/v1/finance/risk

Assess portfolio risk.

**Request Body:**
```json
{
  "portfolio_id": "portfolio-001",
  "confidence_level": 0.95,
  "time_horizon_days": 30
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "var": 50000.0,
    "cvar": 75000.0,
    "max_drawdown": 0.12,
    "beta": 1.05,
    "correlation_matrix": [[1.0, 0.6], [0.6, 1.0]]
  }
}
```

### GET /api/v1/finance/portfolio/:id

Get portfolio details.

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "portfolio-001",
    "name": "Growth Portfolio",
    "value": 1000000.0,
    "cash": 50000.0,
    "positions": [
      {
        "symbol": "AAPL",
        "shares": 1000,
        "avg_cost": 140.0,
        "current_price": 150.0,
        "value": 150000.0,
        "unrealized_gain": 10000.0
      }
    ],
    "last_updated": 1234567890
  }
}
```

---

## Telecom Endpoints

Network optimization and congestion management.

### POST /api/v1/telecom/optimize

Optimize network routing.

**Request Body:**
```json
{
  "topology": {
    "nodes": [
      {
        "id": "router-1",
        "node_type": "router",
        "capacity": 1000.0,
        "current_load": 450.0
      }
    ],
    "links": [
      {
        "source": "router-1",
        "target": "router-2",
        "bandwidth": 1000.0,
        "latency_ms": 5.0,
        "utilization": 0.45
      }
    ]
  },
  "traffic_demands": [
    {
      "source": "router-1",
      "destination": "router-5",
      "bandwidth_required": 100.0,
      "priority": 1
    }
  ],
  "objective": "minimize_latency"
}
```

**Objectives:**
- `minimize_latency` - Minimize end-to-end latency
- `maximize_throughput` - Maximize network throughput
- `balance_load` - Balance load across links

**Response:**
```json
{
  "success": true,
  "data": {
    "routes": [
      {
        "demand_id": 0,
        "path": ["router-1", "router-3", "router-5"],
        "latency_ms": 12.5,
        "bandwidth": 100.0
      }
    ],
    "total_latency_ms": 12.5,
    "max_link_utilization": 0.65,
    "optimization_time_ms": 8.3
  }
}
```

---

## Robotics Endpoints

Motion planning and trajectory execution.

### POST /api/v1/robotics/plan

Plan robot motion from start to goal.

**Request Body:**
```json
{
  "robot_id": "robot-001",
  "start_state": {
    "position": [0.0, 0.0, 0.0],
    "orientation": [0.0, 0.0, 0.0, 1.0],
    "joint_angles": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "velocity": [0.0, 0.0, 0.0]
  },
  "goal_state": {
    "position": [1.0, 1.0, 0.5],
    "orientation": [0.0, 0.0, 0.707, 0.707],
    "joint_angles": [0.5, 0.3, 0.2, 0.1, 0.0, 0.0],
    "velocity": [0.0, 0.0, 0.0]
  },
  "obstacles": [
    {
      "id": "obstacle-1",
      "position": [0.5, 0.5, 0.3],
      "size": [0.2, 0.2, 0.4],
      "obstacle_type": "static"
    }
  ],
  "constraints": {
    "max_velocity": 1.0,
    "max_acceleration": 2.0,
    "max_jerk": 5.0,
    "collision_margin": 0.05
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "trajectory": [
      {
        "time": 0.0,
        "state": { /* start state */ }
      },
      {
        "time": 1.0,
        "state": { /* intermediate */ }
      },
      {
        "time": 5.2,
        "state": { /* goal state */ }
      }
    ],
    "total_time": 5.2,
    "total_distance": 10.5,
    "is_collision_free": true,
    "planning_time_ms": 15.3
  }
}
```

---

## LLM Endpoints

Multi-model LLM orchestration and consensus.

### POST /api/v1/llm/query

Query single LLM (optimal model selection).

**Request Body:**
```json
{
  "prompt": "Explain quantum entanglement in simple terms",
  "temperature": 0.7,
  "max_tokens": 500,
  "model": "claude-3-5-sonnet",  // Optional
  "system_prompt": "You are a physics teacher"  // Optional
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "text": "Quantum entanglement is a phenomenon where...",
    "model_used": "claude-3-5-sonnet",
    "tokens_used": 156,
    "cost_usd": 0.0042,
    "latency_ms": 234.5,
    "confidence": 0.92
  }
}
```

### POST /api/v1/llm/consensus

Multi-model consensus query.

**Request Body:**
```json
{
  "prompt": "Should we approve this financial transaction?",
  "models": ["gpt-4", "claude-3-5-sonnet", "gemini-2.0-flash"],
  "voting_strategy": "quantum",
  "min_agreement": 0.8
}
```

**Voting Strategies:**
- `majority` - Simple majority vote
- `weighted` - Quality-weighted voting
- `quantum` - Quantum voting consensus
- `thermodynamic` - Thermodynamic balancing

**Response:**
```json
{
  "success": true,
  "data": {
    "consensus_text": "Yes, approve the transaction",
    "participating_models": ["gpt-4", "claude-3-5-sonnet", "gemini-2.0-flash"],
    "agreement_score": 0.95,
    "individual_responses": [
      {
        "model": "gpt-4",
        "text": "Yes, approve",
        "confidence": 0.92
      }
    ],
    "total_cost_usd": 0.0168,
    "latency_ms": 456.2
  }
}
```

### GET /api/v1/llm/models

List available models.

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "name": "gpt-4",
      "provider": "openai",
      "enabled": true,
      "quality_score": 0.85,
      "avg_latency_ms": 250.0,
      "cost_per_1k_tokens": 0.03
    },
    {
      "name": "claude-3-5-sonnet",
      "provider": "anthropic",
      "enabled": true,
      "quality_score": 0.90,
      "avg_latency_ms": 220.0,
      "cost_per_1k_tokens": 0.015
    }
  ]
}
```

---

## Time Series Endpoints

Forecasting and trajectory prediction.

### POST /api/v1/timeseries/forecast

Generic time series forecast.

**Request Body:**
```json
{
  "series_id": "btc-usd",
  "historical_data": [100, 102, 105, 103, 108, 112, 110],
  "timestamps": [1, 2, 3, 4, 5, 6, 7],
  "horizon": 10,
  "method": {
    "arima": {
      "p": 2,
      "d": 1,
      "q": 2
    }
  },
  "include_uncertainty": true
}
```

**Methods:**
- `arima` - ARIMA(p, d, q)
- `lstm` - LSTM network
- `gru` - GRU network
- `exponential_smoothing` - Exponential smoothing
- `prophet` - Facebook Prophet

**Response:**
```json
{
  "success": true,
  "data": {
    "series_id": "btc-usd",
    "predictions": [115, 118, 120, 122, 125, 127, 130, 132, 135, 138],
    "timestamps": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    "confidence_intervals": [
      {"lower": 110, "upper": 120, "confidence_level": 0.95}
    ],
    "method_used": "arima",
    "computation_time_ms": 12.5,
    "metrics": {
      "mae": 0.05,
      "rmse": 0.08,
      "mape": 0.03
    }
  }
}
```

### POST /api/v1/timeseries/trajectory

Predict missile/object trajectory.

**Request Body:**
```json
{
  "track_id": "track-001",
  "historical_positions": [
    [100.0, 100.0, 250.0],
    [150.0, 120.0, 240.0],
    [200.0, 140.0, 230.0]
  ],
  "historical_velocities": [
    [50.0, 20.0, -10.0],
    [50.0, 20.0, -10.0],
    [50.0, 20.0, -10.0]
  ],
  "timestamps": [1, 2, 3],
  "horizon_seconds": 10.0
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "track_id": "track-001",
    "predicted_positions": [ /* 100 points */ ],
    "predicted_velocities": [ /* 100 points */ ],
    "timestamps": [ /* 100 timestamps */ ],
    "uncertainty": [0.05, 0.06, 0.07, ...],
    "computation_time_ms": 8.3
  }
}
```

---

## Pixel Processing Endpoints

IR frame analysis, entropy maps, TDA.

### POST /api/v1/pixels/process

Full pixel processing pipeline.

**Request Body:**
```json
{
  "frame_id": "frame-001",
  "width": 640,
  "height": 480,
  "pixels": [1000, 1010, 1005, ...],  // 640*480 = 307200 values
  "processing_options": {
    "compute_entropy": true,
    "compute_tda": true,
    "compute_segmentation": false,
    "extract_features": true,
    "entropy_window_size": 16,
    "tda_threshold": 1000.0
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "frame_id": "frame-001",
    "entropy_map": [ /* 307200 values */ ],
    "tda_features": {
      "persistence_diagram": [
        {"birth": 1000.0, "death": 1500.0, "dimension": 0}
      ],
      "betti_numbers": [5, 2, 0],
      "connected_components": 5,
      "topological_entropy": 2.3
    },
    "segmentation_mask": null,
    "extracted_features": {
      "mean_intensity": 1250.0,
      "std_intensity": 320.5,
      "shannon_entropy": 3.2,
      "edge_density": 0.15,
      "convolution_features": []
    },
    "computation_time_ms": 25.7
  }
}
```

### POST /api/v1/pixels/entropy

Compute entropy map only.

**Request Body:**
```json
{
  "frame_id": "frame-001",
  "width": 640,
  "height": 480,
  "pixels": [1000, 1010, ...],
  "window_size": 16
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "frame_id": "frame-001",
    "entropy_map": [ /* 307200 values */ ],
    "global_entropy": 3.5,
    "high_entropy_regions": [
      {
        "x": 320,
        "y": 240,
        "width": 50,
        "height": 50,
        "entropy": 4.2
      }
    ],
    "computation_time_ms": 8.2
  }
}
```

---

## WebSocket

Real-time event streaming via WebSocket.

### Connection

```javascript
const ws = new WebSocket('ws://localhost:8080/ws');
```

### Event Types

All events have a `type` field:

**ThreatDetected:**
```json
{
  "type": "ThreatDetected",
  "threat_id": "threat-001",
  "confidence": 0.95,
  "position": [150.0, 200.0, 180.0],
  "timestamp": 1234567890
}
```

**PortfolioUpdate:**
```json
{
  "type": "PortfolioUpdate",
  "portfolio_id": "portfolio-001",
  "value": 1050000.0,
  "returns": 0.05,
  "timestamp": 1234567890
}
```

**LlmStream:**
```json
{
  "type": "LlmStream",
  "request_id": "req-uuid",
  "token": "The",
  "is_final": false
}
```

**ForecastUpdate:**
```json
{
  "type": "ForecastUpdate",
  "series_id": "btc-usd",
  "predictions": [115, 118, 120],
  "confidence_intervals": [...],
  "timestamp": 1234567890
}
```

**Ping/Pong:**
```json
// Send
{"type": "Ping"}

// Receive
{"type": "Pong"}
```

---

## Error Handling

### Error Response Format

```json
{
  "error": "BadRequest",
  "message": "Invalid input data",
  "details": "Field 'pixels' must have exactly 307200 elements"
}
```

### Common Errors

| Error | HTTP Code | Description |
|-------|-----------|-------------|
| `BadRequest` | 400 | Invalid input parameters |
| `Unauthorized` | 401 | Missing or invalid API key |
| `Forbidden` | 403 | Insufficient permissions |
| `NotFound` | 404 | Resource not found |
| `ServerError` | 500 | Internal server error |
| `ServiceUnavailable` | 503 | Service temporarily unavailable |

---

## Rate Limiting

Default rate limits:
- **100 requests/second** per API key
- **10 concurrent connections** per API key

Rate limit headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1234567890
```

When exceeded:
```json
{
  "error": "TooManyRequests",
  "message": "Rate limit exceeded",
  "details": "Limit: 100 req/s, Reset at: 2025-10-12T12:00:00Z"
}
```

---

## Examples

### Python Example

```python
import requests

API_KEY = "your-api-key"
BASE_URL = "http://localhost:8080"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# PWSA threat detection
response = requests.post(
    f"{BASE_URL}/api/v1/pwsa/detect",
    headers=headers,
    json={
        "sv_id": 42,
        "timestamp": 1234567890,
        "ir_frame": {
            "width": 640,
            "height": 480,
            "centroid_x": 320.0,
            "centroid_y": 240.0,
            "hotspot_count": 5
        }
    }
)

result = response.json()
print(f"Threat detected: {result['data']['threat_id']}")
print(f"Confidence: {result['data']['confidence']}")
```

### JavaScript Example

```javascript
const API_KEY = 'your-api-key';
const BASE_URL = 'http://localhost:8080';

async function queryLLM(prompt) {
  const response = await fetch(`${BASE_URL}/api/v1/llm/query`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${API_KEY}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      prompt: prompt,
      temperature: 0.7,
      max_tokens: 500
    })
  });

  const result = await response.json();
  console.log(result.data.text);
}

queryLLM('Explain quantum computing');
```

### cURL Examples

```bash
# Health check
curl http://localhost:8080/health

# PWSA detect (with API key)
curl -X POST http://localhost:8080/api/v1/pwsa/detect \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"sv_id": 42, "timestamp": 1234567890, "ir_frame": {...}}'

# Finance optimize
curl -X POST http://localhost:8080/api/v1/finance/optimize \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"assets": [...], "constraints": {...}, "objective": "maximize_sharpe"}'

# Time series forecast
curl -X POST http://localhost:8080/api/v1/timeseries/forecast \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"series_id": "btc-usd", "historical_data": [...], "horizon": 10, "method": {"arima": {"p": 2, "d": 1, "q": 2}}}'
```

---

## Support

- **Documentation**: See [API Server README](../03-Source-Code/src/api_server/README.md)
- **Deployment**: See [Deployment Guide](../deployment/README.md)
- **Issues**: Report on GitHub
- **Email**: support@prism-ai.example.com

---

**Worker 8 - API Documentation v1.0**
