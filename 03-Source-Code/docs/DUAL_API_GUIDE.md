# PRISM-AI Dual API Guide

**Worker 8 Deliverable** - REST + GraphQL API Integration

---

## Overview

PRISM-AI now supports **dual API access** - clients can choose between:

1. **REST API** - Traditional HTTP endpoints at `/api/v1/*`
2. **GraphQL API** - Flexible queries and mutations at `/graphql`

Both APIs provide access to the same underlying capabilities (Workers 1, 3, 7) with different strengths.

---

## Quick Start

### REST API Example

```bash
curl -X POST http://localhost:8080/api/v1/timeseries/forecast \
  -H 'Content-Type: application/json' \
  -d '{
    "historical_data": [100, 102, 101, 105, 108],
    "horizon": 5,
    "method": {"Arima": {"p": 2, "d": 1, "q": 1}}
  }'
```

### GraphQL API Example

```bash
curl -X POST http://localhost:8080/graphql \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "query { forecastTimeSeries(input: {historicalData: [100,102,101,105,108], horizon: 5, method: \"ARIMA\"}) { predictions method horizon } }"
  }'
```

### GraphQL Playground

Visit http://localhost:8080/graphql in your browser for an interactive GraphQL playground with:
- Schema introspection
- Auto-completion
- Documentation explorer
- Query builder

---

## API Comparison

### When to Use REST

✅ **Use REST when you:**
- Want simple, straightforward requests
- Need HTTP caching (GET requests)
- Are using standard tools (curl, Postman, wget)
- Prefer traditional REST conventions
- Need one resource per request

**REST Advantages:**
- Simplicity - easy to understand and use
- Caching - HTTP caching works out of the box
- Tooling - curl, Postman, etc work perfectly
- Stateless - no session management needed

### When to Use GraphQL

✅ **Use GraphQL when you:**
- Need flexible, dynamic queries
- Want to fetch multiple resources in one request
- Need nested data with relationships
- Want type-safe contracts with introspection
- Prefer avoiding API versioning (/v1, /v2)

**GraphQL Advantages:**
- Flexibility - request exactly the data you need
- Efficiency - single request for complex data
- Type safety - strong schema validation
- Introspection - self-documenting API
- Versioning - no need for /v1, /v2 endpoints

---

## GraphQL Schema

### Queries

#### `health` - API Health Status
```graphql
query {
  health {
    status
    version
    uptimeSeconds
  }
}
```

#### `gpuStatus` - GPU Information
```graphql
query {
  gpuStatus {
    available
    deviceCount
    totalMemoryMb
    freeMemoryMb
    utilizationPercent
  }
}
```

#### `forecastTimeSeries` - Time Series Forecasting (Worker 1)
```graphql
query ForecastTimeSeries($input: TimeSeriesForecastInput!) {
  forecastTimeSeries(input: $input) {
    predictions
    method
    horizon
    confidenceIntervals {
      lower
      upper
    }
  }
}
```

**Variables:**
```json
{
  "input": {
    "historicalData": [100.0, 102.0, 101.0, 105.0, 108.0],
    "horizon": 5,
    "method": "ARIMA"
  }
}
```

#### `optimizePortfolio` - Portfolio Optimization (Worker 3)
```graphql
query OptimizePortfolio($input: PortfolioOptimizationInput!) {
  optimizePortfolio(input: $input) {
    weights {
      symbol
      weight
    }
    expectedReturn
    portfolioRisk
    sharpeRatio
  }
}
```

**Variables:**
```json
{
  "input": {
    "assets": [
      {
        "symbol": "AAPL",
        "expectedReturn": 0.12,
        "volatility": 0.20
      },
      {
        "symbol": "GOOGL",
        "expectedReturn": 0.15,
        "volatility": 0.25
      }
    ],
    "objective": "MaximizeSharpe"
  }
}
```

#### `planRobotMotion` - Motion Planning (Worker 7)
```graphql
query PlanRobotMotion($input: MotionPlanInput!) {
  planRobotMotion(input: $input) {
    waypoints {
      time
      position { x y z }
      velocity { x y z }
    }
    totalTime
    totalDistance
    isCollisionFree
  }
}
```

**Variables:**
```json
{
  "input": {
    "robotId": "robot-1",
    "start": { "x": 0.0, "y": 0.0, "z": 0.0 },
    "goal": { "x": 5.0, "y": 3.0, "z": 0.0 }
  }
}
```

#### `performanceMetrics` - Endpoint Performance
```graphql
query {
  performanceMetrics {
    endpoint
    avgResponseTimeMs
    p95ResponseTimeMs
    requestsPerSecond
    errorRate
  }
}
```

### Mutations

#### `submitForecast` - Submit Time Series Forecast
```graphql
mutation SubmitForecast($input: TimeSeriesForecastInput!) {
  submitForecast(input: $input) {
    predictions
    method
    horizon
  }
}
```

#### `submitPortfolioOptimization` - Submit Portfolio Optimization
```graphql
mutation SubmitPortfolio($input: PortfolioOptimizationInput!) {
  submitPortfolioOptimization(input: $input) {
    weights {
      symbol
      weight
    }
    expectedReturn
    sharpeRatio
  }
}
```

#### `submitMotionPlan` - Submit Motion Planning Request
```graphql
mutation SubmitMotionPlan($input: MotionPlanInput!) {
  submitMotionPlan(input: $input) {
    waypoints {
      time
      position { x y z }
    }
    totalTime
    isCollisionFree
  }
}
```

---

## Advanced GraphQL Examples

### Example 1: Dashboard Query (Multiple Resources)

**GraphQL Strength**: Fetch health, GPU status, and metrics in ONE request

```graphql
query Dashboard {
  health {
    status
    version
  }
  gpuStatus {
    available
    utilizationPercent
    freeMemoryMb
  }
  performanceMetrics {
    endpoint
    avgResponseTimeMs
    requestsPerSecond
  }
}
```

**REST Equivalent**: Would require 3 separate requests!
```bash
# Request 1
curl http://localhost:8080/health

# Request 2
curl http://localhost:8080/api/v1/gpu/status

# Request 3
curl http://localhost:8080/api/v1/gpu/metrics
```

### Example 2: Partial Fields (Reduce Bandwidth)

**GraphQL**: Request only the fields you need
```graphql
query {
  forecastTimeSeries(input: {
    historicalData: [100, 102, 104],
    horizon: 3,
    method: "ARIMA"
  }) {
    predictions  # Only return predictions, skip method/horizon
  }
}
```

**REST**: Always returns full response (predictions + method + horizon + metadata)

### Example 3: Nested Query with Relationships

```graphql
query CompleteAnalysis {
  # Get forecast
  forecast: forecastTimeSeries(input: {
    historicalData: [100, 102, 104, 106],
    horizon: 5,
    method: "LSTM"
  }) {
    predictions
    horizon
  }

  # Get portfolio optimization
  portfolio: optimizePortfolio(input: {
    assets: [
      { symbol: "AAPL", expectedReturn: 0.12, volatility: 0.20 },
      { symbol: "GOOGL", expectedReturn: 0.15, volatility: 0.25 }
    ],
    objective: "MaximizeSharpe"
  }) {
    weights {
      symbol
      weight
    }
    sharpeRatio
  }

  # Get GPU status
  gpuStatus {
    utilizationPercent
    freeMemoryMb
  }
}
```

This single query replaces 3 REST calls and returns exactly the data needed!

---

## REST API Endpoints

For comparison, here are the equivalent REST endpoints:

### Time Series (Worker 1)
```
POST /api/v1/timeseries/forecast
```

### Finance (Worker 3)
```
POST /api/v1/finance/optimize
GET  /api/v1/finance/backtest
GET  /api/v1/finance/risk
```

### Robotics (Worker 7)
```
POST /api/v1/robotics/plan
POST /api/v1/robotics/execute
```

### GPU Monitoring
```
GET  /api/v1/gpu/status
GET  /api/v1/gpu/metrics
GET  /api/v1/gpu/utilization
POST /api/v1/gpu/benchmark
```

### Health
```
GET  /health
GET  /
```

---

## GraphQL Schema Introspection

Get the full GraphQL schema definition:

```bash
curl http://localhost:8080/graphql/schema
```

Response:
```json
{
  "sdl": "type Query { ... }",
  "endpoints": {
    "playground": "/graphql",
    "endpoint": "/graphql",
    "schema": "/graphql/schema"
  }
}
```

---

## Client Libraries

### JavaScript/TypeScript

```typescript
// REST API
const response = await fetch('http://localhost:8080/api/v1/timeseries/forecast', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    historical_data: [100, 102, 104],
    horizon: 5,
    method: { Arima: { p: 2, d: 1, q: 1 } }
  })
});

// GraphQL API
const response = await fetch('http://localhost:8080/graphql', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: `
      query {
        forecastTimeSeries(input: {
          historicalData: [100, 102, 104]
          horizon: 5
          method: "ARIMA"
        }) {
          predictions
        }
      }
    `
  })
});
```

### Python

```python
import requests

# REST API
response = requests.post(
    'http://localhost:8080/api/v1/timeseries/forecast',
    json={
        'historical_data': [100, 102, 104],
        'horizon': 5,
        'method': {'Arima': {'p': 2, 'd': 1, 'q': 1}}
    }
)

# GraphQL API
response = requests.post(
    'http://localhost:8080/graphql',
    json={
        'query': '''
            query {
                forecastTimeSeries(input: {
                    historicalData: [100, 102, 104]
                    horizon: 5
                    method: "ARIMA"
                }) {
                    predictions
                }
            }
        '''
    }
)
```

---

## Performance Considerations

### REST
- **Caching**: GET requests can be cached by browsers/proxies
- **Overfetching**: May return more data than needed
- **Underfetching**: May require multiple requests (N+1 problem)

### GraphQL
- **Caching**: Requires custom caching strategies
- **Efficiency**: Single request for complex data
- **Flexibility**: Request exactly what you need

**Recommendation**: Use REST for simple CRUD operations, GraphQL for complex queries.

---

## Testing

### REST API Tests

```bash
# Test time series forecasting
./client-libraries/bash/test_timeseries_rest.sh

# Test portfolio optimization
./client-libraries/bash/test_portfolio_rest.sh

# Test robotics planning
./client-libraries/bash/test_robotics_rest.sh
```

### GraphQL API Tests

```bash
# Test GraphQL queries
curl -X POST http://localhost:8080/graphql \
  -H 'Content-Type: application/json' \
  -d @graphql_test_queries.json

# Use GraphQL playground
open http://localhost:8080/graphql
```

---

## Troubleshooting

### GraphQL Errors

If you see GraphQL errors, check:
1. **Syntax**: Ensure query syntax is valid (use playground)
2. **Fields**: Verify field names match schema (case-sensitive)
3. **Variables**: Check variable types match schema
4. **Introspection**: Use `/graphql/schema` to see available fields

### REST Errors

If you see REST errors, check:
1. **HTTP Method**: POST for mutations, GET for queries
2. **Content-Type**: Must be `application/json`
3. **Request Body**: Valid JSON matching expected schema
4. **Endpoint**: Verify endpoint path is correct

---

## Migration Guide

### From REST to GraphQL

**Before (REST):**
```bash
# Request 1: Get forecast
curl -X POST http://localhost:8080/api/v1/timeseries/forecast -d '{...}'

# Request 2: Get GPU status
curl http://localhost:8080/api/v1/gpu/status

# Request 3: Get metrics
curl http://localhost:8080/api/v1/gpu/metrics
```

**After (GraphQL):**
```bash
# Single request
curl -X POST http://localhost:8080/graphql -d '{
  "query": "query { forecastTimeSeries(...) { predictions } gpuStatus { available } performanceMetrics { endpoint } }"
}'
```

**Benefits**:
- 3 requests → 1 request
- Reduced latency (no round trips)
- Fetch only needed fields
- Type-safe with schema validation

---

## Support

- **Documentation**: `/docs/API_DOCUMENTATION.md`
- **GraphQL Schema**: http://localhost:8080/graphql/schema
- **GraphQL Playground**: http://localhost:8080/graphql
- **Health Check**: http://localhost:8080/health

---

**Worker 8 Dual API Implementation** - Complete ✅
**Status**: Production Ready (pending Worker 1/2 GPU kernel dependencies)
**Date**: 2025-10-13
