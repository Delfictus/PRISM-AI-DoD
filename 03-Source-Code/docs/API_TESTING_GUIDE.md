# PRISM-AI API Testing Guide

**Phase 3 - Task 2: End-to-End API Testing**
**Worker 8 - Integration Lead**
**Date**: October 13, 2025

---

## Overview

This guide covers comprehensive testing of all PRISM-AI API endpoints:
- **50 REST endpoints** across 8 worker domains
- **15 GraphQL endpoints** (10 queries, 5 mutations)
- **Load testing** (100+ concurrent requests)
- **Performance profiling** and optimization

---

## Test Infrastructure

### Test Suite Location
```bash
03-Source-Code/test_all_apis.sh
```

### Prerequisites
1. **API Server Running**:
   ```bash
   cargo run --bin api_server
   # OR
   ./target/release/api_server
   ```

2. **Dependencies**:
   - `curl` (HTTP client)
   - `jq` (JSON parsing)
   - `ab` (Apache Bench - for load testing)

### Running Tests
```bash
# Run all endpoint tests
cd 03-Source-Code
./test_all_apis.sh

# Run with custom API URL
API_URL=http://production-server:8080 ./test_all_apis.sh

# Run specific category
./test_all_apis.sh --category worker3
```

---

## API Endpoints Coverage

### 1. Core Infrastructure (3 endpoints)

| Endpoint | Method | Status |
|----------|--------|--------|
| `/health` | GET | ✅ Operational |
| `/` | GET | ✅ Operational |
| `/ws` | WebSocket | ✅ Operational |

**Test Example**:
```bash
curl http://localhost:8080/health
# Expected: "PRISM-AI API Server - Healthy"
```

---

### 2. Worker 3: Application Domains (12 endpoints)

#### Healthcare (2 endpoints)
```bash
# Risk Prediction
POST /api/v1/applications/healthcare/predict_risk
{
  "historical_metrics": [0.2, 0.25, 0.3, 0.28, 0.32],
  "horizon": 5,
  "risk_factors": ["age", "bmi"]
}

# Expected Response
{
  "risk_trajectory": [0.3, 0.35, 0.4, 0.45, 0.5],
  "risk_level": "MEDIUM",
  "confidence": 0.85,
  "warnings": ["Elevated risk trend detected"]
}
```

#### Energy (1 endpoint)
```bash
POST /api/v1/applications/energy/forecast_load
{
  "historical_load": [100.0, 105.0, 110.0, 108.0],
  "horizon": 5
}
```

#### Manufacturing (1 endpoint)
```bash
POST /api/v1/applications/manufacturing/predict_maintenance
{
  "sensor_data": [95.0, 96.5, 98.0, 99.5],
  "equipment_id": "PUMP-001",
  "window": 24
}
```

#### Supply Chain (1 endpoint)
```bash
POST /api/v1/applications/supply_chain/forecast_demand
{
  "historical_demand": [100.0, 105.0, 110.0],
  "product_id": "SKU-123",
  "horizon": 7
}
```

#### Agriculture (1 endpoint)
```bash
POST /api/v1/applications/agriculture/predict_yield
{
  "historical_yield": [4000.0, 4200.0, 4100.0],
  "horizon": 4
}
```

#### Cybersecurity (1 endpoint)
```bash
POST /api/v1/applications/cybersecurity/predict_threats
{
  "historical_events": [10.0, 12.0, 15.0],
  "threat_levels": [2.0, 2.5, 3.0],
  "horizon": 6
}
```

#### Climate (1 endpoint)
```bash
POST /api/v1/applications/climate/forecast
{
  "historical_data": [20.0, 21.5, 23.0],
  "location": "NYC",
  "horizon": 5
}
```

#### Smart Cities (1 endpoint)
```bash
POST /api/v1/applications/smart_city/optimize
{
  "resource_type": "energy",
  "current_levels": [80.0, 85.0, 90.0],
  "horizon": 24
}
```

#### Education (1 endpoint)
```bash
POST /api/v1/applications/education/predict_performance
{
  "historical_performance": [85.0, 82.0, 80.0],
  "student_id": "STU-001",
  "horizon": 4
}
```

#### Retail (1 endpoint)
```bash
POST /api/v1/applications/retail/optimize_inventory
{
  "historical_sales": [100.0, 110.0, 105.0],
  "product_id": "PROD-001",
  "current_inventory": 50.0,
  "horizon": 7
}
```

#### Construction (1 endpoint)
```bash
POST /api/v1/applications/construction/forecast_project
{
  "project_id": "PROJ-001",
  "historical_progress": [10.0, 25.0, 40.0, 55.0],
  "horizon": 30
}
```

---

### 3. Worker 4: Advanced Finance (4 endpoints)

#### Portfolio Optimization
```bash
POST /api/v1/finance_advanced/optimize_advanced
{
  "assets": [
    {"symbol": "AAPL", "expected_return": 0.12, "volatility": 0.20},
    {"symbol": "GOOGL", "expected_return": 0.15, "volatility": 0.25}
  ],
  "strategy": "maximize_sharpe",
  "use_gnn": false,
  "risk_free_rate": 0.03
}

# Expected Response
{
  "weights": [
    {"symbol": "AAPL", "weight": 0.30},
    {"symbol": "GOOGL", "weight": 0.40},
    {"symbol": "MSFT", "weight": 0.30}
  ],
  "expected_return": 0.15,
  "portfolio_risk": 0.18,
  "sharpe_ratio": 0.72,
  "method": "Interior Point QP (Max Sharpe)"
}
```

#### GNN Portfolio Prediction
```bash
POST /api/v1/finance_advanced/gnn/predict
{
  "assets": [{
    "symbol": "AAPL",
    "expected_return": 0.12,
    "volatility": 0.20,
    "price_history": [100.0, 102.0, 104.0]
  }],
  "horizon": 30
}
```

#### Transfer Entropy Causality
```bash
POST /api/v1/finance_advanced/causality/transfer_entropy
{
  "time_series": [
    {"symbol": "AAPL", "values": [100, 102, 101, 105, 108]},
    {"symbol": "MSFT", "values": [200, 198, 202, 205, 207]}
  ],
  "window": 100
}
```

#### Portfolio Rebalancing
```bash
POST /api/v1/finance_advanced/rebalance
{
  "current_weights": [
    {"symbol": "AAPL", "weight": 0.5},
    {"symbol": "GOOGL", "weight": 0.5}
  ],
  "target_weights": [
    {"symbol": "AAPL", "weight": 0.6},
    {"symbol": "GOOGL", "weight": 0.4}
  ],
  "frequency": 30,
  "transaction_cost": 0.001
}
```

---

### 4. Worker 7: Specialized Applications (6 endpoints)

#### Robotics (2 endpoints)
```bash
# Motion Planning
POST /api/v1/worker7/robotics/plan_motion
{
  "start_position": [0.0, 0.0, 0.0],
  "goal_position": [10.0, 10.0, 5.0],
  "algorithm": "RRT"
}

# Trajectory Optimization
POST /api/v1/worker7/robotics/optimize_trajectory
{
  "waypoints": [[0,0,0], [5,5,2], [10,10,5]],
  "objective": "time",
  "max_velocity": 2.0
}
```

#### Drug Discovery (2 endpoints)
```bash
# Molecular Screening
POST /api/v1/worker7/drug_discovery/screen_molecules
{
  "molecules": ["CCO", "CC(=O)O", "CC(C)O"],
  "target_protein": "ACE2",
  "criteria": ["binding_affinity"]
}

# Drug Optimization
POST /api/v1/worker7/drug_discovery/optimize_drug
{
  "seed_molecule": "CCO",
  "target_properties": {
    "target_binding_affinity": 8.0,
    "max_toxicity": 0.3,
    "min_drug_likeness": 0.7
  },
  "max_iterations": 10
}
```

#### Scientific Discovery (2 endpoints)
```bash
# Experiment Design
POST /api/v1/worker7/scientific/design_experiment
{
  "hypothesis": "Temperature affects reaction rate",
  "variables": [{
    "name": "temp",
    "min_value": 20.0,
    "max_value": 100.0,
    "variable_type": "continuous"
  }],
  "num_experiments": 5
}

# Hypothesis Testing
POST /api/v1/worker7/scientific/test_hypothesis
{
  "hypothesis": "Mean > 50",
  "data": [55.0, 58.0, 52.0, 60.0, 56.0],
  "alpha": 0.05
}
```

---

### 5. GraphQL API (15 endpoints)

#### Queries (10)
```graphql
# Health
query {
  health {
    status
    version
    uptimeSeconds
  }
}

# GPU Status
query {
  gpuStatus {
    available
    deviceCount
    totalMemoryMb
    freeMemoryMb
    utilizationPercent
  }
}

# Time Series Forecast
query {
  forecastTimeSeries(input: {
    historicalData: [100.0, 102.0, 104.0]
    horizon: 5
    method: "ARIMA"
  }) {
    predictions
    method
    horizon
    confidenceIntervals {
      lower
      upper
    }
  }
}

# Portfolio Optimization
query {
  optimizePortfolio(input: {
    assets: [{
      symbol: "AAPL"
      expectedReturn: 0.12
      volatility: 0.20
    }]
    objective: "MaximizeSharpe"
  }) {
    weights {
      symbol
      weight
    }
    expectedReturn
    portfolioRisk
    sharpeRatio
  }
}

# Robot Motion Planning
query {
  planRobotMotion(input: {
    startPosition: {x: 0.0, y: 0.0, z: 0.0}
    goalPosition: {x: 10.0, y: 5.0, z: 0.0}
    obstacles: []
  }) {
    waypoints {
      time
      position {x y z}
      velocity {x y z}
    }
    totalTime
    totalDistance
    isCollisionFree
  }
}

# Healthcare Risk Prediction
query {
  healthcarePredictRisk(input: {
    historicalMetrics: [0.2, 0.3, 0.35]
    horizon: 5
    riskFactors: ["age", "bmi"]
  }) {
    riskTrajectory
    riskLevel
    confidence
    warnings
  }
}

# Energy Load Forecast
query {
  energyForecastLoad(input: {
    historicalLoad: [100.0, 105.0, 110.0]
    horizon: 5
  }) {
    forecastedLoad
    peakLoad
    confidenceLower
    confidenceUpper
  }
}

# Molecular Screening
query {
  screenMolecules(input: {
    molecules: ["CCO", "CC(=O)O"]
    targetProtein: "ACE2"
  }) {
    topCandidates
    screeningTimeMs
    numScreened
  }
}

# Experiment Design
query {
  designExperiment(input: {
    hypothesis: "Temperature affects rate"
    numExperiments: 5
  }) {
    numExperiments
    expectedInformationGain
    designStrategy
  }
}

# Performance Metrics
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

#### Mutations (5)
```graphql
# Submit Forecast
mutation {
  submitForecast(input: {
    data: [100.0, 102.0]
    horizon: 5
    method: "ARIMA"
  }) {
    requestId
    status
  }
}

# Submit Portfolio Optimization
mutation {
  submitPortfolioOptimization(input: {
    assets: [...]
    objective: "MaximizeSharpe"
  }) {
    requestId
    status
  }
}

# Submit Motion Plan
mutation {
  submitMotionPlan(input: {
    startPosition: {x: 0, y: 0, z: 0}
    goalPosition: {x: 10, y: 10, z: 5}
  }) {
    requestId
    status
  }
}
```

---

## Load Testing

### Prerequisites
```bash
sudo apt-get install apache2-utils
```

### Basic Load Test (100 concurrent requests)
```bash
# Healthcare risk prediction endpoint
ab -n 1000 -c 100 \
  -p healthcare_payload.json \
  -T application/json \
  http://localhost:8080/api/v1/applications/healthcare/predict_risk

# Expected Results:
# - Requests/sec: 100-500 (depending on hardware)
# - Mean response time: 10-50ms (CPU only)
# - Mean response time: 1-10ms (with GPU)
# - Failed requests: 0
```

### Advanced Load Test (Multiple Endpoints)
```bash
# Create test payloads
cat > healthcare_payload.json << EOF
{"historical_metrics":[0.2,0.3,0.35],"horizon":5,"risk_factors":["age"]}
EOF

cat > finance_payload.json << EOF
{"assets":[{"symbol":"AAPL","expected_return":0.12,"volatility":0.20}],"strategy":"maximize_sharpe","use_gnn":false,"risk_free_rate":0.03}
EOF

# Run concurrent tests
ab -n 1000 -c 100 -p healthcare_payload.json -T application/json \
  http://localhost:8080/api/v1/applications/healthcare/predict_risk &

ab -n 1000 -c 100 -p finance_payload.json -T application/json \
  http://localhost:8080/api/v1/finance_advanced/optimize_advanced &

wait

echo "Load tests complete"
```

### Expected Performance Targets
| Metric | Target | Actual |
|--------|--------|--------|
| Requests/sec | >100 | TBD |
| Mean latency | <50ms | TBD |
| P95 latency | <100ms | TBD |
| P99 latency | <200ms | TBD |
| Error rate | <1% | TBD |
| Concurrent users | 100+ | TBD |

---

## Performance Profiling

### Response Time Measurement
```bash
# Measure individual endpoint latency
time curl -X POST http://localhost:8080/api/v1/applications/healthcare/predict_risk \
  -H "Content-Type: application/json" \
  -d '{"historical_metrics":[0.2,0.3],"horizon":5,"risk_factors":["age"]}'

# Output:
# real    0m0.015s  <- Total time
# user    0m0.005s
# sys     0m0.003s
```

### GraphQL Query Performance
```bash
# Measure GraphQL query time
time curl -X POST http://localhost:8080/graphql \
  -H "Content-Type: application/json" \
  -d '{"query":"query { health { status } }"}'
```

### Profiling Tools
```bash
# CPU profiling
cargo flamegraph --bin api_server

# Memory profiling
valgrind --tool=massif ./target/release/api_server

# GPU profiling (if available)
nsys profile --stats=true ./target/release/api_server
```

---

## Error Handling Tests

### Invalid Input
```bash
# Missing required field
curl -X POST http://localhost:8080/api/v1/applications/healthcare/predict_risk \
  -H "Content-Type: application/json" \
  -d '{"horizon":5}'

# Expected: HTTP 400 Bad Request
```

### Invalid Endpoint
```bash
curl http://localhost:8080/api/v1/invalid_endpoint

# Expected: HTTP 404 Not Found
```

### Malformed JSON
```bash
curl -X POST http://localhost:8080/api/v1/applications/healthcare/predict_risk \
  -H "Content-Type: application/json" \
  -d '{invalid json}'

# Expected: HTTP 400 Bad Request
```

---

## Test Results Format

### JSON Output
```json
{
  "test_suite": "PRISM-AI API Comprehensive Tests",
  "timestamp": "2025-10-13T10:00:00Z",
  "summary": {
    "total_tests": 50,
    "passed": 48,
    "failed": 2,
    "skipped": 0,
    "pass_rate": 96.0
  },
  "categories": {
    "worker3_applications": {"passed": 12, "failed": 0},
    "worker4_finance": {"passed": 4, "failed": 0},
    "worker7_specialized": {"passed": 6, "failed": 0},
    "graphql": {"passed": 15, "failed": 0}
  },
  "failures": [
    {
      "test": "Healthcare: Trajectory Forecast",
      "endpoint": "/api/v1/applications/healthcare/forecast_trajectory",
      "error": "Connection timeout",
      "http_code": 504
    }
  ]
}
```

---

## Continuous Integration

### GitHub Actions Workflow
```yaml
name: API Tests

on: [push, pull_request]

jobs:
  api-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Start API Server
        run: |
          cargo build --release --bin api_server
          ./target/release/api_server &
          sleep 5
      - name: Run API Tests
        run: ./03-Source-Code/test_all_apis.sh
      - name: Upload Results
        uses: actions/upload-artifact@v2
        with:
          name: api-test-results
          path: api_test_results.json
```

---

## Troubleshooting

### Server Not Responding
```bash
# Check if server is running
curl http://localhost:8080/health

# Check server logs
tail -f api_server.log

# Restart server
pkill api_server
cargo run --bin api_server
```

### Slow Response Times
1. Check CPU/GPU utilization
2. Verify GPU acceleration is enabled
3. Check for network latency
4. Review server logs for bottlenecks

### Test Failures
1. Verify API server is running
2. Check payload format matches API spec
3. Review error messages in response
4. Check API server logs

---

## Success Criteria

### Phase 3 Task 2 Complete When:
- ✅ All 50+ REST endpoints tested
- ✅ All 15 GraphQL endpoints tested
- ✅ Pass rate ≥ 90%
- ✅ Load testing passed (100+ concurrent requests)
- ✅ Performance targets met (<50ms mean latency)
- ✅ Error handling validated
- ✅ Test results documented

---

**Document Status**: Complete
**Last Updated**: October 13, 2025
**Maintained By**: Worker 8 (Integration Lead)
