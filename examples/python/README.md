# PRISM-AI Python Client Library

Official Python SDK for the PRISM-AI REST API.

## Installation

### From source

```bash
cd examples/python
pip install -e .
```

### With pip (when published)

```bash
pip install prism-client
```

## Quick Start

```python
from prism_client import PrismClient

# Initialize client
client = PrismClient(api_key="your-api-key-here")

# Check API health
health = client.health()
print(health)

# Detect threat
threat = client.detect_threat(
    sv_id=42,
    timestamp=1234567890,
    ir_frame={
        "width": 640,
        "height": 480,
        "centroid_x": 320.0,
        "centroid_y": 240.0,
        "hotspot_count": 5
    }
)
print(f"Threat detected: {threat.threat_type} with {threat.confidence:.2%} confidence")

# Optimize portfolio
portfolio = client.optimize_portfolio(
    assets=[
        {"symbol": "AAPL", "expected_return": 0.12, "volatility": 0.25, "current_price": 150.0},
        {"symbol": "GOOGL", "expected_return": 0.15, "volatility": 0.30, "current_price": 2800.0},
    ],
    constraints={
        "max_position_size": 0.5,
        "min_position_size": 0.1,
        "max_total_risk": 0.20
    }
)
print(f"Sharpe ratio: {portfolio.sharpe_ratio:.2f}")

# Query LLM
response = client.query_llm(
    prompt="Explain quantum computing in one sentence.",
    temperature=0.7,
    max_tokens=100
)
print(f"LLM response: {response.text}")
```

## Usage Examples

### Context Manager

```python
with PrismClient(api_key="your-key") as client:
    health = client.health()
    print(health)
# Client automatically closed
```

### PWSA - Threat Detection

```python
# Detect threat from IR frame
threat = client.detect_threat(
    sv_id=42,
    timestamp=1234567890,
    ir_frame={
        "width": 640,
        "height": 480,
        "centroid_x": 320.0,
        "centroid_y": 240.0,
        "hotspot_count": 5
    }
)

print(f"Threat ID: {threat.threat_id}")
print(f"Type: {threat.threat_type}")
print(f"Confidence: {threat.confidence:.2%}")
print(f"Position: {threat.position}")
```

### PWSA - Sensor Fusion

```python
# Fuse multi-sensor data
fusion = client.fuse_sensors(
    sv_id=42,
    timestamp=1234567890,
    sensors={
        "ir": {
            "frame_id": 1234,
            "targets": [
                {
                    "id": "target_1",
                    "azimuth": 10.0,
                    "elevation": 5.0,
                    "range": 25000.0,
                    "velocity": 500.0,
                    "ir_signature": 0.85,
                    "radar_cross_section": 1.5
                }
            ]
        },
        "radar": {
            "scan_id": 5678,
            "targets": []
        }
    }
)

print(f"Tracks: {fusion.num_tracks}")
print(f"Fusion quality: {fusion.fusion_quality:.2%}")
```

### PWSA - Trajectory Prediction

```python
# Predict threat trajectory
prediction = client.predict_trajectory(
    track_id="threat_001",
    history=[
        {
            "timestamp": 1234567890,
            "position": [10000.0, 20000.0, 5000.0],
            "velocity": [800.0, -200.0, 0.0]
        },
        {
            "timestamp": 1234567900,
            "position": [18000.0, 18000.0, 5000.0],
            "velocity": [800.0, -200.0, 0.0]
        }
    ],
    prediction_horizon=30,
    model="kalman_filter"
)

print(f"Confidence: {prediction.confidence:.2%}")
print(f"Time to impact: {prediction.time_to_impact}s")
```

### Finance - Portfolio Optimization

```python
# Optimize portfolio
portfolio = client.optimize_portfolio(
    assets=[
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
        },
        {
            "symbol": "MSFT",
            "expected_return": 0.13,
            "volatility": 0.22,
            "current_price": 380.0
        }
    ],
    constraints={
        "max_position_size": 0.5,
        "min_position_size": 0.1,
        "max_total_risk": 0.20
    },
    objective="maximize_sharpe"
)

print("Optimal weights:")
for weight in portfolio.weights:
    print(f"  {weight['symbol']}: {weight['weight']:.1%}")
print(f"Expected return: {portfolio.expected_return:.2%}")
print(f"Expected risk: {portfolio.expected_risk:.2%}")
print(f"Sharpe ratio: {portfolio.sharpe_ratio:.2f}")
```

### Finance - Risk Assessment

```python
# Assess portfolio risk
risk = client.assess_risk(
    portfolio_id="portfolio_001",
    positions=[
        {"symbol": "AAPL", "quantity": 100, "entry_price": 145.0, "current_price": 150.0},
        {"symbol": "TSLA", "quantity": 50, "entry_price": 200.0, "current_price": 190.0}
    ],
    risk_metrics=["var", "cvar", "max_drawdown", "beta"]
)

print(f"VaR: {risk['var']:.2%}")
print(f"CVaR: {risk['cvar']:.2%}")
print(f"Max Drawdown: {risk['max_drawdown']:.2%}")
```

### LLM - Single Query

```python
# Query language model
response = client.query_llm(
    prompt="What is the capital of France?",
    temperature=0.1,
    max_tokens=50
)

print(f"Response: {response.text}")
print(f"Model: {response.model_used}")
print(f"Tokens: {response.tokens_used}")
print(f"Cost: ${response.cost_usd:.4f}")
```

### LLM - Multi-Model Consensus

```python
# Get consensus from multiple models
consensus = client.llm_consensus(
    prompt="What is the capital of Australia?",
    models=[
        {"name": "gpt-4", "weight": 1.0},
        {"name": "gpt-3.5-turbo", "weight": 0.8},
        {"name": "claude-3", "weight": 1.0}
    ],
    strategy="majority_vote",
    temperature=0.1,
    max_tokens=50
)

print(f"Consensus: {consensus.consensus_text}")
print(f"Confidence: {consensus.confidence:.2%}")
print(f"Agreement: {consensus.agreement_rate:.1%}")
print(f"Total cost: ${consensus.total_cost_usd:.4f}")
```

### Time Series - Forecasting

```python
# Forecast time series
forecast = client.forecast_timeseries(
    series_id="sales_data",
    historical_data=[100, 105, 110, 108, 115, 120, 125, 130],
    horizon=5,
    method="arima"
)

print(f"Predictions: {forecast.predictions}")
print(f"Confidence intervals: {forecast.confidence_intervals}")
print(f"Metrics: {forecast.metrics}")
```

### Pixel Processing

```python
import numpy as np

# Generate sample IR frame
frame = np.random.randint(0, 256, size=(480, 640), dtype=np.uint8)

# Process pixels
result = client.process_pixels(
    frame_id=12345,
    width=640,
    height=480,
    pixels=frame.flatten().tolist(),
    options={
        "detect_hotspots": True,
        "compute_entropy": True,
        "apply_tda": True
    }
)

print(f"Hotspots detected: {len(result.hotspots)}")
print(f"Entropy: {result.entropy:.3f}")
print(f"TDA features: {len(result.tda_features)}")
```

## Error Handling

```python
from prism_client import (
    PrismClient,
    AuthenticationError,
    RateLimitError,
    ValidationError,
    ServerError,
)

client = PrismClient(api_key="your-key")

try:
    threat = client.detect_threat(sv_id=42, timestamp=1234567890, ir_frame={})
except AuthenticationError:
    print("Invalid API key")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after}s")
except ValidationError as e:
    print(f"Validation error: {e.message}")
except ServerError:
    print("Server error occurred")
```

## Configuration

### Custom Base URL

```python
client = PrismClient(
    api_key="your-key",
    base_url="https://api.prism-ai.example.com"
)
```

### Custom Timeout

```python
client = PrismClient(
    api_key="your-key",
    timeout=60  # 60 seconds
)
```

### Disable SSL Verification (development only)

```python
client = PrismClient(
    api_key="your-key",
    verify_ssl=False  # NOT recommended for production
)
```

## API Reference

### Client Methods

**Health & Info:**
- `health()` - Check API health
- `info()` - Get API information

**PWSA:**
- `detect_threat()` - Detect threats from IR data
- `fuse_sensors()` - Fuse multi-sensor data
- `predict_trajectory()` - Predict threat trajectory
- `prioritize_threats()` - Prioritize multiple threats

**Finance:**
- `optimize_portfolio()` - Optimize portfolio allocation
- `assess_risk()` - Assess portfolio risk
- `backtest_strategy()` - Backtest trading strategy

**LLM:**
- `query_llm()` - Query language model
- `llm_consensus()` - Multi-model consensus
- `list_llm_models()` - List available models

**Time Series:**
- `forecast_timeseries()` - Forecast time series

**Pixel Processing:**
- `process_pixels()` - Process pixel data

## Development

### Setup Development Environment

```bash
cd examples/python
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black prism_client/
flake8 prism_client/
```

### Type Checking

```bash
mypy prism_client/
```

## License

MIT License

## Support

- Documentation: https://docs.prism-ai.example.com
- Issues: https://github.com/your-org/prism-ai/issues
- Email: support@prism-ai.example.com
