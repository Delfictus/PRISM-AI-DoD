# PRISM-AI Command Line Interface

A powerful command-line tool for interacting with the PRISM-AI REST API.

## Installation

### From Source

```bash
cd examples/cli
cargo build --release
sudo cp target/release/prism /usr/local/bin/
```

### From Cargo

```bash
cargo install prism-cli
```

## Quick Start

### 1. Initialize Configuration

```bash
prism config init
```

This creates a configuration file at `~/.config/prism-cli/config.toml`.

### 2. Set API Key

```bash
prism config set api_key your-api-key-here
prism config set api_url http://localhost:8080
```

Or set environment variables:

```bash
export PRISM_API_KEY=your-api-key-here
export PRISM_API_URL=http://localhost:8080
```

### 3. Test Connection

```bash
prism health
```

## Commands

### Configuration

```bash
# Initialize configuration
prism config init

# Show current configuration
prism config show

# Set configuration values
prism config set api_key <key>
prism config set api_url <url>
```

### Health Check

```bash
# Check API health
prism health

# JSON output
prism --output json health
```

### PWSA Commands

#### Detect Threats

```bash
prism pwsa detect \
  --sv-id 42 \
  --timestamp 1234567890 \
  --ir-frame data/ir_frame.json
```

Example `ir_frame.json`:
```json
{
  "width": 640,
  "height": 480,
  "centroid_x": 320.0,
  "centroid_y": 240.0,
  "hotspot_count": 5
}
```

#### Sensor Fusion

```bash
prism pwsa fuse \
  --sv-id 42 \
  --timestamp 1234567890 \
  --sensors data/sensors.json
```

Example `sensors.json`:
```json
{
  "ir": {
    "frame_id": 1234,
    "targets": [
      {
        "id": "target_1",
        "azimuth": 10.0,
        "elevation": 5.0,
        "range": 25000.0,
        "velocity": 500.0
      }
    ]
  },
  "radar": {
    "scan_id": 5678,
    "targets": []
  }
}
```

#### Trajectory Prediction

```bash
prism pwsa predict \
  --track-id threat_001 \
  --history data/history.json \
  --horizon 30 \
  --model kalman_filter
```

Example `history.json`:
```json
[
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
]
```

#### Threat Prioritization

```bash
prism pwsa prioritize \
  --threats data/threats.json \
  --strategy time_weighted_risk
```

### Finance Commands

#### Portfolio Optimization

```bash
prism finance optimize \
  --assets data/assets.json \
  --constraints data/constraints.json \
  --objective maximize_sharpe
```

Example `assets.json`:
```json
[
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
]
```

Example `constraints.json`:
```json
{
  "max_position_size": 0.5,
  "min_position_size": 0.1,
  "max_total_risk": 0.20
}
```

#### Risk Assessment

```bash
prism finance risk \
  --portfolio-id portfolio_001 \
  --positions data/positions.json \
  --metrics var,cvar,max_drawdown,beta
```

Example `positions.json`:
```json
[
  {
    "symbol": "AAPL",
    "quantity": 100,
    "entry_price": 145.0,
    "current_price": 150.0
  },
  {
    "symbol": "TSLA",
    "quantity": 50,
    "entry_price": 200.0,
    "current_price": 190.0
  }
]
```

#### Strategy Backtesting

```bash
prism finance backtest \
  --strategy-id momentum_001 \
  --parameters data/params.json \
  --historical-data data/market_data.json \
  --initial-capital 100000.0
```

### LLM Commands

#### Query Single Model

```bash
# Simple query
prism llm query "What is the capital of France?"

# With options
prism llm query "Explain quantum computing" \
  --model gpt-4 \
  --temperature 0.7 \
  --max-tokens 200
```

#### Multi-Model Consensus

```bash
prism llm consensus "What is 2+2?" \
  --models data/models.json \
  --strategy majority_vote \
  --temperature 0.1
```

Example `models.json`:
```json
[
  {"name": "gpt-4", "weight": 1.0},
  {"name": "gpt-3.5-turbo", "weight": 0.8},
  {"name": "claude-3", "weight": 1.0}
]
```

#### List Models

```bash
prism llm models
```

### Time Series Commands

#### Forecast

```bash
prism timeseries forecast \
  --series-id sales_data \
  --data data/historical.json \
  --horizon 10 \
  --method arima
```

Example `historical.json`:
```json
[100, 105, 110, 108, 115, 120, 125, 130]
```

### Pixel Processing Commands

#### Process Pixels

```bash
prism pixels process \
  --frame-id 12345 \
  --width 640 \
  --height 480 \
  --pixels data/frame.json \
  --detect-hotspots \
  --compute-entropy \
  --apply-tda
```

## Output Formats

The CLI supports multiple output formats:

### Table (default)

```bash
prism health
```

Output:
```
✓ API is healthy (status: healthy)
Version: 0.1.0
Uptime: 3600 seconds
```

### JSON

```bash
prism --output json health
```

Output:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "uptime_seconds": 3600
}
```

### YAML

```bash
prism --output yaml health
```

Output:
```yaml
status: healthy
version: 0.1.0
uptime_seconds: 3600
```

## Global Options

```bash
# Override API URL
prism --api-url https://api.example.com health

# Override API key
prism --api-key your-key health

# Change output format
prism --output json health

# Disable colors
prism --no-color health
```

## Environment Variables

- `PRISM_API_KEY` - API authentication key
- `PRISM_API_URL` - API base URL

## Configuration File

Location: `~/.config/prism-cli/config.toml`

```toml
api_url = "http://localhost:8080"
api_key = "your-api-key-here"
```

## Examples

### Complete Workflow: PWSA Threat Detection

```bash
# 1. Create IR frame data
cat > ir_frame.json <<EOF
{
  "width": 640,
  "height": 480,
  "centroid_x": 320.0,
  "centroid_y": 240.0,
  "hotspot_count": 5
}
EOF

# 2. Detect threats
prism pwsa detect \
  --sv-id 42 \
  --timestamp $(date +%s) \
  --ir-frame ir_frame.json

# 3. View JSON output
prism --output json pwsa detect \
  --sv-id 42 \
  --timestamp $(date +%s) \
  --ir-frame ir_frame.json
```

### Complete Workflow: Portfolio Optimization

```bash
# 1. Create assets file
cat > assets.json <<EOF
[
  {"symbol": "AAPL", "expected_return": 0.12, "volatility": 0.25, "current_price": 150.0},
  {"symbol": "GOOGL", "expected_return": 0.15, "volatility": 0.30, "current_price": 2800.0},
  {"symbol": "MSFT", "expected_return": 0.13, "volatility": 0.22, "current_price": 380.0}
]
EOF

# 2. Create constraints file
cat > constraints.json <<EOF
{
  "max_position_size": 0.5,
  "min_position_size": 0.1,
  "max_total_risk": 0.20
}
EOF

# 3. Optimize portfolio
prism finance optimize \
  --assets assets.json \
  --constraints constraints.json \
  --objective maximize_sharpe
```

## Error Handling

The CLI provides clear error messages:

```bash
$ prism health
Error: API key not set. Use 'prism config init' or set PRISM_API_KEY environment variable.

$ prism pwsa detect --sv-id 42
error: the following required arguments were not provided:
  --timestamp <TIMESTAMP>
  --ir-frame <IR_FRAME>
```

## Development

### Build

```bash
cargo build
```

### Run Tests

```bash
cargo test
```

### Install Locally

```bash
cargo install --path .
```

## License

MIT License

## Support

- Documentation: https://docs.prism-ai.example.com
- Issues: https://github.com/your-org/prism-ai/issues
- Email: support@prism-ai.example.com
