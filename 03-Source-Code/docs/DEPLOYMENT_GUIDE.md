# PRISM-AI API Server - Deployment Guide

**Version**: 1.0.0
**Last Updated**: 2025-10-14
**Worker**: Worker 8 (API Server & Finance)

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Local Development Deployment](#local-development-deployment)
4. [Docker Deployment](#docker-deployment)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Production Deployment](#production-deployment)
7. [Monitoring & Logging](#monitoring--logging)
8. [Troubleshooting](#troubleshooting)
9. [Security Considerations](#security-considerations)

---

## Overview

The PRISM-AI API Server provides RESTful and GraphQL APIs for all application domains including:
- **Worker 3**: 12 application domains (Healthcare, Energy, Manufacturing, etc.)
- **Worker 4**: Advanced finance (Portfolio optimization, GNN, Transfer Entropy)
- **Worker 7**: Specialized applications (Robotics, Drug Discovery, Scientific)

### Architecture

```
┌─────────────────┐
│  Nginx (80/443) │  ← Load Balancer / Reverse Proxy
└────────┬────────┘
         │
┌────────▼────────┐
│  API Server     │  ← Rust Axum Server (port 8080)
│  (REST+GraphQL) │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼──┐  ┌──▼────┐
│ Postgres│  │ Redis │  ← Data & Cache Layers
└─────────┘  └───────┘
```

---

## Prerequisites

### Software Requirements

- **Rust**: 1.75+ (for building from source)
- **Docker**: 20.10+ (for containerized deployment)
- **Docker Compose**: 2.0+ (for multi-container setup)
- **Kubernetes**: 1.25+ (for production orchestration)
- **kubectl**: Latest version

### Hardware Requirements

**Minimum** (Development):
- CPU: 2 cores
- RAM: 4 GB
- Disk: 10 GB

**Recommended** (Production):
- CPU: 8+ cores
- RAM: 16+ GB
- Disk: 100 GB SSD
- Network: 1 Gbps+

---

## Local Development Deployment

### Option 1: Cargo Run (Development)

```bash
# Navigate to source directory
cd 03-Source-Code

# Build and run (development mode)
cargo run --bin api_server

# Or build release binary
cargo build --release --bin api_server
./target/release/api_server
```

The server will start on `http://localhost:8080`.

### Option 2: Systemd Service (Linux)

Create `/etc/systemd/system/prism-api.service`:

```ini
[Unit]
Description=PRISM-AI API Server
After=network.target

[Service]
Type=simple
User=prism
WorkingDirectory=/opt/prism-ai
ExecStart=/opt/prism-ai/api_server
Restart=on-failure
RestartSec=10
Environment="RUST_LOG=info"
Environment="API_HOST=0.0.0.0"
Environment="API_PORT=8080"

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable prism-api
sudo systemctl start prism-api
sudo systemctl status prism-api
```

---

## Docker Deployment

### Single Container

```bash
cd 03-Source-Code

# Build image
docker build -t prism-ai/api-server:latest .

# Run container
docker run -d \
  --name prism-api \
  -p 8080:8080 \
  -e RUST_LOG=info \
  -e API_HOST=0.0.0.0 \
  -e API_PORT=8080 \
  prism-ai/api-server:latest

# Check logs
docker logs -f prism-api

# Health check
curl http://localhost:8080/health
```

### Docker Compose (Full Stack)

```bash
cd 03-Source-Code

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api-server

# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

Services started:
- **api-server**: Port 8080 (REST + GraphQL)
- **postgres**: Port 5432 (Database)
- **redis**: Port 6379 (Cache)
- **nginx**: Ports 80/443 (Reverse proxy)
- **prometheus**: Port 9090 (Metrics)
- **grafana**: Port 3000 (Dashboards)

---

## Kubernetes Deployment

### Prerequisites

1. Kubernetes cluster running
2. kubectl configured
3. Docker registry access (GitHub Container Registry)

### Deploy to Kubernetes

```bash
# Create namespace
kubectl create namespace prism-ai

# Apply configurations
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n prism-ai
kubectl get services -n prism-ai

# View logs
kubectl logs -f deployment/prism-api-server -n prism-ai

# Port forward for testing
kubectl port-forward svc/prism-api-server 8080:8080 -n prism-ai
```

### Scaling

```bash
# Scale horizontally
kubectl scale deployment/prism-api-server --replicas=5 -n prism-ai

# Autoscaling
kubectl autoscale deployment/prism-api-server \
  --cpu-percent=70 \
  --min=2 \
  --max=10 \
  -n prism-ai
```

### Rolling Updates

```bash
# Update image
kubectl set image deployment/prism-api-server \
  api-server=ghcr.io/delfictus/prism-ai-dod/api-server:v1.1.0 \
  -n prism-ai

# Check rollout status
kubectl rollout status deployment/prism-api-server -n prism-ai

# Rollback if needed
kubectl rollout undo deployment/prism-api-server -n prism-ai
```

---

## Production Deployment

### Pre-Deployment Checklist

- [ ] Environment variables configured
- [ ] Secrets stored in vault (not in code)
- [ ] Database migrations applied
- [ ] SSL certificates installed
- [ ] Firewall rules configured
- [ ] Monitoring alerts configured
- [ ] Backup strategy in place
- [ ] Load testing completed
- [ ] Security audit passed
- [ ] Documentation updated

### Environment Variables

```bash
# Required
export RUST_LOG=info
export API_HOST=0.0.0.0
export API_PORT=8080

# Database
export DATABASE_URL=postgresql://user:pass@host:5432/prism
export DATABASE_POOL_SIZE=20

# Cache
export REDIS_URL=redis://host:6379
export REDIS_POOL_SIZE=10

# Security
export JWT_SECRET=<generate-secure-secret>
export API_KEY=<generate-api-key>

# Optional
export ENABLE_CORS=true
export MAX_REQUEST_SIZE=10485760
export REQUEST_TIMEOUT=30
```

### Generate Secrets

```bash
# JWT secret (32 bytes)
openssl rand -hex 32

# API key
openssl rand -base64 32
```

### SSL/TLS Configuration

```bash
# Using Let's Encrypt
certbot certonly --webroot -w /var/www/certbot \
  -d api.prism-ai.example.com \
  --email admin@prism-ai.example.com \
  --agree-tos

# Copy certificates
cp /etc/letsencrypt/live/api.prism-ai.example.com/fullchain.pem /etc/nginx/ssl/cert.pem
cp /etc/letsencrypt/live/api.prism-ai.example.com/privkey.pem /etc/nginx/ssl/key.pem

# Set permissions
chmod 600 /etc/nginx/ssl/key.pem
```

---

## Monitoring & Logging

### Prometheus Metrics

Access metrics at: `http://localhost:8080/metrics`

**Key Metrics:**
- `http_requests_total`: Total HTTP requests
- `http_request_duration_seconds`: Request latency
- `http_requests_in_flight`: Active requests
- `process_cpu_seconds_total`: CPU usage
- `process_resident_memory_bytes`: Memory usage

### Grafana Dashboards

1. Access Grafana: `http://localhost:3000`
2. Default credentials: `admin/admin`
3. Add Prometheus data source: `http://prometheus:9090`
4. Import dashboard: Use dashboard ID `1860` or custom

### Logs

**Structured JSON Logging:**

```bash
# View logs with Docker
docker logs -f prism-api-server

# View logs with kubectl
kubectl logs -f deployment/prism-api-server -n prism-ai

# Filter by log level
docker logs prism-api-server 2>&1 | grep ERROR
```

**Log Levels:**
- `ERROR`: Critical errors requiring immediate attention
- `WARN`: Warning conditions
- `INFO`: Informational messages
- `DEBUG`: Detailed debug information
- `TRACE`: Very detailed trace information

### Health Checks

```bash
# Health endpoint
curl http://localhost:8080/health

# Expected response
# PRISM-AI API Server - Healthy

# GraphQL health query
curl -X POST http://localhost:8080/graphql \
  -H "Content-Type: application/json" \
  -d '{"query":"query { health { status version uptimeSeconds } }"}'
```

### Alerting

Configure alerts in `prometheus.yml`:

```yaml
groups:
  - name: api_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"

      - alert: HighLatency
        expr: histogram_quantile(0.95, http_request_duration_seconds) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency detected (p95 > 1s)"

      - alert: APIServerDown
        expr: up{job="prism-api-server"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "API server is down"
```

---

## Troubleshooting

### Common Issues

#### 1. Port Already in Use

```bash
# Error: Address already in use (os error 98)

# Find process using port 8080
lsof -i :8080
# or
netstat -tulpn | grep 8080

# Kill process
kill -9 <PID>
```

#### 2. Database Connection Failed

```bash
# Check database connectivity
psql -h localhost -U postgres -d prism -c "SELECT 1;"

# Check environment variables
echo $DATABASE_URL

# Test from container
docker exec -it prism-api-server env | grep DATABASE
```

#### 3. High Memory Usage

```bash
# Check memory usage
docker stats prism-api-server

# Restart container
docker restart prism-api-server

# Adjust memory limits in docker-compose.yml
services:
  api-server:
    mem_limit: 2g
    mem_reservation: 1g
```

#### 4. Build Failures

```bash
# Clean build cache
cargo clean

# Update dependencies
cargo update

# Build with verbose output
cargo build --release --bin api_server -vv
```

### Debug Mode

```bash
# Enable debug logging
export RUST_LOG=debug

# Or trace level
export RUST_LOG=trace

# Enable backtraces
export RUST_BACKTRACE=full
```

---

## Security Considerations

### 1. Authentication & Authorization

- Implement JWT-based authentication
- Use API keys for service-to-service communication
- Rotate secrets regularly

### 2. Rate Limiting

Already configured in `nginx.conf`:
- API endpoints: 100 req/s per IP
- GraphQL endpoint: Lower limit (30 req/s)

### 3. HTTPS/TLS

- **Always use HTTPS in production**
- Use TLS 1.2+ only
- Strong cipher suites configured in nginx

### 4. Input Validation

- All inputs validated at API layer
- Request size limits enforced
- SQL injection protection (parameterized queries)

### 5. Security Headers

All security headers configured in nginx:
- `Strict-Transport-Security`
- `X-Frame-Options`
- `X-Content-Type-Options`
- `X-XSS-Protection`

### 6. Container Security

```bash
# Run as non-root user
USER prism

# Scan for vulnerabilities
docker scan prism-ai/api-server:latest

# Use minimal base image
FROM debian:bullseye-slim
```

### 7. Network Security

- Use private networks for internal communication
- Expose only necessary ports
- Configure firewall rules:

```bash
# UFW example
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw deny 8080/tcp  # Internal only
```

---

## Performance Tuning

### 1. Connection Pooling

```rust
// In src/api_server/mod.rs
let pool = PgPoolOptions::new()
    .max_connections(20)
    .connect(&database_url).await?;
```

### 2. Async Runtime

```bash
# Set worker threads (default: num_cpus)
export TOKIO_WORKER_THREADS=8
```

### 3. Caching Strategy

- Redis for frequently accessed data
- HTTP caching headers
- Application-level caching

### 4. Database Optimization

```sql
-- Add indexes for frequently queried columns
CREATE INDEX idx_user_email ON users(email);
CREATE INDEX idx_created_at ON requests(created_at);

-- Analyze query performance
EXPLAIN ANALYZE SELECT * FROM ...;
```

---

## Backup & Recovery

### Database Backups

```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR=/backups/postgres
DATE=$(date +%Y%m%d_%H%M%S)

docker exec prism-postgres pg_dump -U postgres prism > \
  $BACKUP_DIR/prism_$DATE.sql

# Compress
gzip $BACKUP_DIR/prism_$DATE.sql

# Cleanup old backups (keep last 30 days)
find $BACKUP_DIR -name "*.gz" -mtime +30 -delete
```

### Restore from Backup

```bash
# Restore database
gunzip prism_20251014_120000.sql.gz
docker exec -i prism-postgres psql -U postgres prism < prism_20251014_120000.sql
```

---

## Additional Resources

- **API Documentation**: See `docs/API_TESTING_GUIDE.md`
- **Test Results**: See `docs/API_TEST_RESULTS.md`
- **Architecture**: See `docs/ARCHITECTURE.md`
- **Contributing**: See `CONTRIBUTING.md`

---

## Support

For issues and questions:
- **GitHub Issues**: https://github.com/Delfictus/PRISM-AI-DoD/issues
- **Documentation**: https://docs.prism-ai.example.com

---

**Deployment Guide Version**: 1.0.0
**Last Updated**: 2025-10-14
**Maintained by**: Worker 8 (API Server & Finance)
