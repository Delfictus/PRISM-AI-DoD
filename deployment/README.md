# PRISM-AI API Server Deployment Guide

**Worker 8 - Production Deployment Infrastructure**

## Overview

Complete production-ready deployment infrastructure for PRISM-AI API Server with:
- **Docker**: Multi-stage builds with CUDA support
- **Kubernetes**: Full manifests with GPU scheduling
- **CI/CD**: Automated build, test, and deployment pipelines
- **Monitoring**: Prometheus, Grafana, alerts
- **Security**: RBAC, NetworkPolicy, secrets management
- **Scalability**: HPA, PDB, load balancing

## Quick Start

### Docker Compose (Development)

```bash
# Navigate to deployment directory
cd deployment

# Copy environment file
cp .env.example .env

# Edit .env with your API keys
nano .env

# Start services
docker-compose up -d

# View logs
docker-compose logs -f api-server

# Stop services
docker-compose down
```

Access:
- API Server: http://localhost:8080
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

### Kubernetes (Production)

#### Prerequisites

- Kubernetes cluster (v1.24+)
- kubectl configured
- GPU nodes with NVIDIA drivers
- NVIDIA device plugin installed

#### Deploy

```bash
# Create namespace and secrets
kubectl create namespace prism-ai

# Create secrets (DO NOT commit actual secrets)
kubectl create secret generic prism-api-secrets \
  --from-literal=API_KEY='your-secret-key' \
  --from-literal=OPENAI_API_KEY='sk-...' \
  --from-literal=ANTHROPIC_API_KEY='sk-ant-...' \
  --namespace=prism-ai

# Deploy all resources
kubectl apply -k deployment/k8s/

# Check deployment status
kubectl get pods -n prism-ai -w

# Check service
kubectl get svc -n prism-ai

# Check ingress
kubectl get ingress -n prism-ai
```

#### Verify Deployment

```bash
# Port forward for local testing
kubectl port-forward svc/prism-api-service 8080:80 -n prism-ai

# Test health endpoint
curl http://localhost:8080/health

# Test API endpoint
curl http://localhost:8080/api/v1/pwsa/health
```

## Architecture

### Docker Architecture

```
Dockerfile (Multi-stage)
├── Stage 1: Builder (nvidia/cuda:13.0-devel)
│   ├── Install Rust toolchain
│   ├── Cache dependencies
│   ├── Build release binary
│   └── Optimize for size
└── Stage 2: Runtime (nvidia/cuda:13.0-runtime)
    ├── Minimal runtime dependencies
    ├── Non-root user (prism)
    ├── Health checks
    └── GPU access
```

### Kubernetes Architecture

```
prism-ai namespace
├── Workloads
│   ├── Deployment (3-10 replicas)
│   │   ├── GPU scheduling
│   │   ├── Resource limits
│   │   ├── Health probes
│   │   └── Rolling updates
│   └── HorizontalPodAutoscaler
│       ├── CPU-based (70%)
│       ├── Memory-based (80%)
│       └── Custom metrics
├── Networking
│   ├── Service (ClusterIP)
│   ├── Service (Headless for WebSocket)
│   ├── Ingress (NGINX)
│   │   ├── TLS termination
│   │   ├── Rate limiting
│   │   └── CORS
│   └── NetworkPolicy
│       ├── Ingress rules
│       └── Egress rules
├── Configuration
│   ├── ConfigMap
│   ├── Secret
│   └── ServiceAccount
├── Monitoring
│   ├── ServiceMonitor
│   └── PrometheusRule
│       ├── Error rate alerts
│       ├── Latency alerts
│       ├── Availability alerts
│       └── Resource alerts
└── Resilience
    └── PodDisruptionBudget (min 2 available)
```

## Docker

### Build Image

```bash
# Build locally
docker build -t prism-ai/api-server:latest -f deployment/Dockerfile .

# Build with specific tag
docker build -t prism-ai/api-server:v1.0.0 -f deployment/Dockerfile .

# Build for specific platform
docker buildx build --platform linux/amd64 -t prism-ai/api-server:latest .
```

### Run Container

```bash
# Basic run
docker run -p 8080:8080 \
  -e API_KEY=secret \
  -e RUST_LOG=info \
  prism-ai/api-server:latest

# With GPU
docker run --gpus all \
  -p 8080:8080 \
  -e API_KEY=secret \
  prism-ai/api-server:latest

# With all environment variables
docker run --gpus all \
  -p 8080:8080 \
  --env-file .env \
  prism-ai/api-server:latest
```

### Push to Registry

```bash
# Docker Hub
docker tag prism-ai/api-server:latest username/prism-api-server:latest
docker push username/prism-api-server:latest

# GitHub Container Registry
docker tag prism-ai/api-server:latest ghcr.io/username/prism-api-server:latest
docker push ghcr.io/username/prism-api-server:latest
```

## Kubernetes

### Configuration

#### Update Secrets

```bash
# Create/update API key
kubectl create secret generic prism-api-secrets \
  --from-literal=API_KEY='new-secret-key' \
  --namespace=prism-ai \
  --dry-run=client -o yaml | kubectl apply -f -

# From files
kubectl create secret generic prism-api-secrets \
  --from-file=openai-key=./secrets/openai.txt \
  --namespace=prism-ai
```

#### Update ConfigMap

```bash
kubectl create configmap prism-api-config \
  --from-literal=RUST_LOG=debug \
  --namespace=prism-ai \
  --dry-run=client -o yaml | kubectl apply -f -
```

### Scaling

#### Manual Scaling

```bash
# Scale to 5 replicas
kubectl scale deployment prism-api-server --replicas=5 -n prism-ai

# Check status
kubectl get pods -n prism-ai -l app=prism-api-server
```

#### Auto-scaling

HPA is configured with:
- Min replicas: 3
- Max replicas: 10
- CPU target: 70%
- Memory target: 80%

```bash
# Check HPA status
kubectl get hpa -n prism-ai

# Describe HPA
kubectl describe hpa prism-api-hpa -n prism-ai
```

### Updates and Rollbacks

#### Rolling Update

```bash
# Update image
kubectl set image deployment/prism-api-server \
  api-server=prism-ai/api-server:v1.1.0 \
  -n prism-ai

# Watch rollout
kubectl rollout status deployment/prism-api-server -n prism-ai

# Check history
kubectl rollout history deployment/prism-api-server -n prism-ai
```

#### Rollback

```bash
# Rollback to previous version
kubectl rollout undo deployment/prism-api-server -n prism-ai

# Rollback to specific revision
kubectl rollout undo deployment/prism-api-server --to-revision=2 -n prism-ai
```

### Troubleshooting

#### Check Pods

```bash
# List pods
kubectl get pods -n prism-ai

# Describe pod
kubectl describe pod <pod-name> -n prism-ai

# View logs
kubectl logs <pod-name> -n prism-ai

# Follow logs
kubectl logs -f <pod-name> -n prism-ai

# Previous container logs
kubectl logs <pod-name> -n prism-ai --previous
```

#### Debug Pod

```bash
# Shell into pod
kubectl exec -it <pod-name> -n prism-ai -- /bin/sh

# Run command
kubectl exec <pod-name> -n prism-ai -- curl localhost:8080/health
```

#### Check Events

```bash
# All events in namespace
kubectl get events -n prism-ai --sort-by='.lastTimestamp'

# Events for specific pod
kubectl describe pod <pod-name> -n prism-ai | grep -A 10 Events
```

#### Check Resources

```bash
# Resource usage
kubectl top pods -n prism-ai
kubectl top nodes

# GPU usage
kubectl describe node <node-name> | grep -A 10 nvidia
```

## CI/CD

### GitHub Actions

Three workflows configured:

#### 1. CI (ci.yml)
- **Trigger**: Push, Pull Request
- **Steps**:
  - Format check (rustfmt)
  - Lint check (clippy)
  - Build & test
  - Security audit
  - Documentation check

#### 2. CD (cd.yml)
- **Trigger**: Push to worker-8-finance-deploy, Tags
- **Steps**:
  - Build Docker image
  - Push to registry
  - Deploy to staging
  - Deploy to production (on tags)

#### 3. Release (release.yml)
- **Trigger**: Version tags (v*.*.*)
- **Steps**:
  - Generate changelog
  - Create GitHub release
  - Build binaries for multiple targets
  - Upload release assets

### Secrets Required

Add these to GitHub repository settings:

```bash
# Kubernetes access
KUBE_CONFIG_STAGING
KUBE_CONFIG_PRODUCTION

# Container registry
GITHUB_TOKEN (automatic)

# Notifications (optional)
SLACK_WEBHOOK
```

### Manual Trigger

```bash
# Trigger workflow manually
gh workflow run cd.yml --ref worker-8-finance-deploy
```

## Monitoring

### Prometheus

Access: http://localhost:9090 (docker-compose) or via Ingress

#### Queries

```promql
# Request rate
rate(http_requests_total[5m])

# Error rate
sum(rate(http_requests_total{status=~"5.."}[5m])) /
sum(rate(http_requests_total[5m]))

# Latency (P95)
histogram_quantile(0.95,
  rate(http_request_duration_seconds_bucket[5m])
)

# GPU utilization
nvidia_gpu_duty_cycle
```

### Grafana

Access: http://localhost:3000 (docker-compose)
- Username: admin
- Password: admin (change via GRAFANA_PASSWORD env)

#### Dashboards

Import pre-built dashboards:
1. Kubernetes cluster monitoring
2. NVIDIA GPU monitoring
3. Application metrics

### Alerts

Configured in `servicemonitor.yaml`:
- High error rate (>5%)
- High latency (P95 > 1s)
- Low availability (<80%)
- High memory usage (>90%)
- GPU unavailable

## Security

### Best Practices

1. **Secrets Management**
   - Never commit secrets to git
   - Use external secret managers (Vault, AWS Secrets Manager)
   - Rotate secrets regularly

2. **Network Security**
   - NetworkPolicy restricts traffic
   - TLS for external communication
   - mTLS for internal (optional)

3. **RBAC**
   - Minimal permissions via ServiceAccount
   - Role limits API access
   - Audit logging enabled

4. **Container Security**
   - Non-root user (UID 1000)
   - Read-only root filesystem (optional)
   - Security context constraints

5. **API Security**
   - API key authentication
   - Rate limiting
   - Input validation

### Security Scanning

```bash
# Scan Docker image
docker scan prism-ai/api-server:latest

# Scan with Trivy
trivy image prism-ai/api-server:latest

# Scan Kubernetes manifests
kubesec scan deployment/k8s/deployment.yaml
```

## Performance Tuning

### Docker

```dockerfile
# Optimize binary size
ENV RUSTFLAGS="-C target-cpu=native"
RUN cargo build --release --features api_server

# Multi-threaded compilation
ENV CARGO_BUILD_JOBS=8
```

### Kubernetes

```yaml
# Resource requests/limits
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"

# Topology spread
topologySpreadConstraints:
- maxSkew: 1
  topologyKey: kubernetes.io/hostname
  whenUnsatisfiable: DoNotSchedule
```

### Application

```bash
# Tokio worker threads
TOKIO_WORKER_THREADS=8

# Connection pool size
MAX_BLOCKING_THREADS=512

# Logging level
RUST_LOG=info  # or warn in production
```

## Cost Optimization

### GPU Scheduling

```yaml
# Time-slicing for GPU sharing
nodeSelector:
  nvidia.com/gpu.product: NVIDIA-GeForce-RTX-4090

# Fractional GPU
resources:
  limits:
    nvidia.com/gpu: 0.5  # Half GPU
```

### Spot/Preemptible Instances

```yaml
# Tolerate spot instances
tolerations:
- key: cloud.google.com/gke-preemptible
  operator: Exists
- key: eks.amazonaws.com/capacityType
  value: SPOT
```

## Backup and Disaster Recovery

### State Backup

```bash
# Backup ConfigMaps and Secrets
kubectl get configmap -n prism-ai -o yaml > backup-configmaps.yaml
kubectl get secret -n prism-ai -o yaml > backup-secrets.yaml

# Backup entire namespace
kubectl get all -n prism-ai -o yaml > backup-namespace.yaml
```

### Disaster Recovery

```bash
# Restore from backup
kubectl apply -f backup-configmaps.yaml
kubectl apply -f backup-secrets.yaml
kubectl apply -k deployment/k8s/
```

## Support

For issues or questions:
1. Check logs: `kubectl logs <pod> -n prism-ai`
2. Check events: `kubectl get events -n prism-ai`
3. Check metrics: Grafana dashboards
4. GitHub Issues: Report bugs/feature requests

## References

- [API Documentation](../03-Source-Code/src/api_server/README.md)
- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/configuration/overview/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/getting-started.html)

---

**Worker 8 - Deployment Complete**
