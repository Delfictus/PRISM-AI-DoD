# TSP Interactive Demo - Implementation Plan

**Goal:** Interactive web-based TSP demo with tunable difficulty and real-time visualization
**Platform:** Docker container + Web UI, deployable to Google Cloud
**Timeline:** 4-6 days

---

## 🎯 Demo Overview

### What It Does
An interactive Traveling Salesman Problem demonstration that:
1. Lets users tune problem difficulty (10 to 20,000 cities)
2. Visualizes route optimization in real-time
3. Shows GPU vs CPU performance comparison
4. Displays PRISM-AI's advantage over classical algorithms
5. Generates shareable performance reports
6. Runs in browser with GPU backend

### Demo Flow
```
Web UI (Difficulty Slider) → Backend API → PRISM-AI TSP Solver → Real-time Updates → Visualization
                                                    ↓
                                              GPU Acceleration
```

### Key Features
- **Tunable Difficulty:** 10 → 20,000 cities with slider
- **Real-time Visualization:** See route improvement live
- **Performance Comparison:** PRISM vs Classical (Greedy, 2-opt)
- **GPU Metrics:** Live GPU utilization display
- **Interactive:** Click to regenerate, change difficulty
- **Shareable:** Export results as report

---

## 🎚️ Difficulty Levels

### Difficulty Parameters

| Level | Cities | Type | Est. Time | GPU Mem | Complexity | Visual |
|-------|--------|------|-----------|---------|------------|--------|
| **Trivial** | 10-20 | Random | <1s | <10MB | Easy to solve | ✅ Clear |
| **Easy** | 50-100 | Geometric | 1-5s | <50MB | Good visual | ✅ Clear |
| **Medium** | 200-500 | Clustered | 5-30s | <200MB | Interesting | ✅ Visible |
| **Hard** | 1,000-2,000 | Random | 30-120s | <500MB | Challenging | ⚠️ Dense |
| **Extreme** | 5,000-10,000 | Sparse | 2-10min | 1-4GB | Very hard | ⚠️ Very dense |
| **Maximum** | 15,000-20,000 | Benchmark | 10-30min | 4-8GB | Research-scale | ❌ Too dense |

### Difficulty Tuning Options

**1. Number of Cities**
- Slider: 10 to 20,000
- Logarithmic scale for better control
- Presets: 10, 50, 100, 500, 1K, 5K, 10K, 20K

**2. City Distribution**
```rust
enum CityDistribution {
    Random,           // Uniformly random (hardest)
    Geometric,        // 2D plane, Euclidean (medium)
    Clustered,        // Cities in clusters (easier)
    Grid,             // Regular grid (easiest)
    Circle,           // Around circle (very easy)
    RealWorld,        // From actual data (varies)
}
```

**3. Problem Hardness**
```rust
struct DifficultyConfig {
    n_cities: usize,
    distribution: CityDistribution,
    distance_metric: DistanceMetric,  // Euclidean, Manhattan, etc.
    symmetry: bool,                   // Symmetric vs asymmetric
    obstacles: bool,                  // Add barriers (harder)
}
```

**4. Solver Configuration**
```rust
struct SolverConfig {
    max_iterations: usize,     // More = better but slower
    use_gpu: bool,             // GPU vs CPU comparison
    algorithm: Algorithm,      // PRISM vs Greedy vs 2-opt
    early_stop_threshold: f64, // Stop when good enough
}
```

---

## 📋 Implementation Tasks

### Phase 1: Backend Service (Day 1-2, ~10 hours)

#### Task 1.1: Create TSP API Server
**File:** `tsp-demo/src/main.rs`
**Effort:** 3 hours

**Responsibilities:**
- [ ] Create Axum/Actix web server
- [ ] REST API endpoints (start, status, results)
- [ ] WebSocket for real-time updates
- [ ] CORS configuration
- [ ] Error handling

**API Endpoints:**
```rust
POST   /api/solve      // Start TSP optimization
GET    /api/status/:id // Get progress
GET    /api/results/:id // Get final results
WS     /api/live/:id   // Real-time updates
GET    /api/presets    // Get difficulty presets
```

#### Task 1.2: Implement Difficulty Engine
**File:** `tsp-demo/src/difficulty.rs`
**Effort:** 2 hours

**Responsibilities:**
- [ ] City generation for each distribution type
- [ ] Distance matrix computation
- [ ] Difficulty scoring algorithm
- [ ] Preset configurations

**Code:**
```rust
pub struct DifficultyEngine;

impl DifficultyEngine {
    pub fn generate_problem(config: &DifficultyConfig) -> TSPProblem {
        let cities = match config.distribution {
            CityDistribution::Random => Self::generate_random(config.n_cities),
            CityDistribution::Geometric => Self::generate_geometric(config.n_cities),
            CityDistribution::Clustered => Self::generate_clustered(config.n_cities),
            // ... other types
        };

        TSPProblem {
            cities,
            distance_matrix: Self::compute_distances(&cities, config.distance_metric),
            difficulty_score: Self::estimate_difficulty(config),
        }
    }

    fn estimate_difficulty(config: &DifficultyConfig) -> f64 {
        // Difficulty = f(n_cities, distribution, symmetry, obstacles)
        let base = (config.n_cities as f64).log10();
        let distribution_factor = match config.distribution {
            CityDistribution::Circle => 0.5,
            CityDistribution::Grid => 0.7,
            CityDistribution::Geometric => 1.0,
            CityDistribution::Clustered => 1.2,
            CityDistribution::Random => 1.5,
        };
        base * distribution_factor
    }
}
```

#### Task 1.3: Integrate PRISM-AI TSP Solver
**File:** `tsp-demo/src/solver.rs`
**Effort:** 2 hours

**Responsibilities:**
- [ ] Wrap GpuTspSolver with progress tracking
- [ ] Implement callback system for live updates
- [ ] Add comparison with classical algorithms
- [ ] Collect performance metrics

**Code:**
```rust
pub struct TSPSolverService {
    gpu_solver: Option<GpuTspSolver>,
    progress_tx: UnboundedSender<ProgressUpdate>,
}

impl TSPSolverService {
    pub async fn solve(&mut self, problem: TSPProblem, config: SolverConfig) -> Result<TSPResult> {
        // Convert problem to coupling matrix
        let coupling = problem_to_coupling_matrix(&problem);

        // Initialize PRISM-AI solver
        let mut solver = GpuTspSolver::new(&coupling)?;

        // Run optimization with progress callbacks
        for iteration in 0..config.max_iterations {
            solver.optimize_step()?;

            // Send progress update
            if iteration % 10 == 0 {
                self.send_progress(iteration, &solver)?;
            }

            // Check early stopping
            if solver.improvement_ratio() < config.early_stop_threshold {
                break;
            }
        }

        Ok(TSPResult {
            tour: solver.get_tour(),
            length: solver.get_tour_length(),
            iterations: solver.iterations(),
            gpu_time_ms: solver.gpu_time(),
            // ...
        })
    }
}
```

#### Task 1.4: Add Performance Comparison
**File:** `tsp-demo/src/comparison.rs`
**Effort:** 2 hours

**Responsibilities:**
- [ ] Implement classical baseline (Greedy + 2-opt)
- [ ] Run side-by-side comparison
- [ ] Collect timing metrics
- [ ] Calculate speedup ratios

#### Task 1.5: Fix Example Imports (BLOCKER)
**Files:** All `examples/*.rs`
**Effort:** 1 hour

**Must complete first:**
- [ ] Update imports: `active_inference_platform` → `prism_ai`
- [ ] Update imports: `neuromorphic_quantum_platform` → `prism_ai`
- [ ] Test compilation

---

### Phase 2: Frontend UI (Day 2-3, ~10 hours)

#### Task 2.1: Create Web Interface
**File:** `tsp-demo/frontend/index.html`
**Effort:** 3 hours

**Responsibilities:**
- [ ] Create single-page app (HTML/CSS/JS)
- [ ] Difficulty controls (sliders, dropdowns)
- [ ] Start/Stop buttons
- [ ] Real-time status display
- [ ] Responsive design

**UI Layout:**
```
┌─────────────────────────────────────────────┐
│  PRISM-AI TSP Demo                          │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│                                             │
│  🎚️ Difficulty Controls                     │
│  ┌─────────────────────────────────────┐   │
│  │ Cities: [====●=========] 500        │   │
│  │ Distribution: [Geometric ▼]         │   │
│  │ Algorithm: [PRISM-AI ▼]             │   │
│  │                                     │   │
│  │ [🚀 Start Optimization]             │   │
│  └─────────────────────────────────────┘   │
│                                             │
│  📊 Visualization                           │
│  ┌─────────────────────────────────────┐   │
│  │                                     │   │
│  │    [Interactive Map Canvas]         │   │
│  │    • Cities as dots                 │   │
│  │    • Route as lines                 │   │
│  │    • Real-time updates              │   │
│  │                                     │   │
│  └─────────────────────────────────────┘   │
│                                             │
│  📈 Performance Metrics                     │
│  ┌──────────────┬──────────────────────┐   │
│  │ Current:     │ Best:                │   │
│  │ 12,345 km    │ 11,234 km            │   │
│  │              │                      │   │
│  │ GPU: 94%     │ Iteration: 234/1000  │   │
│  └──────────────┴──────────────────────┘   │
│                                             │
│  🏆 Comparison                              │
│  ┌─────────────────────────────────────┐   │
│  │ PRISM-AI:  11,234 km (42s)          │   │
│  │ Classical: 14,567 km (125s)         │   │
│  │ Speedup:   2.97x faster, 22% better │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

#### Task 2.2: Route Visualization
**File:** `tsp-demo/frontend/visualization.js`
**Effort:** 3 hours

**Technologies:**
- Canvas API or D3.js for rendering
- WebGL for large problems (5K+ cities)

**Responsibilities:**
- [ ] Render cities as points
- [ ] Draw current route
- [ ] Animate route improvements
- [ ] Zoom/pan for large problems
- [ ] Color code by improvement
- [ ] Show best route in different color

#### Task 2.3: Real-time Updates via WebSocket
**File:** `tsp-demo/frontend/websocket.js`
**Effort:** 2 hours

**Responsibilities:**
- [ ] Connect to backend WebSocket
- [ ] Receive progress updates
- [ ] Update visualization in real-time
- [ ] Update metrics display
- [ ] Handle disconnections
- [ ] Reconnect logic

#### Task 2.4: Performance Charts
**File:** `tsp-demo/frontend/charts.js`
**Effort:** 2 hours

**Technologies:** Chart.js or Plotly.js

**Responsibilities:**
- [ ] Tour length convergence curve
- [ ] GPU utilization over time
- [ ] Comparison bar chart (PRISM vs Classical)
- [ ] Speedup ratio display
- [ ] Iteration time histogram

---

### Phase 3: Containerization (Day 3-4, ~8 hours)

#### Task 3.1: Create Dockerfile
**File:** `tsp-demo/Dockerfile`
**Effort:** 2 hours

**Responsibilities:**
- [ ] Multi-stage build (Rust backend + static frontend)
- [ ] CUDA 12.8 runtime
- [ ] Copy PTX kernels
- [ ] Nginx for serving frontend
- [ ] Backend service startup

**Dockerfile:**
```dockerfile
# Stage 1: Build Rust backend
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS rust-builder

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build
COPY Cargo.* ./
COPY src ./src
COPY examples ./examples
COPY cuda ./cuda

RUN cargo build --release --bin tsp-demo-server

# Stage 2: Build frontend
FROM node:18 AS frontend-builder

WORKDIR /build
COPY tsp-demo/frontend ./
RUN npm install
RUN npm run build

# Stage 3: Runtime
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Install nginx
RUN apt-get update && apt-get install -y nginx && rm -rf /var/lib/apt/lists/*

# Copy backend
COPY --from=rust-builder /build/target/release/tsp-demo-server /app/
COPY --from=rust-builder /build/target/ptx/*.ptx /app/ptx/

# Copy frontend
COPY --from=frontend-builder /build/dist /var/www/html

# Nginx config
COPY tsp-demo/nginx.conf /etc/nginx/nginx.conf

# Start script
COPY tsp-demo/start.sh /app/
RUN chmod +x /app/start.sh

EXPOSE 80 8080

CMD ["/app/start.sh"]
```

#### Task 3.2: Create Web Server Binary
**File:** `tsp-demo/src/bin/server.rs`
**Effort:** 3 hours

**Framework:** Axum (async Rust web framework)

**Responsibilities:**
- [ ] HTTP server setup
- [ ] WebSocket handler
- [ ] TSP solver integration
- [ ] Progress broadcasting
- [ ] Static file serving (fallback)

**Code Structure:**
```rust
use axum::{
    Router, Json,
    extract::{State, WebSocketUpgrade},
    response::IntoResponse,
};
use tokio::sync::broadcast;

#[derive(Clone)]
struct AppState {
    solver_pool: Arc<Mutex<Vec<TSPSolverService>>>,
    progress_tx: broadcast::Sender<ProgressUpdate>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let app = Router::new()
        .route("/api/solve", post(start_solve))
        .route("/api/status/:id", get(get_status))
        .route("/api/live/:id", get(websocket_handler))
        .with_state(app_state);

    axum::Server::bind(&"0.0.0.0:8080".parse()?)
        .serve(app.into_make_service())
        .await?;

    Ok(())
}
```

#### Task 3.3: Add Docker Compose
**File:** `tsp-demo/docker-compose.yml`
**Effort:** 1 hour

**Responsibilities:**
- [ ] Define services (backend + nginx)
- [ ] GPU configuration
- [ ] Volume mounts
- [ ] Port mapping

#### Task 3.4: Test Locally
**Effort:** 2 hours

**Responsibilities:**
- [ ] Build Docker image
- [ ] Test with GPU
- [ ] Verify web UI works
- [ ] Test all difficulty levels
- [ ] Check real-time updates

---

### Phase 4: Google Cloud Deployment (Day 4-5, ~6 hours)

#### Task 4.1: Cloud Run Configuration
**File:** `tsp-demo/gcp/cloudrun.yaml`
**Effort:** 2 hours

**Challenges:**
- Cloud Run may not support persistent WebSocket connections
- Alternative: Use Cloud Compute Engine or GKE

**Options:**

**Option A: Cloud Run (simpler, stateless)**
```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: prism-tsp-demo
spec:
  template:
    spec:
      containers:
      - image: gcr.io/PROJECT/prism-tsp-demo:latest
        ports:
        - containerPort: 80
        resources:
          limits:
            memory: 16Gi
            nvidia.com/gpu: "1"
```

**Option B: GKE with GPU (better for WebSocket)**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prism-tsp-demo
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: tsp-demo
        image: gcr.io/PROJECT/prism-tsp-demo:latest
        resources:
          limits:
            nvidia.com/gpu: 1
---
apiVersion: v1
kind: Service
metadata:
  name: prism-tsp-demo
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 80
```

#### Task 4.2: Deployment Scripts
**File:** `tsp-demo/gcp/deploy.sh`
**Effort:** 1 hour

**Responsibilities:**
- [ ] Build and push image
- [ ] Deploy to GKE/Cloud Run
- [ ] Configure load balancer
- [ ] Set up DNS (optional)

#### Task 4.3: Add Monitoring
**File:** Part of server
**Effort:** 2 hours

**Responsibilities:**
- [ ] Structured logging (JSON)
- [ ] Metrics export (Prometheus format)
- [ ] GPU metrics collection
- [ ] Performance tracking
- [ ] Error reporting

#### Task 4.4: Test on GCP
**Effort:** 1 hour

**Responsibilities:**
- [ ] Deploy to GCP
- [ ] Test from multiple browsers
- [ ] Verify GPU usage
- [ ] Check WebSocket stability
- [ ] Load testing

---

### Phase 5: Visualization & UX (Day 5-6, ~8 hours)

#### Task 5.1: Advanced Route Visualization
**File:** `tsp-demo/frontend/advanced-viz.js`
**Effort:** 3 hours

**Responsibilities:**
- [ ] Smooth animations between route updates
- [ ] Highlight improvements (green) vs unchanged (gray)
- [ ] Show algorithm progress (which edges being evaluated)
- [ ] Cluster visualization for clustered problems
- [ ] Heatmap overlay for difficulty

**Features:**
- Color gradient showing tour order
- Animated route transitions
- City labels on hover
- Distance display on edge hover
- Zoom to fit on problem change

#### Task 5.2: Interactive Controls
**File:** `tsp-demo/frontend/controls.js`
**Effort:** 2 hours

**Responsibilities:**
- [ ] Difficulty slider with live preview
- [ ] Distribution selector with thumbnails
- [ ] Algorithm comparison toggles
- [ ] Speed control (iteration delay)
- [ ] Export report button
- [ ] Share link generation

#### Task 5.3: Performance Dashboard
**File:** `tsp-demo/frontend/dashboard.js`
**Effort:** 2 hours

**Responsibilities:**
- [ ] Live GPU utilization gauge
- [ ] Convergence chart (updating in real-time)
- [ ] Speedup calculator
- [ ] Quality metrics (% improvement over initial)
- [ ] Time remaining estimate

#### Task 5.4: Report Generation
**File:** `tsp-demo/frontend/report.js`
**Effort:** 1 hour

**Responsibilities:**
- [ ] Generate HTML report with results
- [ ] Include static route visualization
- [ ] Add performance comparison
- [ ] Export as PDF (optional)
- [ ] Share via URL

---

### Phase 6: Presets & Testing (Day 6, ~6 hours)

#### Task 6.1: Create Difficulty Presets
**File:** `tsp-demo/presets.json`
**Effort:** 2 hours

**Presets:**

```json
{
  "quick_demo": {
    "name": "Quick Demo (30 seconds)",
    "cities": 100,
    "distribution": "geometric",
    "iterations": 500,
    "description": "Fast demonstration for meetings"
  },
  "visual_showcase": {
    "name": "Visual Showcase",
    "cities": 200,
    "distribution": "clustered",
    "iterations": 1000,
    "description": "Best for visualization"
  },
  "performance_test": {
    "name": "Performance Test (5 minutes)",
    "cities": 2000,
    "distribution": "random",
    "iterations": 2000,
    "description": "Show GPU advantage"
  },
  "research_scale": {
    "name": "Research Scale (30 minutes)",
    "cities": 10000,
    "distribution": "geometric",
    "iterations": 5000,
    "description": "Real-world scale problem"
  },
  "maximum_challenge": {
    "name": "Maximum Challenge (1 hour)",
    "cities": 20000,
    "distribution": "random",
    "iterations": 10000,
    "description": "Stress test"
  }
}
```

#### Task 6.2: Add Benchmark Mode
**File:** Part of server
**Effort:** 2 hours

**Responsibilities:**
- [ ] Run full benchmark suite
- [ ] Compare against known optimal solutions
- [ ] Test all difficulty levels
- [ ] Generate benchmark report
- [ ] Validate GPU performance claims

#### Task 6.3: Create Tutorial/Walkthrough
**File:** `tsp-demo/frontend/tutorial.js`
**Effort:** 1 hour

**Responsibilities:**
- [ ] First-time user overlay
- [ ] Explain each control
- [ ] Show example workflow
- [ ] Tips for best results

#### Task 6.4: Final Testing & Polish
**Effort:** 1 hour

**Responsibilities:**
- [ ] Test all presets
- [ ] Verify on different browsers
- [ ] Mobile responsiveness
- [ ] Error handling
- [ ] Performance optimization

---

## 📁 File Structure

```
PRISM-AI/
├── tsp-demo/                           # NEW demo directory
│   ├── src/
│   │   ├── bin/
│   │   │   └── server.rs              # Web server binary
│   │   ├── difficulty.rs              # Difficulty engine
│   │   ├── solver.rs                  # TSP solver service
│   │   ├── comparison.rs              # Algorithm comparison
│   │   └── lib.rs                     # Common utilities
│   ├── frontend/
│   │   ├── index.html                 # Main UI
│   │   ├── style.css                  # Styling
│   │   ├── app.js                     # Main app logic
│   │   ├── visualization.js           # Route rendering
│   │   ├── charts.js                  # Performance charts
│   │   ├── websocket.js               # Real-time connection
│   │   └── tutorial.js                # User walkthrough
│   ├── Dockerfile                     # Container build
│   ├── docker-compose.yml             # Local deployment
│   ├── nginx.conf                     # Nginx configuration
│   ├── start.sh                       # Container startup
│   ├── presets.json                   # Difficulty presets
│   ├── Cargo.toml                     # Demo dependencies
│   └── README.md                      # Demo documentation
├── tsp-demo/gcp/                      # Google Cloud
│   ├── cloudrun.yaml                  # Cloud Run config
│   ├── gke-deployment.yaml            # GKE alternative
│   ├── deploy.sh                      # Deployment script
│   └── monitoring.yaml                # Monitoring config
└── docs/
    └── TSP_DEMO_GUIDE.md              # User documentation
```

---

## 🔧 Technical Specifications

### Backend (Rust + Axum)

**Dependencies:**
```toml
[dependencies]
prism-ai = { path = ".." }
axum = "0.7"
tokio = { version = "1", features = ["full"] }
tokio-tungstenite = "0.21"  # WebSocket
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tower-http = { version = "0.5", features = ["cors"] }
```

**Endpoints:**
- `POST /api/solve` - Start optimization
- `GET /api/status/:id` - Get current progress
- `WS /api/live/:id` - Real-time WebSocket updates
- `GET /api/presets` - List difficulty presets

### Frontend (Vanilla JS + Canvas/D3.js)

**No Framework:** Keep it simple, fast, and lightweight

**Libraries:**
- Chart.js for performance graphs
- Canvas API for route visualization (or D3.js)
- Native WebSocket API

**Features:**
- Responsive design (desktop + tablet)
- Real-time updates (<100ms latency)
- Smooth animations
- Touch support for mobile

### Docker Image

**Base:** `nvidia/cuda:12.8.0-runtime-ubuntu22.04`
**Size Target:** <1.5GB
**Services:**
- Nginx (port 80) - Frontend
- Axum server (port 8080) - Backend API

---

## 🎮 Difficulty System Design

### City Count Slider
```javascript
// Logarithmic scale for better UX
const citySlider = {
    min: 1,        // 10^1 = 10 cities
    max: 4.3,      // 10^4.3 ≈ 20,000 cities
    step: 0.1,
    value: 2.7,    // 10^2.7 ≈ 500 cities (default)

    getCityCount: (value) => Math.round(Math.pow(10, value))
};

// Display: "Cities: 500" with preview of difficulty
```

### Distribution Types

**Visual Previews:** Show thumbnail of each type

1. **Random** (Hardest)
   - Cities scattered uniformly
   - No structure to exploit
   - Difficulty: ⭐⭐⭐⭐⭐

2. **Geometric** (Medium-Hard)
   - Cities on 2D plane
   - Euclidean distances
   - Difficulty: ⭐⭐⭐⭐

3. **Clustered** (Medium)
   - Cities in groups
   - Inter-cluster optimization
   - Difficulty: ⭐⭐⭐

4. **Grid** (Easy)
   - Regular grid pattern
   - Predictable structure
   - Difficulty: ⭐⭐

5. **Circle** (Easiest)
   - Cities around circle
   - Optimal = circumference
   - Difficulty: ⭐

6. **Real-World** (Variable)
   - Actual city coordinates
   - Realistic road distances
   - Difficulty: Varies

### Algorithm Options

**1. PRISM-AI (Full Platform)**
- GPU-accelerated
- Neuromorphic + Quantum coupling
- Best quality, fastest

**2. PRISM-AI (GPU Only)**
- Just GPU 2-opt
- No neuromorphic coupling
- Fast, good quality

**3. Classical (Greedy + 2-opt)**
- CPU baseline
- Greedy construction + iterative improvement
- Slower, lower quality

**4. Comparison Mode**
- Run all three side-by-side
- Show speedup and quality difference

---

## 📊 Real-time Metrics Display

### Main Dashboard Metrics

**Current Performance:**
- Tour length (km or units)
- Current iteration / total
- Elapsed time
- Estimated time remaining
- GPU utilization %
- Memory usage

**Best Solution:**
- Best tour length found
- Improvement over initial (%)
- Iteration where found
- Time to best solution

**GPU Stats:**
- GPU name and memory
- CUDA kernel count
- Kernel execution time
- Memory bandwidth utilization

**Comparison (if enabled):**
- PRISM-AI: length, time
- Classical: length, time
- Speedup: Xx faster
- Quality improvement: Y% better

---

## 🎨 Visualization Features

### Route Display

**For Small Problems (10-500 cities):**
- All cities visible as circles
- Full route drawn
- Animated improvements
- Labels on hover
- Colors: Blue → Red gradient for tour order

**For Medium Problems (500-2,000):**
- Cities as smaller dots
- Route with transparency
- Zoom enabled
- Click to focus on region
- Simplified labels

**For Large Problems (2,000-20,000):**
- Heatmap visualization
- Route as continuous curve
- WebGL for performance
- Statistical summary (can't show all cities)
- Cluster highlights

### Animation Options

**Speed Settings:**
- Real-time (update every iteration)
- Fast (update every 10 iterations)
- Summary only (just show final result)
- Instant (no animation)

**Visual Effects:**
- Highlight swapped edges during 2-opt
- Fade out old route, fade in new
- Pulse effect on improvement
- Progress bar with color gradient

---

## 🏆 Demonstration Scenarios

### Scenario 1: Quick Sales Pitch (30 seconds)
**Preset:** Quick Demo
**Cities:** 100
**Time:** 30 seconds
**Show:** GPU speedup + visual improvement

### Scenario 2: Technical Demo (5 minutes)
**Preset:** Performance Test
**Cities:** 2,000
**Time:** 5 minutes
**Show:** GPU utilization, comparison charts, metrics

### Scenario 3: Research Presentation (30 minutes)
**Preset:** Research Scale
**Cities:** 10,000
**Time:** 30 minutes
**Show:** Scalability, real-world applicability

### Scenario 4: Investor Meeting (2 minutes)
**Preset:** Visual Showcase
**Cities:** 200
**Time:** 2 minutes
**Show:** Beautiful animation, clear improvement, professional UI

---

## 📈 Success Metrics

### Technical
- ✅ Handles 10-20,000 cities
- ✅ Real-time updates (<100ms latency)
- ✅ GPU utilization >80%
- ✅ 2-10x speedup over classical
- ✅ Responsive on mobile/desktop
- ✅ Docker image <1.5GB

### User Experience
- ✅ Intuitive controls
- ✅ Beautiful visualizations
- ✅ Clear performance metrics
- ✅ Works in any browser (Chrome, Firefox, Safari)
- ✅ No installation required
- ✅ Shareable demo links

### Business Impact
- ✅ Impressive to non-technical audience
- ✅ Shows clear competitive advantage
- ✅ Demonstrates GPU value
- ✅ Professional appearance
- ✅ Easy to demonstrate live
- ✅ Cost per demo <$0.10

---

## 🎯 Advantages Over Materials Demo

| Aspect | Materials Demo | TSP Demo |
|--------|----------------|----------|
| **Visual Appeal** | Charts, formulas | ✅ Animated map routes |
| **Interactivity** | Config file | ✅ Real-time sliders |
| **Accessibility** | Technical | ✅ Easy to understand |
| **Demo Time** | 10-30 min | ✅ 30s - 30min (tunable) |
| **Wow Factor** | Good | ✅ Excellent |
| **Explanation** | Complex | ✅ Simple (shortest route) |
| **GPU Showcase** | Yes | ✅ Yes + real-time viz |
| **Web-based** | No | ✅ Yes |

**TSP Demo is more accessible and visually impressive!**

---

## 💰 Cost Analysis

### Development Costs
- **Time:** 4-6 days (28-40 hours)
- **Local testing:** Free (with GPU)
- **GCP testing:** $5-10

### Operational Costs

**Per Demo Session (GKE with T4 GPU):**
- GPU: $0.35/hour × time
- Compute: $0.10/hour × time
- Load Balancer: $0.025/hour

**Cost by Scenario:**
- Quick demo (30s): $0.01
- Technical demo (5min): $0.04
- Research demo (30min): $0.23

**Monthly (if production service):**
- 100 quick demos/day: $30/month
- 10 research demos/day: $70/month

---

## 🚧 Challenges & Solutions

### Challenge 1: Real-time WebSocket at Scale
**Problem:** WebSocket may not work well on Cloud Run
**Solution:** Use GKE instead, or poll-based updates for Cloud Run
**Recommendation:** GKE with GPU node pool

### Challenge 2: Visualization Performance
**Problem:** 20K cities is too many to render smoothly
**Solution:**
- Use WebGL for >2K cities
- Statistical sampling for >10K cities
- Heatmap mode for ultra-large problems

### Challenge 3: GPU Memory for Large Problems
**Problem:** 20K cities = 400M distance matrix (3.2GB)
**Solution:**
- Compute distances on-the-fly
- Sparse representation
- Chunked processing
- Use A100 GPU for max problems

### Challenge 4: Cold Start Latency
**Problem:** First request initializes GPU (~5-10s)
**Solution:**
- Pre-warm instances
- Show "Initializing GPU..." message
- Keep-alive pings
- GPU stays warm between requests

---

## 🎬 Demo User Flow

### Step 1: Landing Page
```
┌─────────────────────────────────────┐
│  🌍 PRISM-AI TSP Demo               │
│                                     │
│  Discover the world's fastest      │
│  TSP solver with GPU acceleration  │
│                                     │
│  [ Try Quick Demo ]  [ Customize ] │
└─────────────────────────────────────┘
```

### Step 2: Configure Problem
```
Select Difficulty:
[ ] Quick (100 cities)
[●] Medium (500 cities)  ← Selected
[ ] Hard (2,000 cities)
[ ] Extreme (10,000 cities)

Distribution: [Geometric ▼]
Algorithm: [PRISM-AI Full ▼]

[🚀 Start Optimization]
```

### Step 3: Watch Optimization
```
┌────────────────────────────────────┐
│ Route Visualization                │
│                                    │
│  [Animated map with route]         │
│                                    │
│  Current: 12,345 km                │
│  Best: 11,234 km (-9%)             │
│  GPU: ████████░░ 89%               │
│  Progress: ████████░░░ 76% (762/1000)│
└────────────────────────────────────┘
```

### Step 4: View Results
```
┌────────────────────────────────────┐
│ ✅ Optimization Complete!          │
│                                    │
│ Final Route: 11,234 km             │
│ Improvement: 22% better than initial│
│ Time: 42 seconds                   │
│ GPU Speedup: 3.2x faster           │
│                                    │
│ [ Download Report ] [ Share ] [Try Again]│
└────────────────────────────────────┘
```

---

## 📊 Report Contents

### Generated Report Sections

**1. Problem Configuration**
- Number of cities: 500
- Distribution: Geometric
- Difficulty score: 3.2/5
- Initial tour length: 14,532 km

**2. Optimization Results**
- Final tour length: 11,234 km
- Improvement: 22.7%
- Iterations: 762
- Time: 42.3 seconds
- GPU utilization: 89% average

**3. Route Visualization**
- Static image of final route
- Comparison: initial vs final

**4. Performance Comparison**
- PRISM-AI: 11,234 km (42s)
- Classical: 13,891 km (138s)
- Speedup: 3.27x faster
- Quality: 19.1% better solution

**5. GPU Metrics**
- GPU: NVIDIA Tesla T4
- CUDA kernels: 23 active
- Memory used: 1.2GB / 16GB
- Kernel time: 38.5s (91% of total)

**6. Algorithm Breakdown**
- Initial construction: Nearest-neighbor (0.8s)
- 2-opt optimization: GPU-parallel (41.5s)
- Number of swaps evaluated: 1.2M
- Improvements found: 847

---

## 🎯 Next Steps

### Immediate (This Session)
1. Review this plan
2. Decide: TSP demo first OR Materials demo first?
3. Fix example imports (BLOCKER for both)

### Day 1 (TSP Demo)
1. Create tsp-demo directory structure
2. Implement backend server
3. Create basic frontend
4. Test locally

### Day 2-3
5. Add real-time visualization
6. Implement difficulty system
7. Containerize
8. Test with Docker + GPU

### Day 4-5
9. Deploy to GCP
10. Add presets
11. Polish UI/UX
12. Create documentation

### Day 6
13. Final testing
14. Generate sample reports
15. Prepare demo walkthrough

---

## 🌟 Why This Demo is Powerful

### For Investors
- **Visual:** Watch optimization happen in real-time
- **Interactive:** They can try different difficulties
- **Measurable:** Clear speedup numbers (3-10x)
- **Accessible:** Everyone understands "shortest route"
- **Impressive:** Handle 20,000 cities in minutes

### For Customers
- **Relatable:** Logistics, delivery routes, field service
- **Practical:** Solves real business problems
- **Scalable:** Works from 10 to 20,000 cities
- **Fast:** Real-world problems in seconds/minutes
- **Confident:** Mathematical guarantees on solutions

### For Engineers
- **Technical depth:** GPU acceleration visible
- **Architecture:** Clean separation of concerns
- **Performance:** Measurable, reproducible
- **Extensible:** Easy to add new features
- **Well-tested:** Uses battle-tested TSP solver

---

## 🔗 Related Documents

- [[Materials Discovery Demo Plan]] - Alternative demo
- [[Use Cases and Responsibilities]] - Library integration
- [[Module Reference]] - TSP solver documentation
- [[Architecture Overview]] - System design

---

## 📝 Implementation Priorities

### Must Have (MVP)
- [ ] Working backend with TSP solver
- [ ] Basic web UI with controls
- [ ] Route visualization
- [ ] 3 difficulty presets
- [ ] Docker container
- [ ] Works locally with GPU

### Should Have (Production)
- [ ] Real-time WebSocket updates
- [ ] Performance comparison
- [ ] 5 presets
- [ ] GCP deployment
- [ ] Report generation
- [ ] Mobile responsive

### Nice to Have (Polish)
- [ ] Tutorial walkthrough
- [ ] Shareable demo links
- [ ] Benchmark mode
- [ ] Advanced visualizations (heatmaps)
- [ ] Custom city placement (click to add)

---

## 💡 Advanced Features (Future)

### Interactive Editing
- Click to add cities
- Drag to reposition
- Delete cities
- Manual route editing
- Compare manual vs optimal

### Multi-Algorithm Race
- Run 3 algorithms simultaneously
- Side-by-side visualization
- Real-time performance comparison
- Winner announcement

### Leaderboard
- Track best solutions
- Compare user submissions
- Public leaderboard
- Challenge mode

### Educational Mode
- Explain algorithm step-by-step
- Highlight what's happening
- Interactive tutorials
- Learn TSP complexity

---

## 🎓 Educational Value

### Teaches Concepts
1. **TSP Complexity:** NP-hard, combinatorial explosion
2. **GPU Acceleration:** Why parallel matters
3. **Heuristics:** Greedy, 2-opt, metaheuristics
4. **Optimization:** Local minima, convergence
5. **Performance:** Time/space tradeoffs

### Demo Talking Points
- "This 1,000 city problem has 10^2567 possible solutions"
- "Our GPU evaluates 1 million swaps per second"
- "Watch how the route gets shorter in real-time"
- "Classical takes 3x longer and finds worse solutions"
- "Scale to real-world problems with 20,000 locations"

---

## 📅 Timeline

### Day 1: Backend + Basic UI (8 hours)
- Morning: Server, API, TSP integration
- Afternoon: Frontend HTML/CSS, basic visualization

### Day 2: Interactivity (8 hours)
- Morning: WebSocket, real-time updates
- Afternoon: Difficulty controls, presets

### Day 3: Containerization (6 hours)
- Morning: Dockerfile, compose
- Afternoon: Local testing

### Day 4: Cloud Deployment (6 hours)
- Morning: GCP setup, deployment
- Afternoon: Testing, monitoring

### Day 5: Visualization Polish (6 hours)
- Morning: Advanced viz, charts
- Afternoon: UX improvements

### Day 6: Testing & Documentation (6 hours)
- Morning: Full testing suite
- Afternoon: Documentation, guides

**Total:** 4-6 days (28-40 hours)

---

## 🚀 Quick Start (After Build)

### Local Development
```bash
cd tsp-demo
cargo build --release
cd frontend && npm run dev
# Open http://localhost:3000
```

### Docker
```bash
docker-compose up
# Open http://localhost
```

### Google Cloud
```bash
./gcp/deploy.sh
# Returns: https://prism-tsp-demo-xxx.run.app
```

---

*Plan created: 2025-10-04*
*Estimated completion: 2025-10-10*
