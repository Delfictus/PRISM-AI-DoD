# Worker 1 - AI Core & Time Series

**Your Branch**: `worker-1-ai-core`
**Your Time**: 280 hours (7 weeks)
**Your Focus**: Transfer Entropy, Active Inference, Time Series

---

## YOUR RESPONSIBILITIES

### **Weeks 1-3: Transfer Entropy** (120h)
- Full KSG implementation with GPU
- Time-delay embedding
- k-NN search on GPU
- Histogram → MI → Conditional MI pipeline

### **Weeks 4-5: Active Inference** (90h)
- Hierarchical belief propagation
- Advanced policy search
- GPU message passing

### **Weeks 6-7: Time Series** (50h)
- ARIMA on GPU
- LSTM/GRU forecasting
- Uncertainty quantification

### **Week 7: Integration** (20h)
- Testing and validation
- Documentation

---

## YOUR FILES (Exclusive Ownership)

**You OWN and can freely edit**:
```
src/orchestration/routing/
├── te_embedding_gpu.rs (CREATE)
├── gpu_kdtree.rs (CREATE)
├── ksg_transfer_entropy_gpu.rs (CREATE)
└── gpu_transfer_entropy_router.rs (ENHANCE)

src/active_inference/
├── hierarchical_inference_gpu.rs (CREATE)
├── policy_search_gpu.rs (CREATE)
└── [all files] (ENHANCE)

src/time_series/ (CREATE ENTIRE)
├── arima_gpu.rs
├── lstm_forecaster.rs
└── uncertainty.rs
```

**You READ ONLY** (use but don't edit):
```
src/gpu/kernel_executor.rs  # Use Worker 2's kernels
```

---

## KERNELS YOU'LL USE (From Worker 2)

**Week 1-2**: Foundation
- `time_delayed_embedding`
- `histogram_2d`
- `mutual_information`

**Week 3-4**: Advanced TE
- `digamma_vector` (request if not ready)
- `ksg_te_formula` (request if not ready)
- `conditional_entropy`

**Week 6**: Time Series
- `ar_forecast` (request from Worker 2)
- `lstm_cell` (request from Worker 2)
- `kalman_filter` (request from Worker 2)

---

## HOW TO REQUEST KERNELS

Create GitHub issue:
```
Title: [KERNEL] Digamma function for KSG TE
Label: worker-2, priority-high

Description:
Need GPU kernel for digamma function ψ(x)
Input: float* n (neighbor counts)
Output: float* psi_n (digamma values)
Formula: ψ(x) ≈ log(x) - 1/(2x) - 1/(12x²)
Blocks: KSG TE Step 1.3.3
```

---

## DAILY COMMANDS

**Morning**:
```bash
cd /home/diddy/Desktop/PRISM-Worker-1
git pull origin worker-1-ai-core
git merge parallel-development
cargo build --features cuda
```

**Work**:
```bash
# Edit your files
# Build and test frequently
cargo test --lib src/orchestration::routing
cargo test --lib src/active_inference
```

**Evening**:
```bash
git add -A
git commit -m "feat: [what you did]"
git push origin worker-1-ai-core
```

---

## SUCCESS CRITERIA

- [ ] KSG TE: <5% error vs JIDT
- [ ] KSG TE: <100ms for 1000 variables
- [ ] Active Inference: <1ms decision time
- [ ] Time Series: RMSE <5% on validation
- [ ] All tests passing

---

## DEPENDENCIES

**You DEPEND ON**:
- Worker 2 for GPU kernels

**Others DEPEND ON YOU**:
- Worker 5 uses your TE for GNN
- Workers 3,7 use your time series
- Workers 3,4,7 use your Active Inference

---

**Read**: `.obsidian-vault/8_WORKER_ENHANCED_PLAN.md` for full context
**Your detailed tasks**: See WORK ALLOCATION section for Worker 1