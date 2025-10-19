# Quick Start: Choose Your Deployment Method

You have **3 deployment options** based on your hardware.

---

## 🚀 **Option 1: ULTRA-FAST (8x B200 - RECOMMENDED FOR YOU)** ⚡

**Your Hardware:** 8x B200 (1,440 GB VRAM)
**Training Time:** 5-10 minutes
**Cost:** ~$2-5

### Steps:
```bash
# 1. Build Docker image (ONE-TIME)
cd /home/diddy/Desktop/PRISM-AI-DoD/python/gnn_training
bash build_and_push.sh

# 2. Deploy to your B200 instance
# - Custom Container: delfictus/prism-ai-world-record:latest

# 3. Upload training data
scp training_data.tar.gz root@<pod-ip>:/workspace/

# 4. Run MULTI-GPU training
ssh root@<pod-ip>
tar -xzf training_data.tar.gz
bash run_multigpu.sh
```

**Automatic:**
- Uses all 8 GPUs simultaneously
- Effective batch size: 512 (64 per GPU × 8)
- ~8x speedup over single GPU
- Training complete in 5-10 minutes

**See:** `B200_DEPLOYMENT.md` for detailed guide

---

## ⚡ **Option 2: Fast (Single GPU - H100/B200)**

**Best for:** H100 or single B200 GPU
**Training Time:** 15-30 minutes (H100) or 10-20 minutes (B200)

### Steps:
```bash
# 1. Build Docker image
bash build_and_push.sh

# 2. Deploy RunPod
# - Custom Container: delfictus/prism-ai-world-record:latest
# - GPU: H100 80GB or 1x B200

# 3. Upload & run
scp training_data.tar.gz root@<pod-ip>:/workspace/
ssh root@<pod-ip>
tar -xzf training_data.tar.gz
bash run.sh
```

**See:** `DOCKER_DEPLOYMENT.md` for detailed guide

---

## 📦 **Option 3: Simple (No Docker)**

**Best for:** Quick testing, no Docker setup

### Steps:
```bash
# 1. Create package
bash python/gnn_training/package_for_runpod.sh

# 2. Deploy RunPod with PyTorch template
# - Template: PyTorch 2.1 CUDA 12.1

# 3. Upload & run
# Upload prism_gnn_runpod.tar.gz
tar -xzf prism_gnn_runpod.tar.gz
cd runpod_package/gnn_training
bash run.sh
```

**See:** `RUNPOD_DEPLOYMENT.md` for detailed guide

---

## 🎯 **Recommendation**

### **You Have 8x B200 → Use Option 1** ✅

**Why:**
- ✅ Fastest: 5-10 minutes (vs 30-90 min)
- ✅ Cheapest: $2-5 (faster = less cost)
- ✅ Professional: Multi-GPU training
- ✅ Maximizes your hardware

**Commands:**
```bash
# Local (one-time):
bash build_and_push.sh

# On B200 instance:
bash run_multigpu.sh
```

---

## 📊 **Performance Comparison**

| Method | GPUs | Training Time | Command |
|--------|------|---------------|---------|
| **Option 1 (B200 8×)** ⚡ | 8 | **5-10 min** | `bash run_multigpu.sh` |
| Option 2 (B200 1×) | 1 | 10-20 min | `bash run.sh` |
| Option 2 (H100) | 1 | 30-90 min | `bash run.sh` |
| Option 3 (No Docker) | 1 | 30-90 min | `bash run.sh` |

---

## 📁 **What You Get**

All options produce the same output:

```
models/
└── coloring_gnn.onnx      # Trained model for Rust

checkpoints/
├── best_model.pt          # PyTorch checkpoint
└── training_metadata.json # Training stats

logs/
└── events.out.tfevents.*  # TensorBoard logs
```

---

## 🔧 **Quick Commands**

### **Build Docker Image:**
```bash
cd /home/diddy/Desktop/PRISM-AI-DoD/python/gnn_training
bash build_and_push.sh
```

### **Multi-GPU Training (8x B200):**
```bash
# On B200 instance
bash run_multigpu.sh
```

### **Single GPU Training:**
```bash
# On any instance
bash run.sh
```

---

## 📖 **Documentation**

- **8x B200 guide:** `B200_DEPLOYMENT.md` ← **START HERE**
- **Docker (single GPU):** `DOCKER_DEPLOYMENT.md`
- **No Docker:** `RUNPOD_DEPLOYMENT.md`
- **Full details:** `README.md`

---

## ✅ **All Methods Include**

✅ Automated training (early stopping)
✅ Multi-task learning (4 tasks)
✅ ONNX export (automatic)
✅ TensorBoard logging
✅ Professional quality

**Expected Cost:** $2-5 (B200 multi-GPU)

---

## 🏁 **Next Steps**

1. **Read:** `B200_DEPLOYMENT.md`
2. **Build:** `bash build_and_push.sh`
3. **Deploy:** Use your 8x B200 instance
4. **Run:** `bash run_multigpu.sh`
5. **Wait:** 5-10 minutes
6. **Download:** `coloring_gnn.onnx`
7. **Integrate:** Copy to Rust project

**You have the BEST hardware. Let's use it!** 🚀
