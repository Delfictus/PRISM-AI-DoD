# Docker Deployment to RunPod (Professional Method)

Since you have DockerHub ready (`delfictus/prism-ai-world-record`), use this **professional approach** with custom Docker image.

---

## Why Use Docker?

✅ **Faster**: Dependencies pre-installed (~30 sec startup vs 2 min)
✅ **Reproducible**: Exact environment locked in
✅ **Professional**: Industry standard for ML deployment
✅ **Shareable**: Others can reproduce your results

---

## Step-by-Step: Build & Push Docker Image

### 1. Build Docker Image Locally

```bash
cd /home/diddy/Desktop/PRISM-AI-DoD/python/gnn_training

# One command to build and push:
bash build_and_push.sh
```

**What this does:**
1. Builds Docker image with all dependencies
2. Tests the image
3. Logs into DockerHub (if needed)
4. Pushes to `delfictus/prism-ai-world-record:latest`

**Expected output:**
```
Building Docker image...
Step 1/15 : FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
...
✅ Docker image built successfully!
Image size: 8.2 GB

Testing Docker image...
✅ PyTorch: 2.1.0
✅ PyG: 2.4.0
✅ CUDA available: True

Pushing to DockerHub...
✅ Docker image published successfully!

Image: delfictus/prism-ai-world-record:latest
```

**Time:** ~10-15 minutes (one-time build)

---

## Step-by-Step: Deploy to RunPod

### 2. Launch RunPod Instance with Custom Image

1. **Go to RunPod.io** → "Deploy" → "GPU Instance"

2. **Select GPU:**
   - H100 80GB PCIe ($1.99/hr) ✅

3. **Container Settings:**
   - Click "Custom Container" (instead of template)
   - **Container Image:** `delfictus/prism-ai-world-record:latest`
   - **Docker Command:** Leave blank (or `/bin/bash`)

4. **Storage:**
   - Container Disk: 50GB
   - Volume Disk: 10GB (optional)

5. **Click "Deploy"**

**RunPod will:**
- Pull your Docker image from DockerHub
- Start instance with H100
- Container ready in ~30 seconds

---

### 3. Upload Training Data

**Option A: Upload via Web Interface**
1. Click "Connect" → "File Browser"
2. Navigate to `/workspace`
3. Upload `training_data.tar.gz`
4. Extract: `tar -xzf training_data.tar.gz`

**Option B: Upload via SCP**
```bash
# First, package training data locally
cd /home/diddy/Desktop/PRISM-AI-DoD
tar -czf training_data.tar.gz training_data/

# Upload to RunPod
scp training_data.tar.gz root@<runpod-ip>:/workspace/

# SSH into RunPod
ssh root@<runpod-ip>

# Extract
cd /workspace
tar -xzf training_data.tar.gz
```

---

### 4. Run Training

**SSH into RunPod instance:**
```bash
ssh root@<runpod-ip>
```

**Start training:**
```bash
cd /workspace

# Verify everything is ready
ls training_data/graphs/ | wc -l  # Should show 15000
python -c "import torch; print(torch.cuda.is_available())"  # Should be True

# Run training (one command)
bash run.sh
```

**Expected output:**
```
========================================================================
PRISM-AI GNN Training - Automated Execution
========================================================================

GPU Information:
NVIDIA H100 80GB PCIe, 81920 MiB, 9.0

✅ Dependencies installed (already in Docker image - instant!)

Testing dataset loading...
✅ Dataset OK: 12000 graphs

Testing model architecture...
✅ Model OK: 2,487,432 params

========================================================================
Starting Training
========================================================================

Epoch [1/100]
  Batch [50/375] Loss: 2.4531 (color: 1.2, chromatic: 0.8, type: 0.3, diff: 0.15)
  ...
```

**Training will run for 30-90 minutes** with automatic:
- Early stopping
- Checkpointing
- ONNX export

---

### 5. Download Trained Model

**After training completes:**

```bash
# On RunPod instance
cd /workspace
tar -czf trained_model.tar.gz models/ checkpoints/

# Download to local machine
scp root@<runpod-ip>:/workspace/trained_model.tar.gz .

# Extract locally
tar -xzf trained_model.tar.gz
cp models/coloring_gnn.onnx /home/diddy/Desktop/PRISM-AI-DoD/models/
```

---

### 6. Terminate Instance

**CRITICAL:** Stop RunPod instance immediately!
1. RunPod dashboard → "Stop"
2. Verify it's stopped

---

## Complete Workflow Summary

**Local machine (one-time setup):**
```bash
# Build and push Docker image
cd /home/diddy/Desktop/PRISM-AI-DoD/python/gnn_training
bash build_and_push.sh
```

**RunPod (each training run):**
```bash
# 1. Deploy instance with custom image: delfictus/prism-ai-world-record:latest
# 2. Upload training data
scp training_data.tar.gz root@<runpod-ip>:/workspace/

# 3. SSH and run
ssh root@<runpod-ip>
cd /workspace
tar -xzf training_data.tar.gz
bash run.sh

# 4. Download results
scp root@<runpod-ip>:/workspace/trained_model.tar.gz .

# 5. Stop instance
```

**Total time:**
- Setup: ~15 min (one-time)
- Training: ~30-90 min (per run)
- Download: ~2 min

**Total cost:** $4-6 per training run

---

## Advantages of Docker Approach

| Aspect | Docker | No Docker |
|--------|--------|-----------|
| **Startup time** | ~30 sec | ~2 min |
| **Reproducibility** | Perfect | Good |
| **Dependency issues** | None | Possible |
| **Professional** | ✅ Yes | Basic |
| **Shareable** | ✅ Public image | No |

---

## Troubleshooting

### Docker Build Issues

**Problem:** Build fails on PyTorch Geometric
```bash
# Use alternative installation
docker build --build-arg TORCH_CUDA_ARCH_LIST="8.0;9.0" -t delfictus/prism-ai-world-record:latest .
```

**Problem:** Image too large
```bash
# Current size: ~8GB (normal for ML images)
# To reduce: use multi-stage build (advanced)
```

### RunPod Issues

**Problem:** "Image not found"
```bash
# Verify image is public
docker pull delfictus/prism-ai-world-record:latest

# Make image public on DockerHub:
# Settings → Make Public
```

**Problem:** Training data not found
```bash
# Check path
ls /workspace/training_data/graphs/ | wc -l
# Should show 15000
```

---

## Next Steps

After downloading trained model:

1. **Verify ONNX model:**
   ```bash
   cd /home/diddy/Desktop/PRISM-AI-DoD/src
   cargo run --release --example test_onnx_infrastructure --features cuda
   ```

2. **Run DIMACS benchmark:**
   ```bash
   cargo run --release --example benchmark_dimacs --features cuda
   ```

3. **World record attempt:**
   Target: DSJC1000.5 with ≤82 colors

---

## Summary

✅ **Docker image ready:** `delfictus/prism-ai-world-record:latest`
✅ **Deployment:** 1 command (`bash build_and_push.sh`)
✅ **Professional:** Industry-standard approach
✅ **Fast:** 30 sec startup on RunPod
✅ **Reproducible:** Exact environment locked in

**This is the BEST way to deploy for serious work.**
