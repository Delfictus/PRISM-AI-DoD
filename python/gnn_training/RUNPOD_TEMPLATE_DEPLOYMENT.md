# RunPod Template Deployment - 8x B200 (Simplified)

**Using RunPod's built-in PyTorch template - NO custom Docker build needed!**

**Template:** `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
**Your Hardware:** 8x B200 (1,440 GB VRAM)
**Training Time:** 5-10 minutes
**Cost:** ~$2-5

---

## 🚀 **Quick Start (3 Steps)**

### **Step 1: Deploy RunPod Instance**

**Go to RunPod → Deploy:**
1. **GPU:** Select your 8x B200 instance (you already have it!)
2. **Template:** `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
3. **Volume Mount:** `/workspace` (default - keep it)
4. **Deploy**

**That's it - no custom Docker image needed!**

---

### **Step 2: Upload Training Package**

**Create upload package (local machine):**

```bash
cd /home/diddy/Desktop/PRISM-AI-DoD

# Package training code
tar -czf gnn_training.tar.gz python/gnn_training/

# Package training data
tar -czf training_data.tar.gz training_data/

# Both files ready to upload!
```

**Upload to your B200 instance:**

**Option A: SCP (Recommended)**
```bash
# Upload both packages
scp gnn_training.tar.gz root@<your-pod-ip>:/workspace/
scp training_data.tar.gz root@<your-pod-ip>:/workspace/
```

**Option B: RunPod Web Interface**
1. Click "Connect" → "File Browser"
2. Navigate to `/workspace`
3. Upload `gnn_training.tar.gz` and `training_data.tar.gz`

---

### **Step 3: Run Training**

**SSH into your B200 instance:**

```bash
ssh root@<your-pod-ip>

# Navigate to workspace
cd /workspace

# Extract packages
tar -xzf gnn_training.tar.gz
tar -xzf training_data.tar.gz

# Run multi-GPU training (uses all 8 GPUs)
cd python/gnn_training
bash run_multigpu.sh
```

**That's it!** Training will start automatically.

---

## 📊 **What Happens**

### **Automatic Setup:**
1. ✅ Detects 8x B200 GPUs
2. ✅ Installs PyTorch Geometric (if needed)
3. ✅ Validates dataset (15k graphs)
4. ✅ Tests model architecture

### **Training:**
- Uses all 8 GPUs via PyTorch DDP
- Effective batch size: 512 (64 per GPU × 8)
- Early stopping enabled
- TensorBoard logging

### **Expected Output:**
```
========================================================================
PRISM-AI GNN Multi-GPU Training - 8x B200 Optimized
========================================================================

GPU Configuration:
     1  NVIDIA B200, 196608 MiB
     2  NVIDIA B200, 196608 MiB
     3  NVIDIA B200, 196608 MiB
     4  NVIDIA B200, 196608 MiB
     5  NVIDIA B200, 196608 MiB
     6  NVIDIA B200, 196608 MiB
     7  NVIDIA B200, 196608 MiB
     8  NVIDIA B200, 196608 MiB

Total GPUs detected: 8

Installing dependencies...
✅ Dependencies installed

Verifying dataset...
✅ Dataset OK: 12000 graphs

========================================================================
Starting Multi-GPU Training
========================================================================

Configuration:
  GPUs: 8
  Batch size per GPU: 64
  Effective batch size: 512
  Estimated training time: 5-10 minutes

Distributed Training Setup:
  World size: 8 GPUs
  Device: NVIDIA B200
  Total VRAM: 1536 GB

Datasets:
  Train: 12000 graphs (1500 per GPU)
  Val:   3000 graphs

Model:
  Parameters: 2,487,432
  Effective batch size: 512

Epoch [1/100]
  [GPU 0] Batch [50/94] Loss: 2.1234 ...
  Train Loss: 2.0543
  Val Loss:   1.8765
  ✅ New best model saved (val_loss: 1.8765)
  Epoch time: 35.2s

...

✅ Training complete!
  Total time: 6.3 minutes
  Best val loss: 1.2345
  GPUs used: 8

========================================================================
Exporting to ONNX
========================================================================

Loading checkpoint from: ./checkpoints/best_model.pt
  Epoch: 42
  Val loss: 1.2345

Exporting to ONNX...
  Output path: ../../models/coloring_gnn.onnx
  ✅ ONNX export successful!

Verifying ONNX model...
  ✅ ONNX model is valid!
  Model size: 24.37 MB

✅ ONNX export complete!
  Saved to: ../../models/coloring_gnn.onnx
  Ready for Rust ONNX Runtime integration

========================================================================
Multi-GPU Training Complete!
========================================================================

Training Summary:
{
  "best_val_loss": 1.2345,
  "total_epochs": 42,
  "total_time_minutes": 6.3,
  "gpus_used": 8,
  "final_lr": 0.0005,
  "model_params": 2487432
}

Files created:
  - Model checkpoint: ./checkpoints/best_model.pt
  - ONNX model: ../../models/coloring_gnn.onnx
  - Training logs: ./logs/

✅ Ready to download and integrate into Rust!

Performance Stats:
  GPUs used: 8
  Training time: 6.3 minutes
  Speedup: ~8x faster than single GPU
```

---

## 📥 **Download Trained Model**

### **After training completes:**

```bash
# On B200 instance
cd /workspace
tar -czf trained_model.tar.gz models/ python/gnn_training/checkpoints/

# Download to local machine
scp root@<your-pod-ip>:/workspace/trained_model.tar.gz .
```

### **On local machine:**

```bash
# Extract
tar -xzf trained_model.tar.gz

# Copy ONNX model to Rust project
cp models/coloring_gnn.onnx /home/diddy/Desktop/PRISM-AI-DoD/models/

# Verify
ls -lh /home/diddy/Desktop/PRISM-AI-DoD/models/coloring_gnn.onnx
```

---

## 🛑 **Stop Instance (IMPORTANT)**

**After downloading the model:**

1. Go to RunPod dashboard
2. Click "Stop" on your instance
3. **Verify it's stopped** to avoid charges

---

## 📁 **File Structure on B200**

After extraction:

```
/workspace/
├── gnn_training.tar.gz          (uploaded)
├── training_data.tar.gz         (uploaded)
├── python/
│   └── gnn_training/
│       ├── train_multigpu.py    (multi-GPU training)
│       ├── run_multigpu.sh      (execution script)
│       ├── dataset.py
│       ├── model.py
│       ├── export_onnx.py
│       └── ... (other files)
├── training_data/
│   ├── graphs/
│   │   ├── train_*.json (12,000 files)
│   │   └── val_*.json (3,000 files)
│   └── metadata.json
├── models/                      (created during training)
│   └── coloring_gnn.onnx
└── checkpoints/                 (created during training)
    ├── best_model.pt
    └── training_metadata.json
```

---

## 🔧 **PyTorch Version Compatibility**

**RunPod template has PyTorch 2.8.0** (newer than my 2.1.0 code)

**Compatibility:**
- ✅ **Fully compatible** - PyTorch 2.8.0 supports all PyTorch 2.1.0 APIs
- ✅ **Better performance** - Newer version is faster
- ✅ **B200 support** - CUDA 12.8.1 perfect for B200

**No code changes needed!**

---

## ⚡ **Performance Expectations**

| Hardware | Template | Training Time |
|----------|----------|---------------|
| 8x B200 | RunPod PyTorch 2.8.0 | **5-10 min** ✅ |
| 8x B200 | Custom Docker 2.1.0 | 5-10 min |
| 1x B200 | Any | 10-20 min |
| 1x H100 | Any | 30-90 min |

**Using RunPod's template = Same performance, easier setup!**

---

## 💰 **Cost Estimate**

**8x B200 Training:**
- Setup time: ~5 minutes
- Training time: ~5-10 minutes
- Download time: ~2 minutes
- **Total runtime:** ~15-20 minutes
- **Estimated cost:** ~$2-5

---

## 🐛 **Troubleshooting**

### **Issue: PyTorch Geometric Installation**

If `pip install torch-geometric` fails:

```bash
# Alternative installation
pip install git+https://github.com/pyg-team/pytorch_geometric.git
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

### **Issue: CUDA Not Detected**

```bash
# Check CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
nvidia-smi
```

Should show 8 B200 GPUs.

### **Issue: Training Stalls**

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Check logs
tail -f logs/events.out.tfevents.*
```

---

## ✅ **Complete Workflow**

### **1. Local Machine (Preparation):**

```bash
cd /home/diddy/Desktop/PRISM-AI-DoD

# Create packages
tar -czf gnn_training.tar.gz python/gnn_training/
tar -czf training_data.tar.gz training_data/

# Upload to B200
scp gnn_training.tar.gz root@<pod-ip>:/workspace/
scp training_data.tar.gz root@<pod-ip>:/workspace/
```

### **2. B200 Instance (Training):**

```bash
ssh root@<pod-ip>

cd /workspace
tar -xzf gnn_training.tar.gz
tar -xzf training_data.tar.gz

cd python/gnn_training
bash run_multigpu.sh

# Wait 5-10 minutes...

# Package results
cd /workspace
tar -czf trained_model.tar.gz models/ python/gnn_training/checkpoints/
```

### **3. Local Machine (Integration):**

```bash
# Download
scp root@<pod-ip>:/workspace/trained_model.tar.gz .

# Extract and integrate
tar -xzf trained_model.tar.gz
cp models/coloring_gnn.onnx /home/diddy/Desktop/PRISM-AI-DoD/models/

# Test in Rust
cd /home/diddy/Desktop/PRISM-AI-DoD/src
cargo run --release --example test_onnx_infrastructure --features cuda
```

### **4. RunPod (Cleanup):**

**Stop the instance immediately!**

---

## 🎯 **Summary**

**No Docker build needed!** Just:

1. ✅ Deploy with RunPod's PyTorch template
2. ✅ Upload training code + data
3. ✅ Run `bash run_multigpu.sh`
4. ✅ Download trained model
5. ✅ Stop instance

**Total time:** 15-20 minutes (including setup)
**Training time:** 5-10 minutes
**Cost:** ~$2-5

**This is the SIMPLEST way to use your 8x B200!** 🚀
