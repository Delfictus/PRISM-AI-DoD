# PRISM-AI GNN Training Package for RunPod H100

Professional training package for Multi-Task Graph Coloring GNN.

**Target GPU:** RunPod H100 80GB
**Expected Training Time:** 30-90 minutes
**Expected Cost:** $4-6

---

## Package Contents

```
gnn_training/
├── dataset.py         # Data loader for 15k training graphs
├── model.py           # Multi-Task GATv2 architecture
├── train.py           # Training script with early stopping
├── export_onnx.py     # ONNX export for Rust integration
├── requirements.txt   # Python dependencies
├── run.sh             # Automated execution script
└── README.md          # This file
```

---

## RunPod Deployment Guide

### Step 1: Create RunPod Account

1. Go to https://runpod.io
2. Sign up / Log in
3. Add payment method

### Step 2: Select GPU Instance

1. Click "Deploy" → "GPU Instance"
2. **Recommended: H100 80GB PCIe** ($1.99/hr)
   - Alternative: A100 80GB ($1.29/hr) if H100 unavailable
3. **Template:** PyTorch 2.1 CUDA 12.1
4. **Disk Space:** 50GB (minimum)
5. **Region:** Any with H100 availability

### Step 3: Upload Training Package

**Option A: Upload via RunPod**
```bash
# On your local machine, create tarball
cd /home/diddy/Desktop/PRISM-AI-DoD
tar -czf prism_gnn_training.tar.gz python/gnn_training training_data

# Upload to RunPod instance (use web interface or runpodctl)
```

**Option B: Clone from Git (if you pushed to GitHub)**
```bash
# On RunPod instance
git clone <your-repo-url>
cd <repo>/python/gnn_training
```

### Step 4: Run Training

SSH into your RunPod instance, then:

```bash
cd /workspace/prism_gnn_training/python/gnn_training

# One command to run everything:
bash run.sh
```

The script will:
1. ✅ Check GPU availability
2. ✅ Install all dependencies
3. ✅ Test dataset loading
4. ✅ Test model architecture
5. ✅ Train for up to 100 epochs (with early stopping)
6. ✅ Export best model to ONNX
7. ✅ Print training summary

### Step 5: Download Trained Model

After training completes (~30-90 minutes), download:

```bash
# On RunPod instance, create download package
cd /workspace/prism_gnn_training
tar -czf trained_model.tar.gz \
    models/coloring_gnn.onnx \
    python/gnn_training/checkpoints/best_model.pt \
    python/gnn_training/checkpoints/training_metadata.json \
    python/gnn_training/logs/
```

Download via:
- RunPod web interface → File Browser
- Or SCP: `scp user@runpod-ip:/workspace/prism_gnn_training/trained_model.tar.gz .`

### Step 6: Integrate into Rust

On your local machine:

```bash
# Extract trained model
tar -xzf trained_model.tar.gz

# ONNX model is now at: models/coloring_gnn.onnx
# Rust will automatically detect and use it!
```

### Step 7: Terminate RunPod Instance

**IMPORTANT:** Stop the instance immediately to avoid charges!

1. Go to RunPod dashboard
2. Click "Stop" on your instance
3. Verify instance is terminated

---

## Manual Training (If Not Using run.sh)

### Install Dependencies
```bash
pip install -r requirements.txt
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

### Train Model
```bash
python train.py \
    --data-dir ../../training_data \
    --batch-size 64 \
    --epochs 100 \
    --hidden-dim 256 \
    --num-layers 6 \
    --num-heads 8 \
    --early-stop-patience 15
```

### Export to ONNX
```bash
python export_onnx.py \
    --checkpoint ./checkpoints/best_model.pt \
    --output ../../models/coloring_gnn.onnx
```

---

## Training Configuration

### Model Architecture
- **Type:** Multi-Task GATv2
- **Layers:** 6 graph attention layers
- **Hidden Dim:** 256
- **Attention Heads:** 8 per layer
- **Parameters:** ~2-3 million

### Tasks (Multi-Task Learning)
1. **Node Color Prediction** (50% loss weight)
   - Per-node classification
   - 200 color classes

2. **Chromatic Number Prediction** (25% loss weight)
   - Graph-level regression
   - Target: exact chromatic number

3. **Graph Type Classification** (15% loss weight)
   - 8 graph types (Random, Leighton, Mycielski, etc.)

4. **Difficulty Score Prediction** (10% loss weight)
   - Graph-level regression
   - Range: 0-100

### Optimization
- **Optimizer:** AdamW
- **Learning Rate:** 0.001 (with ReduceLROnPlateau)
- **Batch Size:** 64 (optimized for H100)
- **Mixed Precision:** Enabled (AMP)
- **Gradient Clipping:** Max norm 1.0
- **Early Stopping:** 15 epochs patience

### Expected Performance
After training on 15k graphs:
- **Color Accuracy:** >70%
- **Chromatic MAE:** <5 colors
- **Graph Type Accuracy:** >85%
- **Training Time:** 30-90 minutes on H100

---

## Monitoring Training

### TensorBoard (Optional)

If you want to monitor training in real-time:

```bash
# On RunPod instance, start TensorBoard
tensorboard --logdir ./logs --port 6006 --bind_all

# Then SSH tunnel to your local machine:
# ssh -L 6006:localhost:6006 user@runpod-ip

# Open browser: http://localhost:6006
```

### Training Output

The script prints progress every 50 batches:
```
Epoch [1/100]
  Batch [50/375] Loss: 2.4531 (color: 1.2, chromatic: 0.8, type: 0.3, diff: 0.15)
  ...
  Train Loss: 2.1234 (color: 1.0, chromatic: 0.7, type: 0.25, diff: 0.17)
  Val Loss:   1.9876 (color: 0.9, chromatic: 0.6, type: 0.28, diff: 0.20)
  Val Metrics: Chromatic MAE: 4.23, Color Acc: 0.712, Type Acc: 0.856
  ✅ New best model saved (val_loss: 1.9876)
```

---

## Troubleshooting

### CUDA Out of Memory
Reduce batch size:
```bash
python train.py --batch-size 32  # Instead of 64
```

### PyTorch Geometric Installation Issues
```bash
# Install from source
pip install git+https://github.com/pyg-team/pytorch_geometric.git
```

### Dataset Not Found
Ensure `../../training_data` exists with:
- `graphs/train_*.json` (12,000 files)
- `graphs/val_*.json` (3,000 files)
- `metadata.json`

---

## Cost Optimization Tips

1. **Use Spot Instances** (if available) - Save 50-70%
2. **Monitor Training** - Stop early if converging faster
3. **Download Immediately** - Don't leave instance running
4. **Use A100 40GB** ($0.79/hr) if budget-constrained

**Estimated Costs:**
- H100 80GB: ~$4-6 (fastest)
- A100 80GB: ~$3-4
- A100 40GB: ~$2-3 (slower but sufficient)

---

## Output Files

After successful training:

```
checkpoints/
├── best_model.pt              # PyTorch checkpoint (~50 MB)
└── training_metadata.json     # Training statistics

logs/
└── events.out.tfevents.*      # TensorBoard logs

../../models/
└── coloring_gnn.onnx          # ONNX model for Rust (~20-30 MB)
```

---

## Next Steps After Training

1. Download `coloring_gnn.onnx` to your local machine
2. Place it at: `/home/diddy/Desktop/PRISM-AI-DoD/models/coloring_gnn.onnx`
3. Run Rust benchmark: `cargo run --release --example benchmark_dimacs --features cuda`
4. The system will automatically use the trained GNN instead of placeholders!

---

## Support

If training fails or you encounter issues:
1. Check GPU availability: `nvidia-smi`
2. Verify dataset: `ls -l ../../training_data/graphs/ | wc -l` (should be 15000)
3. Test data loading: `python -c "from dataset import GraphColoringDataset; ds = GraphColoringDataset('../../training_data', 'train'); print(len(ds))"`
4. Check logs: `cat logs/events.out.tfevents.*`

---

**Ready to train! Good luck with the world record attempt! 🚀**
