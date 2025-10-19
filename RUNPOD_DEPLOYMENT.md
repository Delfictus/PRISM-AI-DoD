# RunPod H100 Deployment Guide - PRISM-AI GNN Training

**Status:** ✅ Training package ready for deployment
**Target:** RunPod H100 80GB
**Expected Time:** 30-90 minutes
**Expected Cost:** $4-6

---

## Quick Start (5 Steps)

### 1. Create Deployment Package
```bash
cd /home/diddy/Desktop/PRISM-AI-DoD
bash python/gnn_training/package_for_runpod.sh
```

This creates: `prism_gnn_runpod.tar.gz` (~540KB compressed)

### 2. Deploy RunPod Instance

1. Go to https://runpod.io
2. Click "Deploy" → "GPU Instance"
3. Select:
   - **GPU:** H100 80GB PCIe ($1.99/hr)
   - **Template:** PyTorch 2.1 CUDA 12.1
   - **Disk:** 50GB minimum
4. Click "Deploy"

### 3. Upload Package

**SSH into your RunPod instance:**
```bash
ssh root@<runpod-ip-address>
```

**Upload the package** (use SCP or RunPod web interface):
```bash
# From your local machine
scp prism_gnn_runpod.tar.gz root@<runpod-ip>:/workspace/
```

### 4. Run Training

**On RunPod instance:**
```bash
cd /workspace
tar -xzf prism_gnn_runpod.tar.gz
cd runpod_package/gnn_training

# One command to run everything:
bash run.sh
```

**What happens:**
1. Installs dependencies (~2 min)
2. Validates dataset (15k graphs)
3. Tests model architecture
4. Trains for up to 100 epochs (~30-90 min)
5. Exports to ONNX automatically
6. Saves best model

**Expected Output:**
```
Epoch [1/100]
  Batch [50/375] Loss: 2.4531 (color: 1.2, chromatic: 0.8, type: 0.3, diff: 0.15)
  ...
  Val Loss: 1.9876 (color: 0.9, chromatic: 0.6, type: 0.28, diff: 0.20)
  ✅ New best model saved (val_loss: 1.9876)

...

✅ Training complete!
  Total time: 45.2 minutes
  Best val loss: 1.2345
```

### 5. Download Trained Model

**On RunPod instance:**
```bash
cd /workspace/runpod_package
tar -czf trained_model.tar.gz models/ gnn_training/checkpoints/
```

**Download to local machine:**
```bash
# From your local machine
scp root@<runpod-ip>:/workspace/runpod_package/trained_model.tar.gz .
```

**Extract and place model:**
```bash
tar -xzf trained_model.tar.gz
cp models/coloring_gnn.onnx /home/diddy/Desktop/PRISM-AI-DoD/models/
```

**🎉 Done! The trained model is now ready for Rust integration.**

### 6. Terminate RunPod Instance

**CRITICAL:** Stop the instance to avoid charges!

1. Go to RunPod dashboard
2. Click "Stop" on your instance
3. **Verify it's stopped**

---

## What Gets Trained

### Model Architecture
- **Multi-Task GATv2** (Graph Attention Network v2)
- **6 layers** with 8 attention heads each
- **256 hidden dimensions**
- **~2.5M parameters**

### Training Tasks (Multi-Task Learning)
1. **Node Color Prediction** (50% weight)
   - Predicts optimal color for each vertex
   - 200 color classes

2. **Chromatic Number** (25% weight)
   - Predicts minimum colors needed
   - Regression task

3. **Graph Type Classification** (15% weight)
   - Identifies graph structure type
   - 8 classes (Random, Leighton, Mycielski, etc.)

4. **Difficulty Score** (10% weight)
   - Estimates solving difficulty
   - Range: 0-100

### Training Dataset
- **15,000 graphs** total
  - 12,000 training
  - 3,000 validation
- **8 graph types** for diversity
- **Ground truth** from GPU coloring solver

---

## Monitoring Training (Optional)

### Real-Time Monitoring with TensorBoard

**On RunPod instance:**
```bash
# Start TensorBoard (in a separate terminal)
tensorboard --logdir /workspace/runpod_package/gnn_training/logs --port 6006 --bind_all
```

**On your local machine:**
```bash
# Create SSH tunnel
ssh -L 6006:localhost:6006 root@<runpod-ip>

# Open browser
firefox http://localhost:6006
```

You'll see real-time graphs of:
- Training/validation loss
- Task-specific losses
- Learning rate schedule
- Accuracy metrics

---

## Cost Breakdown

| GPU | $/hour | Est. Time | Est. Cost |
|-----|--------|-----------|-----------|
| **H100 80GB** | $1.99 | 30-90 min | **$4-6** ✅ |
| A100 80GB | $1.29 | 60-120 min | $3-5 |
| A100 40GB | $0.79 | 90-180 min | $2-4 |

**Recommendation:** H100 for speed + reliability

---

## Troubleshooting

### Issue: CUDA Out of Memory
**Solution:** Reduce batch size
```bash
# Edit run.sh, change:
--batch-size 32  # Instead of 64
```

### Issue: Package Upload Failed
**Alternative:** Clone from GitHub
```bash
# Push to your GitHub first, then on RunPod:
git clone https://github.com/yourusername/PRISM-AI-DoD
cd PRISM-AI-DoD/python/gnn_training
bash run.sh
```

### Issue: Dataset Not Found
**Check:**
```bash
ls -l ../../training_data/graphs/ | wc -l  # Should be 15000
cat ../../training_data/metadata.json
```

### Issue: Dependencies Failed to Install
**Manual install:**
```bash
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install onnx onnxruntime-gpu tensorboard
```

---

## Expected Results

After successful training, you should see:

**Validation Metrics:**
- **Color Accuracy:** 70-80%
- **Chromatic MAE:** 3-6 colors
- **Graph Type Accuracy:** 85-95%
- **Difficulty R²:** 0.7-0.9

**Files Created:**
```
models/
└── coloring_gnn.onnx          # 20-30 MB ONNX model

gnn_training/checkpoints/
├── best_model.pt              # 50 MB PyTorch checkpoint
└── training_metadata.json     # Training stats

gnn_training/logs/
└── events.out.tfevents.*      # TensorBoard logs
```

---

## Integration with Rust

Once you have `coloring_gnn.onnx`:

1. **Place the model:**
   ```bash
   cp trained_model/models/coloring_gnn.onnx /home/diddy/Desktop/PRISM-AI-DoD/models/
   ```

2. **Verify integration:**
   ```bash
   cd /home/diddy/Desktop/PRISM-AI-DoD/src
   cargo run --release --example test_onnx_infrastructure --features cuda
   ```

   Should output:
   ```
   [GNN] ✅ Real ONNX model loaded with CUDA
   ✅ Prediction succeeded
   ```

3. **Run full benchmark:**
   ```bash
   cargo run --release --example benchmark_dimacs --features cuda
   ```

   The system will now use **real GNN predictions** instead of placeholders!

---

## Next Steps After Training

1. ✅ Download trained model
2. ✅ Integrate into Rust (just copy ONNX file)
3. ⏳ Run DIMACS benchmark suite (world record attempt)
4. ⏳ Statistical validation (100+ runs)
5. ⏳ Publish results

---

## Support Resources

- **RunPod Docs:** https://docs.runpod.io
- **PyTorch Geometric:** https://pytorch-geometric.readthedocs.io
- **Training Script:** `python/gnn_training/train.py`
- **Full README:** `python/gnn_training/README.md`

---

## Summary

**You now have:**
1. ✅ Complete training package ready for RunPod
2. ✅ Automated deployment script (`run.sh`)
3. ✅ Professional training pipeline with early stopping
4. ✅ ONNX export for Rust integration
5. ✅ Comprehensive documentation

**Total setup time:** ~10-15 minutes
**Total training time:** ~30-90 minutes
**Total cost:** ~$4-6

**Professional, fast, and cost-efficient. Ready when you are!** 🚀
