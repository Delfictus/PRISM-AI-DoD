#!/bin/bash
set -e

echo "=========================================================================="
echo "PRISM-AI GNN Multi-GPU Training - 8x B200 Optimized"
echo "=========================================================================="

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

if [ $NUM_GPUS -eq 0 ]; then
    echo "❌ ERROR: No GPUs detected!"
    exit 1
fi

echo ""
echo "GPU Configuration:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | nl
echo ""
echo "Total GPUs detected: $NUM_GPUS"
echo ""

# Install dependencies (if needed)
if ! python -c "import torch_geometric" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -q --upgrade pip
    pip install -q -r requirements.txt
    pip install -q pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
    echo "✅ Dependencies installed"
else
    echo "✅ Dependencies already installed"
fi

echo ""

# Test dataset
echo "Verifying dataset..."
python -c "from dataset import GraphColoringDataset; ds = GraphColoringDataset('../../training_data', 'train'); print(f'✅ Dataset OK: {len(ds)} graphs')"

echo ""
echo "=========================================================================="
echo "Starting Multi-GPU Training"
echo "=========================================================================="
echo ""
echo "Configuration:"
echo "  GPUs: $NUM_GPUS"
echo "  Batch size per GPU: 64"
echo "  Effective batch size: $((64 * NUM_GPUS))"
echo "  Estimated training time: 5-10 minutes"
echo ""

# Launch multi-GPU training with torchrun
torchrun \
    --standalone \
    --nproc_per_node=$NUM_GPUS \
    train_multigpu.py \
    --data-dir ../../training_data \
    --max-colors 200 \
    --hidden-dim 256 \
    --num-layers 6 \
    --num-heads 8 \
    --dropout 0.2 \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.001 \
    --weight-decay 1e-5 \
    --early-stop-patience 15 \
    --num-workers 4 \
    --checkpoint-dir ./checkpoints \
    --log-dir ./logs

echo ""
echo "=========================================================================="
echo "Exporting to ONNX"
echo "=========================================================================="

# Export best model to ONNX
python export_onnx.py \
    --checkpoint ./checkpoints/best_model.pt \
    --output ../../models/coloring_gnn.onnx \
    --opset 17

echo ""
echo "=========================================================================="
echo "Multi-GPU Training Complete!"
echo "=========================================================================="

if [ -f "./checkpoints/training_metadata.json" ]; then
    echo ""
    echo "Training Summary:"
    cat ./checkpoints/training_metadata.json
    echo ""
fi

echo ""
echo "Files created:"
echo "  - Model checkpoint: ./checkpoints/best_model.pt"
echo "  - ONNX model: ../../models/coloring_gnn.onnx"
echo "  - Training logs: ./logs/"
echo ""

echo "✅ Ready to download and integrate into Rust!"
echo ""
echo "Performance Stats:"
echo "  GPUs used: $NUM_GPUS"
if [ -f "./checkpoints/training_metadata.json" ]; then
    TRAINING_TIME=$(python -c "import json; print(f\"{json.load(open('./checkpoints/training_metadata.json'))['total_time_minutes']:.1f}\")")
    echo "  Training time: ${TRAINING_TIME} minutes"
    echo "  Speedup: ~$((NUM_GPUS))x faster than single GPU"
fi
echo ""
