#!/bin/bash
set -e

echo "=========================================================================="
echo "PRISM-AI GNN Training - Automated Execution"
echo "=========================================================================="

# Check CUDA availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ ERROR: nvidia-smi not found. CUDA required for training."
    exit 1
fi

echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
echo ""

# Install dependencies
echo "Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Install PyTorch Geometric dependencies (specific versions for compatibility)
echo "Installing PyTorch Geometric extras..."
pip install -q pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

echo ""
echo "✅ Dependencies installed"
echo ""

# Test dataset loading
echo "Testing dataset loading..."
python -c "from dataset import GraphColoringDataset; ds = GraphColoringDataset('../../training_data', 'train'); print(f'✅ Dataset OK: {len(ds)} graphs')"

# Test model architecture
echo "Testing model architecture..."
python -c "from model import MultiTaskGATv2; import torch; m = MultiTaskGATv2(); print(f'✅ Model OK: {sum(p.numel() for p in m.parameters()):,} params')"

echo ""
echo "=========================================================================="
echo "Starting Training"
echo "=========================================================================="

# Start training with optimized hyperparameters for H100
python train.py \
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
    --num-workers 8 \
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
echo "Training Complete!"
echo "=========================================================================="

# Print summary
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

# Optional: Auto-shutdown RunPod instance (uncomment if using RunPod API)
# if [ ! -z "$RUNPOD_POD_ID" ]; then
#     echo "Auto-stopping RunPod instance in 60 seconds..."
#     sleep 60
#     runpodctl stop $RUNPOD_POD_ID
# fi
