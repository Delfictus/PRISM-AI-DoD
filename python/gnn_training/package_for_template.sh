#!/bin/bash
# Package training code and data for RunPod template deployment
# NO Docker build needed - uses RunPod's PyTorch template

echo "Creating RunPod template deployment packages..."
echo ""

cd /home/diddy/Desktop/PRISM-AI-DoD

# Package training code
echo "Packaging training code..."
tar -czf gnn_training.tar.gz python/gnn_training/
SIZE1=$(du -h gnn_training.tar.gz | cut -f1)
echo "  ✅ gnn_training.tar.gz ($SIZE1)"

# Package training data
echo "Packaging training data (15k graphs)..."
tar -czf training_data.tar.gz training_data/
SIZE2=$(du -h training_data.tar.gz | cut -f1)
echo "  ✅ training_data.tar.gz ($SIZE2)"

echo ""
echo "=========================================================================="
echo "✅ RunPod packages created successfully!"
echo "=========================================================================="
echo ""
echo "Files created:"
echo "  - gnn_training.tar.gz ($SIZE1)"
echo "  - training_data.tar.gz ($SIZE2)"
echo ""
echo "Next steps:"
echo "  1. Deploy RunPod instance with template:"
echo "     runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
echo ""
echo "  2. Upload packages to /workspace:"
echo "     scp gnn_training.tar.gz root@<pod-ip>:/workspace/"
echo "     scp training_data.tar.gz root@<pod-ip>:/workspace/"
echo ""
echo "  3. SSH into instance and run:"
echo "     cd /workspace"
echo "     tar -xzf gnn_training.tar.gz"
echo "     tar -xzf training_data.tar.gz"
echo "     cd python/gnn_training"
echo "     bash run_multigpu.sh"
echo ""
echo "  4. Wait 5-10 minutes for training to complete"
echo ""
echo "  5. Download results:"
echo "     scp root@<pod-ip>:/workspace/models/coloring_gnn.onnx ."
echo ""
echo "See: python/gnn_training/RUNPOD_TEMPLATE_DEPLOYMENT.md for full guide"
echo ""
