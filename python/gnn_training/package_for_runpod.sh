#!/bin/bash
# Package training code and data for RunPod deployment

echo "Creating RunPod deployment package..."

cd /home/diddy/Desktop/PRISM-AI-DoD

# Create package directory
rm -rf runpod_package
mkdir -p runpod_package

# Copy training code
echo "Copying training code..."
cp -r python/gnn_training runpod_package/

# Copy training data
echo "Copying training data (15k graphs)..."
cp -r training_data runpod_package/

# Create tarball
echo "Creating tarball..."
tar -czf prism_gnn_runpod.tar.gz runpod_package/

# Get size
SIZE=$(du -h prism_gnn_runpod.tar.gz | cut -f1)

echo ""
echo "=========================================================================="
echo "✅ RunPod package created successfully!"
echo "=========================================================================="
echo ""
echo "Package: prism_gnn_runpod.tar.gz"
echo "Size: $SIZE"
echo ""
echo "Next steps:"
echo "  1. Upload prism_gnn_runpod.tar.gz to RunPod instance"
echo "  2. Extract: tar -xzf prism_gnn_runpod.tar.gz"
echo "  3. Run: cd runpod_package/gnn_training && bash run.sh"
echo ""
echo "See runpod_package/gnn_training/README.md for detailed instructions."
echo ""
