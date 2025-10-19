#!/bin/bash
# Build and push Docker image to DockerHub

set -e

echo "=========================================================================="
echo "Building PRISM-AI GNN Training Docker Image"
echo "=========================================================================="

# Configuration
IMAGE_NAME="delfictus/prism-ai-world-record"
TAG="latest"
FULL_IMAGE="${IMAGE_NAME}:${TAG}"

echo ""
echo "Image: $FULL_IMAGE"
echo ""

# Build image
echo "Building Docker image..."
docker build -t $FULL_IMAGE .

echo ""
echo "✅ Docker image built successfully!"
echo ""

# Get image size
SIZE=$(docker images $FULL_IMAGE --format "{{.Size}}")
echo "Image size: $SIZE"
echo ""

# Test image
echo "Testing Docker image..."
docker run --rm $FULL_IMAGE python -c "import torch; import torch_geometric; print(f'✅ PyTorch: {torch.__version__}'); print(f'✅ PyG: {torch_geometric.__version__}'); print(f'✅ CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "=========================================================================="
echo "Pushing to DockerHub"
echo "=========================================================================="

# Login check
echo ""
echo "Checking DockerHub login..."
if ! docker info | grep -q "Username"; then
    echo "Not logged in to DockerHub. Logging in..."
    docker login
else
    echo "✅ Already logged in to DockerHub"
fi

# Push image
echo ""
echo "Pushing image to DockerHub..."
docker push $FULL_IMAGE

echo ""
echo "=========================================================================="
echo "✅ Docker image published successfully!"
echo "=========================================================================="
echo ""
echo "Image: $FULL_IMAGE"
echo "Size: $SIZE"
echo ""
echo "To use on RunPod:"
echo "  1. Select 'Custom Container'"
echo "  2. Image: $FULL_IMAGE"
echo "  3. Upload training_data/ to /workspace/training_data"
echo "  4. Run: bash run.sh"
echo ""
