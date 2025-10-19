"""
Export Trained GNN to ONNX for Rust Inference

Converts PyTorch GNN → ONNX format for GPU inference in Rust using ONNX Runtime.

GPU-ONLY: ONNX model configured for CUDA execution provider
"""

import torch
import onnx
import onnxruntime as ort
from pathlib import Path
import numpy as np
from typing import Tuple

from model import ColoringGNN

# GPU-ONLY enforcement
assert torch.cuda.is_available(), "GPU required for ONNX export"
device = torch.device('cuda')


def load_trained_model(checkpoint_path: str, config: dict) -> ColoringGNN:
    """
    Load trained model from checkpoint

    Args:
        checkpoint_path: Path to saved checkpoint (.pt file)
        config: Model configuration dict

    Returns:
        Loaded model in eval mode on GPU
    """
    print(f"[Load] Loading checkpoint: {checkpoint_path}")

    # Create model with same architecture
    model = ColoringGNN(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        max_colors=config['max_colors'],
        num_graph_types=config.get('num_graph_types', 8),
    ).to(device)

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"  ✅ Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    if 'val_loss' in checkpoint:
        print(f"  ✅ Val loss: {checkpoint['val_loss']:.4f}")

    return model


def export_to_onnx(
    model: ColoringGNN,
    onnx_path: str,
    max_nodes: int = 1000,
    max_edges: int = 250000,
    opset_version: int = 17,
) -> None:
    """
    Export PyTorch model to ONNX format

    Args:
        model: Trained ColoringGNN model
        onnx_path: Output path for ONNX file
        max_nodes: Maximum nodes for dynamic axes
        max_edges: Maximum edges for dynamic axes
        opset_version: ONNX opset version (17 for CUDA support)
    """
    print(f"\n[Export] Converting to ONNX...")
    print(f"  Output: {onnx_path}")
    print(f"  Opset: {opset_version}")

    model.eval()

    # Create dummy input matching expected format
    # Use realistic sizes (e.g., DSJC1000.5 has ~1000 nodes, ~250k edges)
    num_nodes = 100  # Dummy graph for export
    num_edges = 300

    dummy_x = torch.randn(num_nodes, 16, device=device)
    dummy_edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
    dummy_batch = torch.zeros(num_nodes, dtype=torch.long, device=device)

    print(f"  Dummy input shapes:")
    print(f"    x: {dummy_x.shape}")
    print(f"    edge_index: {dummy_edge_index.shape}")
    print(f"    batch: {dummy_batch.shape}")

    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_x, dummy_edge_index, dummy_batch),
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['node_features', 'edge_index', 'batch'],
        output_names=['node_colors', 'chromatic_num', 'graph_type', 'difficulty'],
        dynamic_axes={
            # Allow variable number of nodes and edges
            'node_features': {0: 'num_nodes'},
            'edge_index': {1: 'num_edges'},
            'batch': {0: 'num_nodes'},
            'node_colors': {0: 'num_nodes'},
        },
        verbose=False,
    )

    print(f"  ✅ ONNX export complete")


def validate_onnx(onnx_path: str) -> None:
    """
    Validate exported ONNX model

    Checks:
    - Model structure is valid
    - Can be loaded by ONNX Runtime
    - CUDA execution provider available
    """
    print(f"\n[Validate] Checking ONNX model...")

    # Load and check model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"  ✅ ONNX model structure valid")

    # Check file size
    file_size_mb = Path(onnx_path).stat().st_size / (1024 * 1024)
    print(f"  ✅ Model size: {file_size_mb:.2f} MB")

    # Print input/output info
    print(f"\n  Inputs:")
    for input_tensor in onnx_model.graph.input:
        print(f"    {input_tensor.name}: {input_tensor.type}")

    print(f"\n  Outputs:")
    for output_tensor in onnx_model.graph.output:
        print(f"    {output_tensor.name}: {output_tensor.type}")

    # Test with ONNX Runtime
    print(f"\n[ONNX Runtime] Testing inference...")

    # Create session with CUDA provider
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    available_providers = ort.get_available_providers()

    print(f"  Available providers: {available_providers}")

    if 'CUDAExecutionProvider' not in available_providers:
        print(f"  ⚠️  WARNING: CUDA provider not available!")
        print(f"  ⚠️  Rust inference will be CPU-only (not ideal)")

    session = ort.InferenceSession(onnx_path, providers=providers)

    # Check which provider is being used
    used_provider = session.get_providers()[0]
    print(f"  ✅ Using provider: {used_provider}")

    if used_provider != 'CUDAExecutionProvider':
        print(f"  ⚠️  WARNING: Not using CUDA! Check ONNX Runtime GPU installation")

    # Test inference with dummy data
    num_nodes = 50
    num_edges = 150

    dummy_input = {
        'node_features': np.random.randn(num_nodes, 16).astype(np.float32),
        'edge_index': np.random.randint(0, num_nodes, (2, num_edges), dtype=np.int64),
        'batch': np.zeros(num_nodes, dtype=np.int64),
    }

    outputs = session.run(None, dummy_input)

    print(f"\n  Test inference outputs:")
    print(f"    node_colors: {outputs[0].shape}")
    print(f"    chromatic_num: {outputs[1].shape}")
    print(f"    graph_type: {outputs[2].shape}")
    print(f"    difficulty: {outputs[3].shape}")

    print(f"  ✅ ONNX Runtime inference successful!")


def compare_pytorch_onnx(
    pytorch_model: ColoringGNN,
    onnx_path: str,
    num_nodes: int = 50,
    num_edges: int = 150,
    tolerance: float = 1e-4,
) -> bool:
    """
    Compare PyTorch and ONNX outputs to ensure correctness

    Args:
        pytorch_model: Original PyTorch model
        onnx_path: Path to ONNX model
        num_nodes: Test graph size
        num_edges: Test graph edges
        tolerance: Numerical tolerance for comparison

    Returns:
        True if outputs match within tolerance
    """
    print(f"\n[Compare] PyTorch vs ONNX outputs...")

    # Create test input
    x = torch.randn(num_nodes, 16, device=device)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
    batch = torch.zeros(num_nodes, dtype=torch.long, device=device)

    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pt_node_colors, pt_chromatic, pt_type, pt_difficulty = pytorch_model(
            x, edge_index, batch
        )

    # Convert to numpy
    pt_outputs = [
        pt_node_colors.cpu().numpy(),
        pt_chromatic.cpu().numpy(),
        pt_type.cpu().numpy(),
        pt_difficulty.cpu().numpy(),
    ]

    # ONNX inference
    session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])

    onnx_input = {
        'node_features': x.cpu().numpy(),
        'edge_index': edge_index.cpu().numpy(),
        'batch': batch.cpu().numpy(),
    }

    onnx_outputs = session.run(None, onnx_input)

    # Compare outputs
    all_match = True
    output_names = ['node_colors', 'chromatic_num', 'graph_type', 'difficulty']

    for i, name in enumerate(output_names):
        diff = np.abs(pt_outputs[i] - onnx_outputs[i]).max()
        match = diff < tolerance

        status = "✅" if match else "❌"
        print(f"  {status} {name}: max_diff = {diff:.2e}")

        if not match:
            all_match = False

    if all_match:
        print(f"\n  ✅ All outputs match within tolerance ({tolerance})")
    else:
        print(f"\n  ❌ Some outputs differ beyond tolerance!")

    return all_match


def main():
    """Main export script"""

    print("=" * 80)
    print("GNN Model Export to ONNX")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"ONNX: {onnx.__version__}")
    print(f"ONNX Runtime: {ort.__version__}")
    print("=" * 80)

    # Configuration (must match training config)
    config = {
        'input_dim': 16,
        'hidden_dim': 256,
        'num_layers': 6,
        'num_heads': 8,
        'max_colors': 200,
        'num_graph_types': 8,
    }

    # Paths
    checkpoint_path = 'checkpoints/best_model.pt'
    onnx_output_path = 'models/coloring_gnn.onnx'

    # Create output directory
    Path(onnx_output_path).parent.mkdir(exist_ok=True, parents=True)

    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        print(f"\n❌ ERROR: Checkpoint not found: {checkpoint_path}")
        print(f"   Train the model first using: python train.py")
        return 1

    # Load trained model
    model = load_trained_model(checkpoint_path, config)

    # Export to ONNX
    export_to_onnx(model, onnx_output_path)

    # Validate ONNX
    validate_onnx(onnx_output_path)

    # Compare outputs
    if compare_pytorch_onnx(model, onnx_output_path):
        print("\n" + "=" * 80)
        print("✅ SUCCESS: ONNX export complete and validated!")
        print("=" * 80)
        print(f"\nONNX model saved: {onnx_output_path}")
        print(f"\nNext steps:")
        print(f"  1. Copy to Rust project: cp {onnx_output_path} ../../models/")
        print(f"  2. Run Rust inference: cargo run --features cuda")
        print()
        return 0
    else:
        print("\n" + "=" * 80)
        print("❌ WARNING: ONNX outputs differ from PyTorch!")
        print("=" * 80)
        print("Model exported but may have numerical issues.")
        return 1


if __name__ == "__main__":
    exit(main())
