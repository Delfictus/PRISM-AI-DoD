"""
Export trained GNN model to ONNX format

Exports the best checkpoint for Rust inference via ONNX Runtime.
"""

import argparse
import torch
import onnx
from pathlib import Path

from model import MultiTaskGATv2
from torch_geometric.data import Data, Batch


def export_to_onnx(checkpoint_path: str, output_path: str, opset_version: int = 17):
    """
    Export PyTorch model to ONNX format

    Args:
        checkpoint_path: Path to best_model.pt checkpoint
        output_path: Path to save .onnx file
        opset_version: ONNX opset version (17 for CUDA compatibility)
    """
    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    args_dict = checkpoint['args']

    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val loss: {checkpoint['val_loss']:.4f}")
    print(f"  Val metrics: {checkpoint['val_metrics']}")

    # Create model
    model = MultiTaskGATv2(
        node_feature_dim=16,
        hidden_dim=args_dict.get('hidden_dim', 256),
        num_gnn_layers=args_dict.get('num_layers', 6),
        num_attention_heads=args_dict.get('num_heads', 8),
        max_colors=args_dict.get('max_colors', 200),
        num_graph_types=8,
        dropout=0.0,  # No dropout for inference
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"\nModel architecture:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # Create dummy input for tracing
    print(f"\nCreating dummy input for export...")

    # Dummy graph: 100 nodes, random edges
    num_nodes = 100
    x = torch.randn(num_nodes, 16)
    edge_index = torch.randint(0, num_nodes, (2, 200))  # 200 random edges

    # Create batch
    data = Data(x=x, edge_index=edge_index)
    batch = Batch.from_data_list([data])

    print(f"  Dummy input shape: {batch.x.shape}")
    print(f"  Dummy edge_index shape: {batch.edge_index.shape}")

    # Test forward pass
    with torch.no_grad():
        output = model(batch)
    print(f"\nTest forward pass:")
    print(f"  Color logits: {output['color_logits'].shape}")
    print(f"  Chromatic: {output['chromatic'].shape}")
    print(f"  Graph type logits: {output['graph_type_logits'].shape}")
    print(f"  Difficulty: {output['difficulty'].shape}")

    # Export to ONNX
    print(f"\nExporting to ONNX...")
    print(f"  Output path: {output_path}")
    print(f"  ONNX opset version: {opset_version}")

    try:
        torch.onnx.export(
            model,
            batch,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['x', 'edge_index', 'batch'],
            output_names=['color_logits', 'chromatic', 'graph_type_logits', 'difficulty'],
            dynamic_axes={
                'x': {0: 'num_nodes'},
                'edge_index': {1: 'num_edges'},
                'batch': {0: 'num_nodes'},
                'color_logits': {0: 'num_nodes'},
            }
        )
        print(f"  ✅ ONNX export successful!")
    except Exception as e:
        print(f"  ❌ ONNX export failed: {e}")
        print(f"\n  Trying simplified export without dynamic axes...")

        # Fallback: export without dynamic axes
        torch.onnx.export(
            model,
            batch,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['x', 'edge_index', 'batch'],
            output_names=['color_logits', 'chromatic', 'graph_type_logits', 'difficulty'],
        )
        print(f"  ✅ Simplified ONNX export successful!")

    # Verify ONNX model
    print(f"\nVerifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"  ✅ ONNX model is valid!")

    # Get model size
    model_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  Model size: {model_size:.2f} MB")

    print("\n" + "="*80)
    print(f"✅ ONNX export complete!")
    print(f"  Saved to: {output_path}")
    print(f"  Ready for Rust ONNX Runtime integration")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pt',
                        help='Path to checkpoint file')
    parser.add_argument('--output', type=str, default='../../models/coloring_gnn.onnx',
                        help='Output ONNX file path')
    parser.add_argument('--opset', type=int, default=17,
                        help='ONNX opset version')

    args = parser.parse_args()

    # Create output directory
    Path(args.output).parent.mkdir(exist_ok=True, parents=True)

    export_to_onnx(args.checkpoint, args.output, args.opset)
