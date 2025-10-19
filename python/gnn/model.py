"""
Multi-Task Graph Attention Network (GATv2) for Graph Coloring

GPU-ONLY: No CPU fallbacks - requires CUDA GPU

Architecture:
- 6 GATv2 layers with 8 attention heads
- Multi-task learning:
  1. Node color prediction (primary - 50% weight)
  2. Chromatic number prediction (25% weight)
  3. Graph type classification (15% weight)
  4. Difficulty regression (10% weight)

Input: Graph with node features
Output: Color predictions + graph-level predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from typing import Tuple

class ColoringGNN(nn.Module):
    """
    Multi-task GATv2 for graph coloring prediction

    GPU Requirements:
    - All tensors on CUDA
    - No CPU fallback operations
    - Validates GPU availability on initialization
    """

    def __init__(
        self,
        input_dim: int = 16,        # Node feature dimension
        hidden_dim: int = 256,      # Hidden layer size
        num_layers: int = 6,        # GAT layers
        num_heads: int = 8,         # Attention heads per layer
        max_colors: int = 200,      # Maximum color prediction
        num_graph_types: int = 8,   # Number of graph types
        dropout: float = 0.1,
    ):
        super().__init__()

        # GPU-ONLY enforcement
        if not torch.cuda.is_available():
            raise RuntimeError("GPU required - NO CPU FALLBACK ALLOWED")

        self.device = torch.device('cuda')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_colors = max_colors
        self.num_graph_types = num_graph_types

        # Node embedding layer
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ).to(self.device)

        # GATv2 layers with multi-head attention
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            # GATv2 with residual connections
            gat = GATv2Conv(
                hidden_dim,
                hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
                add_self_loops=True,
                share_weights=False,
                concat=True  # Concatenate heads
            ).to(self.device)

            norm = nn.LayerNorm(hidden_dim).to(self.device)

            self.gat_layers.append(gat)
            self.norms.append(norm)

        # Task 1: Node-level color prediction (PRIMARY)
        self.color_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, max_colors)
        ).to(self.device)

        # Task 2: Graph-level chromatic number prediction
        self.chromatic_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for mean+max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, max_colors)
        ).to(self.device)

        # Task 3: Graph type classification
        self.type_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_graph_types)
        ).to(self.device)

        # Task 4: Difficulty score regression
        self.difficulty_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output [0, 1], scale to [0, 100] later
        ).to(self.device)

        # Move entire model to GPU
        self.to(self.device)

        print(f"[GNN] Initialized ColoringGNN on {self.device}")
        print(f"[GNN]   Layers: {num_layers}, Heads: {num_heads}, Hidden: {hidden_dim}")
        print(f"[GNN]   Parameters: {sum(p.numel() for p in self.parameters()):,}")

    def forward(
        self,
        x: torch.Tensor,           # Node features [N, input_dim] on GPU
        edge_index: torch.Tensor,  # Edge list [2, E] on GPU
        batch: torch.Tensor,       # Batch assignment [N] on GPU
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass - ALL operations on GPU

        Args:
            x: Node features [N, input_dim] on GPU
            edge_index: Edge list [2, E] on GPU (COO format)
            batch: Batch assignment [N] on GPU

        Returns:
            node_colors: [N, max_colors] - color logits per node
            chromatic_num: [batch_size, max_colors] - chromatic number logits
            graph_type: [batch_size, num_graph_types] - type classification logits
            difficulty: [batch_size, 1] - difficulty score [0, 1]
        """
        # Validate all inputs are on GPU
        assert x.is_cuda, "Node features must be on GPU"
        assert edge_index.is_cuda, "Edge index must be on GPU"
        assert batch.is_cuda, "Batch must be on GPU"

        # Encode node features
        h = self.node_encoder(x)

        # GATv2 layers with residual connections
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.norms)):
            h_new = gat(h, edge_index)
            h_new = norm(h_new)
            h_new = F.elu(h_new)

            # Residual connection (skip first layer)
            if i > 0:
                h = h + h_new
            else:
                h = h_new

        # Task 1: Node-level color prediction
        node_colors = self.color_head(h)

        # Global pooling for graph-level tasks (mean + max)
        graph_emb_mean = global_mean_pool(h, batch)
        graph_emb_max = global_max_pool(h, batch)
        graph_emb = torch.cat([graph_emb_mean, graph_emb_max], dim=1)

        # Task 2: Chromatic number prediction
        chromatic_num = self.chromatic_head(graph_emb)

        # Task 3: Graph type classification
        graph_type = self.type_head(graph_emb)

        # Task 4: Difficulty score
        difficulty = self.difficulty_head(graph_emb) * 100.0  # Scale to [0, 100]

        return node_colors, chromatic_num, graph_type, difficulty

    def predict_coloring(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict coloring for a single graph (inference mode)

        Returns:
            colors: [N] - predicted color for each node
        """
        self.eval()
        with torch.no_grad():
            # Single graph (batch of 1)
            batch = torch.zeros(x.size(0), dtype=torch.long, device=self.device)

            node_colors, _, _, _ = self.forward(x, edge_index, batch)

            # Take argmax for color prediction
            colors = node_colors.argmax(dim=1)

            return colors


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with adaptive task weighting

    Combines:
    - Node color classification (cross-entropy)
    - Chromatic number prediction (cross-entropy)
    - Graph type classification (cross-entropy)
    - Difficulty regression (MSE)
    """

    def __init__(
        self,
        color_weight: float = 0.5,
        chromatic_weight: float = 0.25,
        type_weight: float = 0.15,
        difficulty_weight: float = 0.10,
        label_smoothing: float = 0.1,
    ):
        super().__init__()

        self.color_weight = color_weight
        self.chromatic_weight = chromatic_weight
        self.type_weight = type_weight
        self.difficulty_weight = difficulty_weight

        # Cross-entropy losses with label smoothing
        self.color_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.chromatic_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.type_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # MSE for difficulty
        self.difficulty_loss_fn = nn.MSELoss()

    def forward(
        self,
        pred_colors: torch.Tensor,      # [N, max_colors]
        pred_chromatic: torch.Tensor,   # [batch_size, max_colors]
        pred_type: torch.Tensor,        # [batch_size, num_types]
        pred_difficulty: torch.Tensor,  # [batch_size, 1]
        true_colors: torch.Tensor,      # [N]
        true_chromatic: torch.Tensor,   # [batch_size]
        true_type: torch.Tensor,        # [batch_size]
        true_difficulty: torch.Tensor,  # [batch_size]
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute multi-task loss

        Returns:
            total_loss: Weighted sum of all task losses
            losses_dict: Individual task losses for logging
        """
        # Task 1: Node color classification
        color_loss = self.color_loss_fn(pred_colors, true_colors)

        # Task 2: Chromatic number prediction
        chromatic_loss = self.chromatic_loss_fn(pred_chromatic, true_chromatic)

        # Task 3: Graph type classification
        type_loss = self.type_loss_fn(pred_type, true_type)

        # Task 4: Difficulty regression
        difficulty_loss = self.difficulty_loss_fn(
            pred_difficulty.squeeze(),
            true_difficulty.float()
        )

        # Weighted combination
        total_loss = (
            self.color_weight * color_loss +
            self.chromatic_weight * chromatic_loss +
            self.type_weight * type_loss +
            self.difficulty_weight * difficulty_loss
        )

        losses_dict = {
            'total': total_loss.item(),
            'color': color_loss.item(),
            'chromatic': chromatic_loss.item(),
            'type': type_loss.item(),
            'difficulty': difficulty_loss.item(),
        }

        return total_loss, losses_dict


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def validate_gpu_only(model: nn.Module) -> bool:
    """
    Validate that model is entirely on GPU with no CPU tensors
    """
    for name, param in model.named_parameters():
        if not param.is_cuda:
            print(f"[ERROR] Parameter {name} is on CPU!")
            return False

    for name, buffer in model.named_buffers():
        if not buffer.is_cuda:
            print(f"[ERROR] Buffer {name} is on CPU!")
            return False

    return True


if __name__ == "__main__":
    # Test model creation
    print("=" * 80)
    print("Testing ColoringGNN Model")
    print("=" * 80)

    # Check GPU
    if not torch.cuda.is_available():
        print("❌ GPU not available - cannot test model")
        exit(1)

    print(f"✅ GPU available: {torch.cuda.get_device_name()}")
    print(f"✅ CUDA version: {torch.version.cuda}")

    # Create model
    model = ColoringGNN(
        input_dim=16,
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        max_colors=200,
        num_graph_types=8,
    )

    # Validate GPU-only
    print("\n[Validation] Checking GPU-only architecture...")
    if validate_gpu_only(model):
        print("✅ All model parameters on GPU")
    else:
        print("❌ CPU tensors detected!")
        exit(1)

    # Test forward pass with dummy data
    print("\n[Test] Forward pass with dummy graph...")

    n = 100  # 100 nodes
    e = 300  # 300 edges
    batch_size = 4

    # Create dummy batch (4 graphs of 25 nodes each)
    x = torch.randn(n, 16, device='cuda')
    edge_index = torch.randint(0, n, (2, e), device='cuda')
    batch = torch.repeat_interleave(
        torch.arange(batch_size, device='cuda'),
        n // batch_size
    )

    # Forward pass
    node_colors, chromatic, graph_type, difficulty = model(x, edge_index, batch)

    print(f"  Input shapes:")
    print(f"    Node features: {x.shape}")
    print(f"    Edge index: {edge_index.shape}")
    print(f"    Batch: {batch.shape}")

    print(f"  Output shapes:")
    print(f"    Node colors: {node_colors.shape}")
    print(f"    Chromatic: {chromatic.shape}")
    print(f"    Graph type: {graph_type.shape}")
    print(f"    Difficulty: {difficulty.shape}")

    print(f"\n  All outputs on GPU: {all([
        node_colors.is_cuda,
        chromatic.is_cuda,
        graph_type.is_cuda,
        difficulty.is_cuda
    ])}")

    # Test loss
    print("\n[Test] Multi-task loss...")
    loss_fn = MultiTaskLoss()

    true_colors = torch.randint(0, 200, (n,), device='cuda')
    true_chromatic = torch.randint(3, 25, (batch_size,), device='cuda')
    true_type = torch.randint(0, 8, (batch_size,), device='cuda')
    true_difficulty = torch.rand(batch_size, device='cuda') * 100.0

    total_loss, losses = loss_fn(
        node_colors, chromatic, graph_type, difficulty,
        true_colors, true_chromatic, true_type, true_difficulty
    )

    print(f"  Total loss: {total_loss.item():.4f}")
    for task, loss_val in losses.items():
        print(f"    {task}: {loss_val:.4f}")

    print(f"\n✅ All tests passed!")
    print(f"✅ Model ready for training")
    print("=" * 80)
