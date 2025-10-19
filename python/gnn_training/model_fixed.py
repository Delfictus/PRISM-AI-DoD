"""
Multi-Task GATv2 for Graph Coloring - FIXED

Fixed difficulty score normalization to prevent gradient explosion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch


class MultiTaskGATv2(nn.Module):
    """Multi-task GATv2 for graph coloring"""

    def __init__(
        self,
        node_feature_dim: int = 16,
        hidden_dim: int = 256,
        num_gnn_layers: int = 6,
        num_attention_heads: int = 8,
        max_colors: int = 200,
        num_graph_types: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.num_heads = num_attention_heads
        self.max_colors = max_colors
        self.num_graph_types = num_graph_types
        self.dropout = dropout

        # Input projection
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)

        # GATv2 backbone
        self.gat_layers = nn.ModuleList()
        for i in range(num_gnn_layers):
            in_channels = hidden_dim
            out_channels = hidden_dim // num_attention_heads

            self.gat_layers.append(
                GATv2Conv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=num_attention_heads,
                    dropout=dropout,
                    concat=True,
                    edge_dim=None,
                )
            )

        # Layer normalization for each GATv2 layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_gnn_layers)
        ])

        # Task-specific heads

        # 1. Node color prediction (per-node classification)
        self.color_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max_colors)
        )

        # 2. Chromatic number prediction (graph-level regression)
        self.chromatic_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # 3. Graph type classification (graph-level)
        self.graph_type_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_graph_types)
        )

        # 4. Difficulty score prediction (graph-level regression)
        self.difficulty_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, data: Batch):
        """Forward pass"""
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Input projection
        h = self.input_proj(x)

        # GATv2 layers with residual connections
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            h_new = gat(h, edge_index)
            h_new = F.elu(h_new)
            h_new = norm(h_new)
            h = h + h_new
            h = F.dropout(h, p=self.dropout, training=self.training)

        node_embeddings = h

        # Task 1: Node color prediction
        color_logits = self.color_head(node_embeddings)

        # Graph-level pooling
        graph_mean = global_mean_pool(node_embeddings, batch)
        graph_max = global_max_pool(node_embeddings, batch)
        graph_repr = torch.cat([graph_mean, graph_max], dim=-1)

        # Task 2: Chromatic number
        chromatic_pred = self.chromatic_head(graph_repr).squeeze(-1)

        # Task 3: Graph type
        graph_type_logits = self.graph_type_head(graph_repr)

        # Task 4: Difficulty score
        difficulty_pred = self.difficulty_head(graph_repr).squeeze(-1)

        return {
            'color_logits': color_logits,
            'chromatic': chromatic_pred,
            'graph_type_logits': graph_type_logits,
            'difficulty': difficulty_pred,
        }


class MultiTaskLoss(nn.Module):
    """Multi-task loss with FIXED difficulty normalization"""

    def __init__(
        self,
        color_weight: float = 0.5,
        chromatic_weight: float = 0.25,
        graph_type_weight: float = 0.15,
        difficulty_weight: float = 0.1,
    ):
        super().__init__()
        self.color_weight = color_weight
        self.chromatic_weight = chromatic_weight
        self.graph_type_weight = graph_type_weight
        self.difficulty_weight = difficulty_weight

        self.color_loss_fn = nn.CrossEntropyLoss()
        self.chromatic_loss_fn = nn.L1Loss()
        self.graph_type_loss_fn = nn.CrossEntropyLoss()
        self.difficulty_loss_fn = nn.MSELoss()

    def forward(self, predictions, targets):
        """Compute multi-task loss"""

        # Task 1: Node color prediction
        color_loss = self.color_loss_fn(
            predictions['color_logits'],
            targets['y_colors']
        )

        # Task 2: Chromatic number (regression)
        chromatic_loss = self.chromatic_loss_fn(
            predictions['chromatic'],
            targets['y_chromatic']
        )

        # Task 3: Graph type classification
        graph_type_loss = self.graph_type_loss_fn(
            predictions['graph_type_logits'],
            targets['y_graph_type'].squeeze()
        )

        # Task 4: Difficulty score (NORMALIZED by 100 to prevent explosion)
        # Original difficulty is 0-100, but data might have larger values
        # Normalize both prediction and target by dividing by 100
        normalized_pred = predictions['difficulty'] / 100.0
        normalized_target = targets['y_difficulty'] / 100.0
        difficulty_loss = self.difficulty_loss_fn(
            normalized_pred,
            normalized_target
        )

        # Weighted sum
        total_loss = (
            self.color_weight * color_loss +
            self.chromatic_weight * chromatic_loss +
            self.graph_type_weight * graph_type_loss +
            self.difficulty_weight * difficulty_loss  # Already normalized, safe now
        )

        losses = {
            'total': total_loss.item(),
            'color': color_loss.item(),
            'chromatic': chromatic_loss.item(),
            'graph_type': graph_type_loss.item(),
            'difficulty': difficulty_loss.item(),  # This will be small now
        }

        return total_loss, losses
