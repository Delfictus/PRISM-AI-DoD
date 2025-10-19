"""
Multi-Task GNN Training Script for Graph Coloring

Professional training pipeline optimized for RunPod H100.

Features:
- Multi-task learning (4 tasks)
- Early stopping
- Learning rate scheduling
- Gradient clipping
- TensorBoard logging
- Automatic checkpointing
- Mixed precision training (AMP)
"""

import argparse
import time
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from dataset import GraphColoringDataset, collate_batch
from model import MultiTaskGATv2, MultiTaskLoss


def train_epoch(model, loader, optimizer, loss_fn, device, scaler, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_losses = {k: 0 for k in ['total', 'color', 'chromatic', 'graph_type', 'difficulty']}
    num_batches = len(loader)

    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)

        # Prepare targets
        targets = {
            'y_colors': batch.y_colors,
            'y_chromatic': batch.y_chromatic,
            'y_graph_type': batch.y_graph_type,
            'y_difficulty': batch.y_difficulty,
        }

        # Forward pass with mixed precision
        optimizer.zero_grad()

        with autocast():
            predictions = model(batch)
            loss, losses = loss_fn(predictions, targets)

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        # Accumulate losses
        for k, v in losses.items():
            total_losses[k] += v

        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch [{batch_idx+1}/{num_batches}] "
                  f"Loss: {losses['total']:.4f} "
                  f"(color: {losses['color']:.4f}, "
                  f"chromatic: {losses['chromatic']:.4f}, "
                  f"type: {losses['graph_type']:.4f}, "
                  f"diff: {losses['difficulty']:.4f})")

    # Average losses
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    return avg_losses


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    """Validate model"""
    model.eval()
    total_losses = {k: 0 for k in ['total', 'color', 'chromatic', 'graph_type', 'difficulty']}
    num_batches = len(loader)

    # Additional metrics
    chromatic_mae = 0
    color_accuracy = 0
    graph_type_accuracy = 0

    for batch in loader:
        batch = batch.to(device)

        targets = {
            'y_colors': batch.y_colors,
            'y_chromatic': batch.y_chromatic,
            'y_graph_type': batch.y_graph_type,
            'y_difficulty': batch.y_difficulty,
        }

        with autocast():
            predictions = model(batch)
            loss, losses = loss_fn(predictions, targets)

        # Accumulate losses
        for k, v in losses.items():
            total_losses[k] += v

        # Additional metrics
        chromatic_mae += torch.abs(predictions['chromatic'] - targets['y_chromatic']).mean().item()

        color_preds = predictions['color_logits'].argmax(dim=1)
        color_accuracy += (color_preds == targets['y_colors']).float().mean().item()

        type_preds = predictions['graph_type_logits'].argmax(dim=1)
        graph_type_accuracy += (type_preds == targets['y_graph_type'].squeeze()).float().mean().item()

    # Average metrics
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    metrics = {
        'chromatic_mae': chromatic_mae / num_batches,
        'color_accuracy': color_accuracy / num_batches,
        'graph_type_accuracy': graph_type_accuracy / num_batches,
    }

    return avg_losses, metrics


def main(args):
    print("="*80)
    print("Multi-Task GNN Training for Graph Coloring")
    print("="*80)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Datasets
    print(f"\nLoading datasets from: {args.data_dir}")
    train_dataset = GraphColoringDataset(args.data_dir, split='train', max_colors=args.max_colors)
    val_dataset = GraphColoringDataset(args.data_dir, split='val', max_colors=args.max_colors)

    print(f"  Train: {len(train_dataset)} graphs")
    print(f"  Val:   {len(val_dataset)} graphs")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )

    # Model
    print(f"\nInitializing model...")
    model = MultiTaskGATv2(
        node_feature_dim=16,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        max_colors=args.max_colors,
        num_graph_types=8,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # Loss function
    loss_fn = MultiTaskLoss(
        color_weight=0.5,
        chromatic_weight=0.25,
        graph_type_weight=0.15,
        difficulty_weight=0.1,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True,
    )

    # Mixed precision scaler
    scaler = GradScaler()

    # TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Early stopping patience: {args.early_stop_patience}")
    print("="*80)

    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        print(f"\nEpoch [{epoch+1}/{args.epochs}]")

        # Train
        train_losses = train_epoch(model, train_loader, optimizer, loss_fn, device, scaler, epoch)
        print(f"  Train Loss: {train_losses['total']:.4f} "
              f"(color: {train_losses['color']:.4f}, "
              f"chromatic: {train_losses['chromatic']:.4f}, "
              f"type: {train_losses['graph_type']:.4f}, "
              f"diff: {train_losses['difficulty']:.4f})")

        # Validate
        val_losses, val_metrics = validate(model, val_loader, loss_fn, device)
        print(f"  Val Loss:   {val_losses['total']:.4f} "
              f"(color: {val_losses['color']:.4f}, "
              f"chromatic: {val_losses['chromatic']:.4f}, "
              f"type: {val_losses['graph_type']:.4f}, "
              f"diff: {val_losses['difficulty']:.4f})")
        print(f"  Val Metrics: "
              f"Chromatic MAE: {val_metrics['chromatic_mae']:.2f}, "
              f"Color Acc: {val_metrics['color_accuracy']:.3f}, "
              f"Type Acc: {val_metrics['graph_type_accuracy']:.3f}")

        # Learning rate scheduling
        scheduler.step(val_losses['total'])

        # TensorBoard logging
        for k, v in train_losses.items():
            writer.add_scalar(f'train/{k}', v, epoch)
        for k, v in val_losses.items():
            writer.add_scalar(f'val/{k}', v, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f'val_metrics/{k}', v, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # Checkpointing
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_losses['total'],
                'val_metrics': val_metrics,
                'args': vars(args),
            }
            torch.save(checkpoint, args.checkpoint_dir / 'best_model.pt')
            print(f"  ✅ New best model saved (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.early_stop_patience})")

        # Early stopping
        if patience_counter >= args.early_stop_patience:
            print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
            break

        epoch_time = time.time() - epoch_start
        print(f"  Epoch time: {epoch_time:.1f}s")

    total_time = time.time() - start_time
    print("\n" + "="*80)
    print(f"✅ Training complete!")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Best model saved to: {args.checkpoint_dir / 'best_model.pt'}")
    print("="*80)

    writer.close()

    # Save final metadata
    metadata = {
        'best_val_loss': best_val_loss,
        'total_epochs': epoch + 1,
        'total_time_minutes': total_time / 60,
        'final_lr': optimizer.param_groups[0]['lr'],
        'model_params': total_params,
    }
    with open(args.checkpoint_dir / 'training_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data-dir', type=str, default='../../training_data',
                        help='Path to training data directory')
    parser.add_argument('--max-colors', type=int, default=200,
                        help='Maximum number of colors')

    # Model
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension size')
    parser.add_argument('--num-layers', type=int, default=6,
                        help='Number of GNN layers')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--early-stop-patience', type=int, default=15,
                        help='Early stopping patience')

    # System
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='./logs',
                        help='TensorBoard log directory')

    args = parser.parse_args()

    # Create directories
    args.checkpoint_dir = Path(args.checkpoint_dir)
    args.checkpoint_dir.mkdir(exist_ok=True, parents=True)
    Path(args.log_dir).mkdir(exist_ok=True, parents=True)

    main(args)
