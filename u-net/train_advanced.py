"""
Advanced Training Script for Stroke Lesion Segmentation (local CLI)

The command line takes only three arguments:
    --dataset_dir : Root directory containing NPZ volumes and split files
    --output_dir  : Destination for checkpoints and results
    --config      : YAML file with all hyperparameters
"""

import argparse
import json
import os
from typing import Dict, Any

import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from advanced_dataset import ISLESAdvancedDataset
from advanced_losses import CombinedLoss, GatingLoss
from advanced_unet import AdvancedUNet
from infer import visualize_predictions


def dice_coefficient(pred, target, smooth=1.0):
    """Calculate Dice coefficient"""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def evaluate_on_lesion_slices(model, dataloader, device, criterion, gating_criterion):
    """
    Evaluate model performance on lesion-positive slices only.
    This is the key metric for comparison.
    """
    model.eval()
    
    total_loss = 0
    total_dice = 0
    n_lesion_slices = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            has_lesion = batch['has_lesion']
            
            # Forward pass
            logits, gate_logits = model(images)
            
            # Only compute metrics on lesion-positive slices
            lesion_mask = torch.tensor(has_lesion).to(device)
            lesion_indices = lesion_mask.nonzero(as_tuple=True)[0]
            
            if len(lesion_indices) == 0:
                continue
            
            # Select only lesion-positive samples
            logits_lesion = logits[lesion_indices]
            masks_lesion = masks[lesion_indices]
            gate_logits_lesion = gate_logits[lesion_indices]
            
            # Calculate losses
            seg_loss, _ = criterion(logits_lesion, masks_lesion, gate_logits_lesion)
            gate_loss = gating_criterion(gate_logits_lesion, masks_lesion)
            loss = seg_loss + 0.1 * gate_loss
            
            # Calculate Dice
            probs = torch.sigmoid(logits_lesion)
            for i in range(probs.shape[0]):
                dice = dice_coefficient(probs[i, 0], masks_lesion[i])
                total_dice += dice.item()
                n_lesion_slices += 1
            
            total_loss += loss.item() * len(lesion_indices)
            
            # Store for visualization
            all_preds.append(probs.cpu())
            all_targets.append(masks_lesion.cpu())
    
    avg_loss = total_loss / max(n_lesion_slices, 1)
    avg_dice = total_dice / max(n_lesion_slices, 1)
    
    return avg_loss, avg_dice, all_preds, all_targets


def train_epoch(model, dataloader, device, optimizer, criterion, gating_criterion, epoch):
    """Train for one epoch"""
    model.train()

    total_loss = 0.0
    total_dice = 0.0
    lesion_samples = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        optimizer.zero_grad()
        logits, gate_logits = model(images)

        seg_loss, loss_components = criterion(logits, masks, gate_logits)
        gate_loss = gating_criterion(gate_logits, masks)
        loss = seg_loss + 0.1 * gate_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            has_lesion = batch['has_lesion']
            lesion_indices = [i for i, hl in enumerate(has_lesion) if hl]
            lesion_samples += len(lesion_indices)
            for idx in lesion_indices:
                dice = dice_coefficient(probs[idx, 0], masks[idx])
                total_dice += dice.item()

        total_loss += loss.item()

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'focal': f"{loss_components['focal']:.4f}",
            'dice': f"{loss_components['dice']:.4f}",
            'tversky': f"{loss_components['tversky']:.4f}"
        })

    n_batches = max(len(dataloader), 1)
    avg_loss = total_loss / n_batches
    avg_dice = total_dice / max(lesion_samples, 1)

    return avg_loss, avg_dice


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config and merge with safe defaults."""
    defaults = {
        'modality': 'all',
        'image_size': 224,
        'batch_size': 16,
        'epochs': 40,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'lr_patience': 8,
        'num_workers': 4,
        'resume': False,
        'keep_checkpoint': False,
        'balance_ratio': 0.3,
        'min_lesion_size': 3,
        'apply_crop': False,
        'crop_margin': 10,
        'focal_weight': 1.0,
        'dice_weight': 1.0,
        'tversky_weight': 1.0,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        'tversky_alpha': 0.3,
        'tversky_beta': 0.7,
        'npz_subdir': 'all',
        'splits_subdir': 'splits',
        'train_split': 'train.txt',
        'val_split': 'val.txt'
    }

    with open(config_path, 'r') as f:
        loaded = yaml.safe_load(f) or {}

    merged = {**defaults, **loaded}
    return merged


def main(dataset_dir: str, output_dir: str, cfg: Dict[str, Any]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset_dir = os.path.abspath(dataset_dir)
    output_dir = os.path.abspath(output_dir)

    npz_dir = os.path.join(dataset_dir, cfg['npz_subdir'])
    splits_dir = os.path.join(dataset_dir, cfg['splits_subdir'])
    train_split = os.path.join(splits_dir, cfg['train_split'])
    val_split = os.path.join(splits_dir, cfg['val_split'])

    if not os.path.isdir(npz_dir):
        raise FileNotFoundError(f"NPZ directory not found: {npz_dir}")
    if not os.path.isdir(splits_dir):
        raise FileNotFoundError(f"Splits directory not found: {splits_dir}")
    if not os.path.isfile(train_split) or not os.path.isfile(val_split):
        raise FileNotFoundError("train.txt or val.txt missing under splits directory")

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'used_config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f)

    print("\nCreating datasets...")
    train_dataset = ISLESAdvancedDataset(
        npz_dir=npz_dir,
        split_file=train_split,
        target_size=cfg['image_size'],
        modality=cfg['modality'],
        balance_ratio=cfg['balance_ratio'],
        min_lesion_size=cfg['min_lesion_size'],
        remove_empty_brain=True,
        apply_crop=cfg['apply_crop'],
        crop_margin=cfg['crop_margin']
    )

    val_dataset = ISLESAdvancedDataset(
        npz_dir=npz_dir,
        split_file=val_split,
        target_size=cfg['image_size'],
        modality=cfg['modality'],
        balance_ratio=0.5,
        min_lesion_size=cfg['min_lesion_size'],
        remove_empty_brain=True,
        apply_crop=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=cfg['num_workers'],
        pin_memory=True
    )

    print("\nCreating model...")
    n_channels = 1 if cfg['modality'] in ['dwi', 'adc'] else 3
    model = AdvancedUNet(
        n_channels=n_channels,
        n_classes=1,
        bilinear=False,
        use_gating=True
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = CombinedLoss(
        focal_weight=cfg['focal_weight'],
        dice_weight=cfg['dice_weight'],
        tversky_weight=cfg['tversky_weight'],
        focal_alpha=cfg['focal_alpha'],
        focal_gamma=cfg['focal_gamma'],
        tversky_alpha=cfg['tversky_alpha'],
        tversky_beta=cfg['tversky_beta']
    )
    gating_criterion = GatingLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay']
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=cfg['lr_patience'],
        min_lr=1e-6
    )

    history = {
        'train_loss': [],
        'train_dice': [],
        'val_loss': [],
        'val_dice': [],
        'lr': []
    }

    start_epoch = 0
    best_val_dice = 0.0
    checkpoint_path = os.path.join(output_dir, 'checkpoint.pth')

    if os.path.exists(checkpoint_path) and cfg['resume']:
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_dice = checkpoint['best_val_dice']
        history = checkpoint['history']
        print(f"Resumed from epoch {start_epoch}, best val dice: {best_val_dice:.4f}")

    print(f"\nStarting training for {cfg['epochs']} epochs...")
    print("=" * 70)

    for epoch in range(start_epoch, cfg['epochs']):
        print(f"\nEpoch {epoch + 1}/{cfg['epochs']}")
        print("-" * 70)

        train_loss, train_dice = train_epoch(
            model, train_loader, device, optimizer, criterion, gating_criterion, epoch + 1
        )

        val_loss, val_dice, _, _ = evaluate_on_lesion_slices(
            model, val_loader, device, criterion, gating_criterion
        )

        scheduler.step(val_dice)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['lr'].append(current_lr)

        print(f"Epoch {epoch + 1}/{cfg['epochs']} | Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_dice': best_val_dice,
            'history': history,
            'config': cfg,
            'dataset_dir': dataset_dir
        }
        torch.save(checkpoint, checkpoint_path)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_model_path = os.path.join(output_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ New best model saved! (Dice: {val_dice:.4f})")

        with open(os.path.join(output_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)

    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth'), map_location=device))

    val_loss, val_dice, _, _ = evaluate_on_lesion_slices(
        model, val_loader, device, criterion, gating_criterion
    )

    print("\nBest Model Performance:")
    print(f"  Validation Dice (lesion-positive slices): {val_dice:.4f}")
    print("  Target: > 0.660 (nnU-Net baseline)")

    print("\nGenerating visualizations...")
    visualize_predictions(model, val_loader, device, output_dir, num_samples=10)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_dice'], label='Train Dice')
    plt.plot(history['val_dice'], label='Val Dice')
    plt.axhline(y=0.660, color='r', linestyle='--', label='nnU-Net baseline')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.legend()
    plt.title('Training and Validation Dice')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
    plt.close()

    print(f"\n✓ All results saved to: {output_dir}")

    if not cfg['keep_checkpoint']:
        if os.path.exists(checkpoint_path):
            checkpoint_size = os.path.getsize(checkpoint_path) / (1024 ** 2)
            os.remove(checkpoint_path)
            print(f"\n✓ Checkpoint cleaned up to save space ({checkpoint_size:.1f} MB freed)")
            print("  Deleted: checkpoint.pth")
            best_model_path = os.path.join(output_dir, 'best_model.pth')
            if os.path.exists(best_model_path):
                best_size = os.path.getsize(best_model_path) / (1024 ** 2)
                print(f"  Retained: best_model.pth ({best_size:.1f} MB)")
            print("\n  Note: set keep_checkpoint: true in config to retain checkpoint.pth")
    else:
        print("\n✓ Checkpoint retained (keep_checkpoint=true)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Advanced U-Net training for stroke lesion segmentation (local)'
    )
    parser.add_argument('dataset_dir', help='Root directory of the dataset')
    parser.add_argument('output_dir', help='Output directory for checkpoints/results')
    parser.add_argument('--config', default='./config.yaml', help='YAML file with hyperparameters')

    args = parser.parse_args()

    config = load_config(args.config)
    main(args.dataset_dir, args.output_dir, config)
