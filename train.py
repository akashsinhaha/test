"""
Training Script for DeepOSWSRM

Supports configurable scale factors and training parameters
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
from datetime import datetime

from deeposwsrm_model import DeepOSWSRM, DeepOSWSRMLoss
from dataset import create_dataloaders


class Trainer:
    """Trainer class for DeepOSWSRM"""
    
    def __init__(self, config):
        """
        Initialize trainer
        
        Args:
            config: Dictionary with configuration parameters
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create model
        self.model = DeepOSWSRM(
            sentinel1_channels=config['sentinel1_channels'],
            sentinel2_channels=config['sentinel2_channels'],
            scale_factor=config['scale_factor'],
            base_channels=config['base_channels']
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {total_params:,}")
        
        # Create loss function
        self.criterion = DeepOSWSRMLoss(
            lambda_weight=config['lambda_weight'],
            eta=config['eta']
        )
        
        # Create optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Create learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=config['learning_rate'] / 100
        )
        
        # Setup directories
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Tensorboard writer
        self.writer = SummaryWriter(self.output_dir / 'tensorboard')
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best model with val loss: {self.best_val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_frac_loss = 0
        total_srm_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            sentinel1 = batch['sentinel1'].to(self.device)
            sentinel2 = batch['sentinel2'].to(self.device)
            water_fraction = batch['water_fraction'].to(self.device)
            water_map = batch['water_map'].to(self.device)
            
            # Forward pass
            pred_fraction, pred_map = self.model(sentinel1, sentinel2)
            
            # Compute loss
            loss, frac_loss, srm_loss = self.criterion(
                pred_fraction, pred_map, water_fraction, water_map
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_frac_loss += frac_loss.item()
            total_srm_loss += srm_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'frac': frac_loss.item(),
                'srm': srm_loss.item()
            })
        
        # Average losses
        avg_loss = total_loss / len(train_loader)
        avg_frac_loss = total_frac_loss / len(train_loader)
        avg_srm_loss = total_srm_loss / len(train_loader)
        
        return avg_loss, avg_frac_loss, avg_srm_loss
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_frac_loss = 0
        total_srm_loss = 0
        
        # Metrics
        total_accuracy = 0
        total_iou = 0
        total_pred_water = 0
        total_target_water = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc='Validation')):
                # Move data to device
                sentinel1 = batch['sentinel1'].to(self.device)
                sentinel2 = batch['sentinel2'].to(self.device)
                water_fraction = batch['water_fraction'].to(self.device)
                water_map = batch['water_map'].to(self.device)
                
                # Forward pass
                pred_fraction, pred_map = self.model(sentinel1, sentinel2)
                
                # Compute loss
                loss, frac_loss, srm_loss = self.criterion(
                    pred_fraction, pred_map, water_fraction, water_map
                )
                
                total_loss += loss.item()
                total_frac_loss += frac_loss.item()
                total_srm_loss += srm_loss.item()
                
                # Compute accuracy and IoU
                pred_class = torch.argmax(pred_map, dim=1)
                target_class = water_map.squeeze(1).long()
                
                accuracy = (pred_class == target_class).float().mean()
                total_accuracy += accuracy.item()
                
                # IoU for water class (class 1)
                pred_water = (pred_class == 1)
                target_water = (target_class == 1)
                
                intersection = (pred_water & target_water).sum().float()
                union = (pred_water | target_water).sum().float()
                
                # Handle edge case where there's no water predicted or in target
                if union > 0:
                    iou = intersection / union
                else:
                    # Both prediction and target have no water - perfect match
                    iou = torch.tensor(1.0)
                
                total_iou += iou.item()
                
                # Track water percentages for debugging
                total_pred_water += pred_water.float().mean().item()
                total_target_water += target_water.float().mean().item()
                
                # Debug info on first batch of first epoch
                if self.current_epoch == 0 and batch_idx == 0:
                    print(f"\n=== DEBUG INFO (Epoch 0, Batch 0) ===")
                    print(f"pred_map shape: {pred_map.shape}")
                    print(f"target_map shape: {water_map.shape}")
                    print(f"pred_class unique: {torch.unique(pred_class)}")
                    print(f"target_class unique: {torch.unique(target_class)}")
                    print(f"Predicted water pixels: {pred_water.sum().item()}/{pred_water.numel()}")
                    print(f"Target water pixels: {target_water.sum().item()}/{target_water.numel()}")
                    print(f"Intersection: {intersection.item()}")
                    print(f"Union: {union.item()}")
                    print(f"IoU: {iou.item():.4f}")
                    print(f"===================================\n")
        
        # Average metrics
        num_batches = len(val_loader)
        avg_loss = total_loss / num_batches
        avg_frac_loss = total_frac_loss / num_batches
        avg_srm_loss = total_srm_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        avg_iou = total_iou / num_batches
        avg_pred_water = total_pred_water / num_batches
        avg_target_water = total_target_water / num_batches
        
        return avg_loss, avg_frac_loss, avg_srm_loss, avg_accuracy, avg_iou, avg_pred_water, avg_target_water
    





    def train(self, train_loader, val_loader):
        """Main training loop"""
        print(f"\nStarting training for {self.config['epochs']} epochs")
        print(f"Scale factor: {self.config['scale_factor']}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Learning rate: {self.config['learning_rate']}")
        
        for epoch in range(self.current_epoch, self.config['epochs']):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_frac_loss, train_srm_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_frac_loss, val_srm_loss, val_acc, val_iou, pred_water_pct, target_water_pct = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Log metrics
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Loss/train_fraction', train_frac_loss, epoch)
            self.writer.add_scalar('Loss/train_srm', train_srm_loss, epoch)
            self.writer.add_scalar('Metrics/val_accuracy', val_acc, epoch)
            self.writer.add_scalar('Metrics/val_iou', val_iou, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Print summary
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_loss:.4f} (Frac: {train_frac_loss:.4f}, SRM: {train_srm_loss:.4f})")
            print(f"  Val Loss: {val_loss:.4f} (Frac: {val_frac_loss:.4f}, SRM: {val_srm_loss:.4f})")
            print(f"  Val Accuracy: {val_acc:.4f}, Val IoU: {val_iou:.4f}")
            print(f"  Water Coverage - Pred: {pred_water_pct:.2%}, Target: {target_water_pct:.2%}")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(is_best=is_best)
            
            # Save training history
            self.train_losses.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'val_iou': val_iou
            })
            
            history_path = self.output_dir / 'training_history.json'
            with open(history_path, 'w') as f:
                json.dump(self.train_losses, f, indent=2)
        
        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        self.writer.close()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train DeepOSWSRM model')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='./deeposwsrm_data',
                       help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Path to output directory')
    
    # Model parameters
    parser.add_argument('--scale_factor', type=int, default=4, choices=[2, 4, 6],
                       help='Super-resolution scale factor')
    parser.add_argument('--sentinel1_channels', type=int, default=2,
                       help='Number of Sentinel-1 channels (VV, VH)')
    parser.add_argument('--sentinel2_channels', type=int, default=4,
                       help='Number of Sentinel-2 channels (B, G, R, NIR)')
    parser.add_argument('--base_channels', type=int, default=64,
                       help='Base number of channels in the network')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay for optimizer')
    parser.add_argument('--lambda_weight', type=float, default=1.0,
                       help='Weight for SRM loss in combined loss')
    parser.add_argument('--eta', type=float, default=-0.5,
                       help='Eta parameter for adaptive loss')
    
    # Data parameters
    parser.add_argument('--patch_size', type=int, default=64,
                       help='Patch size for coarse resolution')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Proportion of data for training')
    
    # Other
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create config dictionary
    config = vars(args)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config['output_dir'] = os.path.join(
        args.output_dir, 
        f"scale{args.scale_factor}_{timestamp}"
    )
    
    # Save config
    os.makedirs(config['output_dir'], exist_ok=True)
    with open(os.path.join(config['output_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        scale_factor=args.scale_factor,
        num_workers=args.num_workers,
        train_split=args.train_split
    )
    
    print(f"Training samples: {len(train_loader) * args.batch_size}")
    print(f"Validation samples: {len(val_loader) * args.batch_size}")
    
    # Create trainer
    trainer = Trainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
