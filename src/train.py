import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import json
import argparse
from datetime import datetime
import logging
from datasets import SkinCancerDataset
from utils import setup_logger


logger = setup_logger()

class EfficientNetTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
        
        logging.info(f"Using device: {self.device}")
        
        self.model = self._create_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
        self.best_model_state = None
        
    def _create_model(self):
        """Create EfficientNet model with custom classifier"""
        # Load pretrained EfficientNet
        model = timm.create_model(
            self.config['model_name'], 
            pretrained=True, 
            num_classes=2  # Binary classification
        )
        
        # Freeze early layers if specified
        if self.config.get('freeze_backbone', False):
            for param in model.parameters():
                param.requires_grad = False
            
            # Unfreeze classifier
            for param in model.classifier.parameters():
                param.requires_grad = True
        
        model = model.to(self.device)
        logging.info(f"Created {self.config['model_name']} with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
        return model
    
    def _create_optimizer(self):
        """Create optimizer based on config"""
        if self.config['optimizer'] == 'adam':
            return optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'] == 'sgd':
            return optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config['learning_rate'],
                momentum=0.9,
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'] == 'adamw':
            return optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if self.config['scheduler'] == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config['epochs']
            )
        elif self.config['scheduler'] == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config['step_size'],
                gamma=self.config['gamma']
            )
        elif self.config['scheduler'] == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
        return None
    
    def get_transforms(self, phase='train'):
        """Get data transforms for different phases"""
        if phase == 'train':
            return transforms.Compose([
                transforms.Resize((self.config['img_size'], self.config['img_size'])),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.config['img_size'], self.config['img_size'])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def create_data_loaders(self):
        """Create train, validation, and test data loaders"""
        # Train dataset and loader
        train_dataset = SkinCancerDataset(
            root_dir=os.path.join(self.config['data_dir'], 'processed', 'train'),
            transform=self.get_transforms('train')
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        val_dataset = SkinCancerDataset(
            root_dir=os.path.join(self.config['data_dir'], 'processed', 'val'),
            transform=self.get_transforms('val')
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        # Test dataset and loader
        test_dataset = SkinCancerDataset(
            root_dir=os.path.join(self.config['data_dir'], 'processed', 'test'),
            transform=self.get_transforms('test')
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        logging.info(f"Train samples: {len(train_dataset)}")
        logging.info(f"Validation samples: {len(val_dataset)}")
        logging.info(f"Test samples: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{accuracy_score(all_labels, all_preds):.4f}'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{accuracy_score(all_labels, all_preds):.4f}'
                })
        
        epoch_loss = running_loss / len(val_loader)
        metrics = self.calculate_metrics(all_labels, all_preds, all_probs)
        
        return epoch_loss, metrics
    
    def calculate_metrics(self, true_labels, predictions, probabilities):
        """Calculate comprehensive metrics"""
        probs_positive_class = np.array(probabilities)[:, 1]  # Probability of malignant class
        
        metrics = {
            'accuracy': accuracy_score(true_labels, predictions),
            'precision': precision_score(true_labels, predictions, average='weighted'),
            'recall': recall_score(true_labels, predictions, average='weighted'),
            'f1': f1_score(true_labels, predictions, average='weighted'),
            'auc': roc_auc_score(true_labels, probs_positive_class),
            'sensitivity': recall_score(true_labels, predictions, pos_label=1),
            'specificity': recall_score(true_labels, predictions, pos_label=0)
        }
        
        return metrics
    
    def train(self):
        """Main training loop"""
        logging.info("Starting training...")
        logging.info(f"Configuration: {self.config}")
        
        # Create data loaders
        train_loader, val_loader, test_loader = self.create_data_loaders()
        
        # Training loop
        for epoch in range(self.config['epochs']):
            logging.info(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validation phase
            val_loss, val_metrics = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_metrics['accuracy'])
            
            # Log metrics
            logging.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logging.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            logging.info(f"Val AUC: {val_metrics['auc']:.4f}, Val F1: {val_metrics['f1']:.4f}")
            logging.info(f"Val Sensitivity: {val_metrics['sensitivity']:.4f}, Val Specificity: {val_metrics['specificity']:.4f}")
            
            # Save best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.best_model_state = self.model.state_dict().copy()
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                logging.info(f"New best model saved with validation accuracy: {self.best_val_acc:.4f}")
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['accuracy'])
                else:
                    self.scheduler.step()
                    
                current_lr = self.optimizer.param_groups[0]['lr']
                logging.info(f"Current learning rate: {current_lr:.6f}")
            
            # Save regular checkpoint
            if (epoch + 1) % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(epoch, val_metrics, is_best=False)
        
        # Load best model and evaluate on test set
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            test_metrics = self.evaluate(test_loader, 'test')
            logging.info(f"Final Test Metrics: {test_metrics}")
        
        # Plot training curves
        self.plot_training_curves()
        
        return self.best_val_acc, test_metrics
    
    def evaluate(self, data_loader, phase='test'):
        """Evaluate model on given dataset"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(data_loader, desc=f'Evaluating {phase}'):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        metrics = self.calculate_metrics(all_labels, all_preds, all_probs)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(all_labels, all_preds, phase)
        
        return metrics
    
    def plot_confusion_matrix(self, true_labels, predictions, phase):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Benign', 'Malignant'],
                   yticklabels=['Benign', 'Malignant'])
        plt.title(f'Confusion Matrix - {phase.capitalize()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        os.makedirs('plots', exist_ok=True)
        plt.savefig(f'plots/confusion_matrix_{phase}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.close()
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Accuracy curves
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        os.makedirs('plots', exist_ok=True)
        plt.savefig(f'plots/training_curves_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.close()
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        os.makedirs(self.config['model_save_dir'], exist_ok=True)
        
        if is_best:
            torch.save(checkpoint, os.path.join(self.config['model_save_dir'], 'best_model.pth'))
        else:
            torch.save(checkpoint, os.path.join(self.config['model_save_dir'], f'checkpoint_epoch_{epoch+1}.pth'))

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train EfficientNet for skin cancer classification')
    
    # Core training arguments
    parser.add_argument('--epochs', type=int, default=5, 
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--learning-rate', '--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')
    
    # Model arguments
    parser.add_argument('--model-name', type=str, default='efficientnet_b0',
                       choices=['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 
                               'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
                               'efficientnet_b6', 'efficientnet_b7'],
                       help='EfficientNet model variant (default: efficientnet_b0)')
    parser.add_argument('--freeze-backbone', action='store_true',
                       help='Freeze backbone and only train classifier')
    
    # Training settings
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'sgd', 'adamw'],
                       help='Optimizer (default: adamw)')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'plateau', 'none'],
                       help='Learning rate scheduler (default: cosine)')
    
    # Data settings
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory (default: data)')
    parser.add_argument('--img-size', type=int, default=224,
                       help='Input image size (default: 224)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers (default: 4)')
    
    # Scheduler specific arguments
    parser.add_argument('--step-size', type=int, default=10,
                       help='Step size for StepLR scheduler (default: 10)')
    parser.add_argument('--gamma', type=float, default=0.1,
                       help='Gamma for StepLR scheduler (default: 0.1)')
    
    # Misc
    parser.add_argument('--save-interval', type=int, default=10,
                       help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--model-save-dir', type=str, default='models',
                       help='Directory to save model checkpoints (default: models)')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create configuration from arguments
    config = {
        # Model settings
        'model_name': args.model_name,
        'freeze_backbone': args.freeze_backbone,
        
        # Training settings
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'optimizer': args.optimizer,
        'scheduler': args.scheduler if args.scheduler != 'none' else None,
        'step_size': args.step_size,
        'gamma': args.gamma,
        
        # Data settings
        'data_dir': args.data_dir,
        'img_size': args.img_size,
        'num_workers': args.num_workers,
        
        # Misc
        'save_interval': args.save_interval,
        'model_save_dir': args.model_save_dir
    }
    
    logging.info("Training Configuration:")
    logging.info("-" * 40)
    for key, value in config.items():
        logging.info(f"{key}: {value}")
    logging.info("-" * 40)
    

    trainer = EfficientNetTrainer(config)
    

    best_val_acc, test_metrics = trainer.train()
    
    logging.info(f"\nTraining completed!")
    logging.info(f"Best validation accuracy: {best_val_acc:.4f}")
    logging.info(f"Test metrics: {test_metrics}")

if __name__ == "__main__":
    main()