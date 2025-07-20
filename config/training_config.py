# config/training_config.py

class TrainingConfig:
    """Training configuration for skin cancer detection"""
    
    # EfficientNet B0 - Fast training, good for experimentation
    EFFICIENTNET_B0 = {
        'model_name': 'efficientnet_b0',
        'img_size': 224,
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'freeze_backbone': False,
        'data_dir': 'data',
        'num_workers': 4,
        'save_interval': 10
    }
    
    # EfficientNet B3 - Balanced performance
    EFFICIENTNET_B3 = {
        'model_name': 'efficientnet_b3',
        'img_size': 300,
        'batch_size': 16,
        'epochs': 75,
        'learning_rate': 5e-5,
        'weight_decay': 1e-4,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'freeze_backbone': False,
        'data_dir': 'data',
        'num_workers': 4,
        'save_interval': 15
    }
    
    # EfficientNet B5 - High accuracy
    EFFICIENTNET_B5 = {
        'model_name': 'efficientnet_b5',
        'img_size': 456,
        'batch_size': 8,
        'epochs': 100,
        'learning_rate': 3e-5,
        'weight_decay': 1e-4,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'freeze_backbone': False,
        'data_dir': 'data',
        'num_workers': 4,
        'save_interval': 20
    }
    
    # Fine-tuning configuration (freeze backbone initially)
    FINETUNE_CONFIG = {
        'model_name': 'efficientnet_b0',
        'img_size': 224,
        'batch_size': 32,
        'epochs': 30,
        'learning_rate': 1e-5,
        'weight_decay': 1e-5,
        'optimizer': 'adam',
        'scheduler': 'plateau',
        'freeze_backbone': True,
        'data_dir': 'data',
        'num_workers': 4,
        'save_interval': 5
    }

# requirements.txt content
REQUIREMENTS = """
torch>=1.9.0
torchvision>=0.10.0
timm>=0.6.0
Pillow>=8.3.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
tqdm>=4.62.0
"""

# Updated main training script
def main():
    """Main training function"""
    import sys
    import os
    
    # Add src to Python path
    sys.path.append('src')
    
    from utils import setup_logger
    from train import EfficientNetTrainer
    import logging
    
    # Setup logging
    logger = setup_logger()
    
    # Choose configuration
    config_options = {
        'b0': TrainingConfig.EFFICIENTNET_B0,
        'b3': TrainingConfig.EFFICIENTNET_B3,
        'b5': TrainingConfig.EFFICIENTNET_B5,
        'finetune': TrainingConfig.FINETUNE_CONFIG
    }
    
    # Default to B0 for initial experiments
    selected_config = 'b0'
    
    if len(sys.argv) > 1:
        selected_config = sys.argv[1]
        if selected_config not in config_options:
            logging.error(f"Invalid config. Choose from: {list(config_options.keys())}")
            sys.exit(1)
    
    config = config_options[selected_config]
    logging.info(f"Using configuration: {selected_config}")
    logging.info(f"Model: {config['model_name']}")
    logging.info(f"Image size: {config['img_size']}")
    logging.info(f"Batch size: {config['batch_size']}")
    
    # Check data directory
    data_dir = config['data_dir']
    processed_dir = os.path.join(data_dir, 'processed')
    
    required_dirs = ['train', 'val', 'test']
    for split in required_dirs:
        split_dir = os.path.join(processed_dir, split)
        if not os.path.exists(split_dir):
            logging.error(f"Missing data directory: {split_dir}")
            logging.error("Please run data organization script first!")
            sys.exit(1)
        
        # Check for class subdirectories
        for class_name in ['Benign', 'Malignant']:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                logging.error(f"Missing class directory: {class_dir}")
                sys.exit(1)
    
    logging.info("Data directory structure validated!")
    
    try:
        # Initialize trainer
        trainer = EfficientNetTrainer(config)
        
        # Start training
        best_val_acc, test_metrics = trainer.train()
        
        logging.info("=" * 60)
        logging.info("TRAINING COMPLETED SUCCESSFULLY!")
        logging.info("=" * 60)
        logging.info(f"Best validation accuracy: {best_val_acc:.4f}")
        logging.info("Final test metrics:")
        for metric, value in test_metrics.items():
            logging.info(f"  {metric.capitalize()}: {value:.4f}")
        
        # Save final results
        results = {
            'config': config,
            'best_val_acc': float(best_val_acc),
            'test_metrics': {k: float(v) for k, v in test_metrics.items()},
            'model_name': config['model_name']
        }
        
        import json
        results_file = f"results_{config['model_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"Results saved to: {results_file}")
        
    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    from datetime import datetime
    main()