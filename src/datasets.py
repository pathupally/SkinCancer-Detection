import os
from PIL import Image
from torch.utils.data import Dataset

class SkinCancerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.classes = ['Benign', 'Malignant']
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}

        self.image_paths = []
        self.labels = []

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(class_dir, fname))
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

import os
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import logging
from collections import Counter
import numpy as np

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class SkinCancerDatasets(Dataset):
    """
    Enhanced SkinCancerDataset compatible with existing project structure
    """
    
    def __init__(self, root_dir, transform=None, balance_classes=False):
        """
        Args:
            root_dir (str): Root directory with class subdirectories (Benign/Malignant)
            transform (callable, optional): Optional transform to be applied on images
            balance_classes (bool): Whether to balance classes using oversampling
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['Benign', 'Malignant']
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        
        self.image_paths = []
        self.labels = []
        
        # Load image paths and labels
        self._load_data()
        
        # Balance classes if requested
        if balance_classes:
            self._balance_classes()
        
        # Log dataset info
        self._log_dataset_info()
    
    def _load_data(self):
        """Load image paths and labels from directory structure"""
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_dir):
                logging.warning(f"Class directory not found: {class_dir}")
                continue
            
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for fname in image_files:
                self.image_paths.append(os.path.join(class_dir, fname))
                self.labels.append(self.class_to_idx[class_name])
    
    def _balance_classes(self):
        """Balance classes using random oversampling"""
        label_counts = Counter(self.labels)
        max_count = max(label_counts.values())
        
        logging.info(f"Original class distribution: {dict(label_counts)}")
        logging.info(f"Balancing to {max_count} samples per class")
        
        balanced_paths = []
        balanced_labels = []
        
        for class_idx in range(len(self.classes)):
            # Get indices for current class
            class_indices = [i for i, label in enumerate(self.labels) if label == class_idx]
            current_count = len(class_indices)
            
            # Add all original samples
            for idx in class_indices:
                balanced_paths.append(self.image_paths[idx])
                balanced_labels.append(self.labels[idx])
            
            # Oversample if needed
            if current_count < max_count:
                oversample_count = max_count - current_count
                oversample_indices = np.random.choice(class_indices, oversample_count, replace=True)
                
                for idx in oversample_indices:
                    balanced_paths.append(self.image_paths[idx])
                    balanced_labels.append(self.labels[idx])
        
        self.image_paths = balanced_paths
        self.labels = balanced_labels
        
        # Shuffle the balanced dataset
        combined = list(zip(self.image_paths, self.labels))
        np.random.shuffle(combined)
        self.image_paths, self.labels = zip(*combined)
        self.image_paths = list(self.image_paths)
        self.labels = list(self.labels)
    
    def _log_dataset_info(self):
        """Log dataset information"""
        label_counts = Counter(self.labels)
        total_samples = len(self.labels)
        
        logging.info(f"Dataset loaded from: {self.root_dir}")
        logging.info(f"Total samples: {total_samples}")
        
        for class_name, class_idx in self.class_to_idx.items():
            count = label_counts.get(class_idx, 0)
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            logging.info(f"{class_name}: {count} samples ({percentage:.1f}%)")
    
    def get_class_weights(self):
        """Calculate class weights for loss function balancing"""
        label_counts = Counter(self.labels)
        total_samples = len(self.labels)
        num_classes = len(self.classes)
        
        weights = []
        for class_idx in range(num_classes):
            class_count = label_counts.get(class_idx, 1)  # Avoid division by zero
            weight = total_samples / (num_classes * class_count)
            weights.append(weight)
        
        return torch.FloatTensor(weights)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {str(e)}")
            # Return a black image as fallback
            if self.transform:
                # Create a dummy image that matches expected input size
                dummy_image = Image.new('RGB', (224, 224), color=(0, 0, 0))
                image = self.transform(dummy_image)
            else:
                image = torch.zeros(3, 224, 224)  # Default tensor
            
            return image, label