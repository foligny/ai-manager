"""
Data generator for creating realistic training data for a small image classification model.
Creates synthetic data that mimics real-world patterns.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os
import json


class SyntheticImageDataset(Dataset):
    """Synthetic image dataset for testing AI Manager."""
    
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def generate_synthetic_data(n_samples=100, image_size=32, n_classes=3, noise_level=0.1):
    """
    Generate synthetic image data that mimics real-world patterns.
    
    Args:
        n_samples: Number of samples to generate
        image_size: Size of square images (32x32)
        n_classes: Number of classes (3: circles, squares, triangles)
        noise_level: Amount of noise to add
    
    Returns:
        X: Image data (n_samples, 1, image_size, image_size)
        y: Labels (n_samples,)
    """
    np.random.seed(42)  # For reproducibility
    
    # Generate base patterns for each class
    patterns = {
        0: generate_circle_pattern(image_size),      # Circles
        1: generate_square_pattern(image_size),      # Squares  
        2: generate_triangle_pattern(image_size)     # Triangles
    }
    
    X = []
    y = []
    
    samples_per_class = n_samples // n_classes
    
    for class_id in range(n_classes):
        base_pattern = patterns[class_id]
        
        for _ in range(samples_per_class):
            # Add variations to the base pattern
            pattern = base_pattern.copy()
            
            # Random rotation
            angle = np.random.uniform(0, 360)
            pattern = rotate_pattern(pattern, angle)
            
            # Random scaling
            scale = np.random.uniform(0.7, 1.3)
            pattern = scale_pattern(pattern, scale)
            
            # Random translation
            dx = np.random.uniform(-2, 2)
            dy = np.random.uniform(-2, 2)
            pattern = translate_pattern(pattern, dx, dy)
            
            # Add noise
            noise = np.random.normal(0, noise_level, pattern.shape)
            pattern = np.clip(pattern + noise, 0, 1)
            
            X.append(pattern)
            y.append(class_id)
    
    # Convert to numpy arrays
    X = np.array(X, dtype=np.float32).reshape(-1, 1, image_size, image_size)
    y = np.array(y, dtype=np.int64)
    
    return X, y


def generate_circle_pattern(size):
    """Generate a circle pattern."""
    pattern = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    radius = size // 4
    
    for i in range(size):
        for j in range(size):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            if dist <= radius:
                pattern[i, j] = 1.0
    
    return pattern


def generate_square_pattern(size):
    """Generate a square pattern."""
    pattern = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    half_size = size // 6
    
    for i in range(size):
        for j in range(size):
            if (abs(i - center) <= half_size and 
                abs(j - center) <= half_size):
                pattern[i, j] = 1.0
    
    return pattern


def generate_triangle_pattern(size):
    """Generate a triangle pattern."""
    pattern = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    height = size // 3
    
    for i in range(size):
        for j in range(size):
            # Create triangle shape
            if i >= center - height//2 and i <= center + height//2:
                width = int((i - (center - height//2)) * 0.8)
                if abs(j - center) <= width:
                    pattern[i, j] = 1.0
    
    return pattern


def rotate_pattern(pattern, angle):
    """Rotate pattern by given angle (simplified)."""
    # Simplified rotation - just return the pattern for now
    return pattern


def scale_pattern(pattern, scale):
    """Scale pattern by given factor (simplified)."""
    # Simplified scaling - just return the pattern for now
    return pattern


def translate_pattern(pattern, dx, dy):
    """Translate pattern by given offsets (simplified)."""
    # Simplified translation - just return the pattern for now
    return pattern


def create_data_loaders(X, y, batch_size=8, test_size=0.2):
    """Create train and test data loaders."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    
    # Create datasets
    train_dataset = SyntheticImageDataset(X_train, y_train)
    test_dataset = SyntheticImageDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def visualize_data(X, y, n_samples=9):
    """Visualize sample data."""
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.ravel()
    
    class_names = ['Circle', 'Square', 'Triangle']
    
    for i in range(n_samples):
        if i < len(X):
            img = X[i].squeeze()
            label = y[i]
            
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'{class_names[label]}')
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_data.png', dpi=150, bbox_inches='tight')
    plt.show()


def save_data_info(X, y, filename='data_info.json'):
    """Save data information."""
    info = {
        'n_samples': len(X),
        'n_classes': len(np.unique(y)),
        'image_size': X.shape[-1],
        'class_distribution': np.bincount(y).tolist(),
        'class_names': ['Circle', 'Square', 'Triangle']
    }
    
    with open(filename, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"Data info saved to {filename}")
    print(f"Total samples: {info['n_samples']}")
    print(f"Classes: {info['class_names']}")
    print(f"Class distribution: {info['class_distribution']}")


if __name__ == "__main__":
    # Generate data
    print("Generating synthetic image data...")
    X, y = generate_synthetic_data(n_samples=100, image_size=32, n_classes=3)
    
    # Save data info
    save_data_info(X, y)
    
    # Visualize sample data
    visualize_data(X, y)
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(X, y, batch_size=8)
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Save data for later use
    np.save('X.npy', X)
    np.save('y.npy', y)
    
    print("Data generation complete!") 