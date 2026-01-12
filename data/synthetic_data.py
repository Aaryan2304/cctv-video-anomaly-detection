"""
Synthetic Data Generation for Anomaly Detection
===============================================

This module generates artificial video data for demonstration and learning purposes.
It creates clear examples of normal vs anomalous patterns to help understand
how anomaly detection works.
"""

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from typing import Tuple


class SyntheticVideoDataset(Dataset):
    """
    Dataset that generates synthetic video frames for anomaly detection training.
    
    Normal patterns: Moving circles with predictable motion
    Anomaly patterns: Rectangles, multiple shapes, or unusual movements
    
    This provides clear, controlled examples for learning and testing.
    """
    
    def __init__(
        self,
        num_normal_frames: int = 800,
        num_anomaly_frames: int = 80,
        frame_size: Tuple[int, int] = (64, 64),
        noise_level: float = 0.03
    ):
        """
        Initialize synthetic dataset.
        
        Args:
            num_normal_frames: Number of normal frames to generate
            num_anomaly_frames: Number of anomalous frames to generate
            frame_size: Size of generated frames (width, height)
            noise_level: Amount of random noise to add (0.0-0.1)
        """
        self.num_normal_frames = num_normal_frames
        self.num_anomaly_frames = num_anomaly_frames
        self.frame_size = frame_size
        self.noise_level = noise_level
        
        # Generate all frames and labels
        self.frames = []
        self.labels = []  # 0 = normal, 1 = anomaly
        
        print(f"Generating {num_normal_frames} normal and {num_anomaly_frames} anomaly frames...")
        self._generate_data()
        print(f"Generated {len(self.frames)} synthetic frames")
    
    def _generate_data(self):
        """Generate all synthetic frames and labels."""
        
        # Generate normal frames (moving circles)
        for i in range(self.num_normal_frames):
            frame = self._generate_normal_frame(i)
            self.frames.append(frame)
            self.labels.append(0)  # Normal label
        
        # Generate anomalous frames (rectangles, multiple shapes, etc.)
        for i in range(self.num_anomaly_frames):
            frame = self._generate_anomaly_frame(i)
            self.frames.append(frame)
            self.labels.append(1)  # Anomaly label
        
        # Convert to numpy arrays
        self.frames = np.array(self.frames, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
    
    def _generate_normal_frame(self, frame_idx: int) -> np.ndarray:
        """
        Generate a normal frame with predictable circular motion.
        
        Args:
            frame_idx: Frame index for temporal consistency
            
        Returns:
            Generated normal frame as numpy array
        """
        frame = np.zeros(self.frame_size, dtype=np.float32)
        
        # Create moving circle with sinusoidal motion
        center_x = int(32 + 20 * np.sin(frame_idx * 0.1))
        center_y = int(32 + 15 * np.cos(frame_idx * 0.15))
        
        # Vary radius slightly over time
        radius = 8 + int(3 * np.sin(frame_idx * 0.05))
        
        # Ensure circle stays within bounds
        center_x = np.clip(center_x, radius, self.frame_size[0] - radius)
        center_y = np.clip(center_y, radius, self.frame_size[1] - radius)
        
        # Draw circle
        cv2.circle(frame, (center_x, center_y), radius, 1.0, -1)
        
        # Add random noise
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level, self.frame_size)
            frame = np.clip(frame + noise, 0, 1)
        
        return frame
    
    def _generate_anomaly_frame(self, frame_idx: int) -> np.ndarray:
        """
        Generate an anomalous frame with unusual patterns.
        
        Args:
            frame_idx: Frame index for variation
            
        Returns:
            Generated anomaly frame as numpy array
        """
        frame = np.zeros(self.frame_size, dtype=np.float32)
        
        # Randomly choose anomaly type
        anomaly_types = ['rectangle', 'multiple_circles', 'line', 'triangle']
        anomaly_type = np.random.choice(anomaly_types)
        
        if anomaly_type == 'rectangle':
            # Rectangle (never seen in normal training)
            x1 = np.random.randint(5, 30)
            y1 = np.random.randint(5, 30)
            x2 = np.random.randint(x1 + 15, min(x1 + 35, self.frame_size[0] - 5))
            y2 = np.random.randint(y1 + 15, min(y1 + 35, self.frame_size[1] - 5))
            cv2.rectangle(frame, (x1, y1), (x2, y2), 1.0, -1)
        
        elif anomaly_type == 'multiple_circles':
            # Multiple small circles (different pattern from normal single circle)
            num_circles = np.random.randint(3, 7)
            for _ in range(num_circles):
                center_x = np.random.randint(8, self.frame_size[0] - 8)
                center_y = np.random.randint(8, self.frame_size[1] - 8)
                radius = np.random.randint(2, 6)
                cv2.circle(frame, (center_x, center_y), radius, 1.0, -1)
        
        elif anomaly_type == 'line':
            # Diagonal or straight line
            pt1 = (np.random.randint(0, self.frame_size[0]//2), 
                   np.random.randint(0, self.frame_size[1]//2))
            pt2 = (np.random.randint(self.frame_size[0]//2, self.frame_size[0]), 
                   np.random.randint(self.frame_size[1]//2, self.frame_size[1]))
            thickness = np.random.randint(2, 5)
            cv2.line(frame, pt1, pt2, 1.0, thickness)
        
        elif anomaly_type == 'triangle':
            # Triangle shape
            pts = np.array([
                [np.random.randint(10, 54), np.random.randint(10, 25)],
                [np.random.randint(10, 54), np.random.randint(35, 54)],
                [np.random.randint(10, 54), np.random.randint(25, 45)]
            ], np.int32)
            cv2.fillPoly(frame, [pts], 1.0)
        
        # Add random noise
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level, self.frame_size)
            frame = np.clip(frame + noise, 0, 1)
        
        return frame
    
    def __len__(self) -> int:
        """Return total number of frames."""
        return len(self.frames)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get frame by index.
        
        Args:
            idx: Frame index
            
        Returns:
            Tuple of (frame_tensor, frame_tensor) for autoencoder training
        """
        frame = self.frames[idx]
        
        # Add channel dimension: (H, W) -> (1, H, W)
        frame = np.expand_dims(frame, axis=0)
        frame_tensor = torch.from_numpy(frame).float()
        
        # For autoencoder training, input and target are the same
        return frame_tensor, frame_tensor
    
    def get_labels(self) -> np.ndarray:
        """
        Get all labels for evaluation.
        
        Returns:
            Array of labels (0=normal, 1=anomaly)
        """
        return self.labels.copy()
    
    def get_normal_subset(self):
        """Get subset containing only normal frames."""
        normal_indices = np.where(self.labels == 0)[0]
        return torch.utils.data.Subset(self, normal_indices)
    
    def get_anomaly_subset(self):
        """Get subset containing only anomalous frames."""
        anomaly_indices = np.where(self.labels == 1)[0]
        return torch.utils.data.Subset(self, anomaly_indices)
    
    def visualize_samples(self, num_samples: int = 8, save_path: str = None):
        """
        Visualize sample frames for inspection.
        
        Args:
            num_samples: Number of samples to show
            save_path: Path to save visualization (optional)
        """
        import matplotlib.pyplot as plt
        
        # Get balanced samples
        normal_indices = np.where(self.labels == 0)[0]
        anomaly_indices = np.where(self.labels == 1)[0]
        
        num_normal = min(num_samples // 2, len(normal_indices))
        num_anomaly = min(num_samples - num_normal, len(anomaly_indices))
        
        selected_indices = []
        selected_indices.extend(np.random.choice(normal_indices, num_normal, replace=False))
        selected_indices.extend(np.random.choice(anomaly_indices, num_anomaly, replace=False))
        
        # Create visualization
        fig, axes = plt.subplots(2, len(selected_indices), figsize=(2*len(selected_indices), 4))
        if len(selected_indices) == 1:
            axes = axes.reshape(-1, 1)
        
        for i, idx in enumerate(selected_indices):
            frame = self.frames[idx]
            label = self.labels[idx]
            label_text = "Normal" if label == 0 else "Anomaly"
            
            # Original frame
            axes[0, i].imshow(frame, cmap='gray', vmin=0, vmax=1)
            axes[0, i].set_title(f'{label_text}\nFrame {idx}')
            axes[0, i].axis('off')
            
            # Create a simple visualization showing the pattern type
            axes[1, i].text(0.5, 0.5, label_text, 
                           horizontalalignment='center', 
                           verticalalignment='center',
                           transform=axes[1, i].transAxes,
                           fontsize=12, 
                           bbox=dict(boxstyle="round,pad=0.3", 
                                   facecolor='lightgreen' if label == 0 else 'lightcoral'))
            axes[1, i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Sample visualization saved to {save_path}")
        
        plt.show()
    
    def get_dataset_info(self) -> dict:
        """Get comprehensive dataset information."""
        return {
            'total_frames': len(self.frames),
            'normal_frames': np.sum(self.labels == 0),
            'anomaly_frames': np.sum(self.labels == 1),
            'anomaly_rate': np.mean(self.labels),
            'frame_size': self.frame_size,
            'noise_level': self.noise_level,
            'data_type': 'synthetic'
        }


def create_synthetic_datasets(
    total_normal: int = 1000,
    total_anomaly: int = 100,
    train_split: float = 0.8,
    **kwargs
) -> Tuple[Dataset, Dataset, np.ndarray]:
    """
    Create train and test datasets from synthetic data.
    
    Args:
        total_normal: Total normal frames to generate
        total_anomaly: Total anomaly frames to generate  
        train_split: Fraction of normal data for training
        **kwargs: Additional arguments for SyntheticVideoDataset
        
    Returns:
        Tuple of (train_dataset, test_dataset, test_labels)
    """
    # Create full synthetic dataset
    full_dataset = SyntheticVideoDataset(
        num_normal_frames=total_normal,
        num_anomaly_frames=total_anomaly,
        **kwargs
    )
    
    # Split normal data for training
    labels = full_dataset.get_labels()
    normal_indices = np.where(labels == 0)[0]
    
    # Use portion of normal data for training
    split_idx = int(train_split * len(normal_indices))
    train_indices = normal_indices[:split_idx]
    
    # Create train dataset (normal only)
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    
    # Test dataset contains remaining normal + all anomalies
    test_indices = np.concatenate([normal_indices[split_idx:], np.where(labels == 1)[0]])
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    test_labels = labels[test_indices]
    
    print(f"Created datasets: {len(train_dataset)} train (normal only), {len(test_dataset)} test (mixed)")
    
    return train_dataset, test_dataset, test_labels


if __name__ == "__main__":
    # Test synthetic data generation
    print("Testing synthetic data generation...")
    
    # Create dataset
    dataset = SyntheticVideoDataset(
        num_normal_frames=50,
        num_anomaly_frames=10,
        frame_size=(64, 64)
    )
    
    # Print info
    info = dataset.get_dataset_info()
    print(f"Dataset info: {info}")
    
    # Test data access
    frame, target = dataset[0]
    print(f"Frame shape: {frame.shape}")
    print(f"Frame range: [{frame.min():.3f}, {frame.max():.3f}]")
    
    # Test labels
    labels = dataset.get_labels()
    print(f"Label distribution: Normal={np.sum(labels==0)}, Anomaly={np.sum(labels==1)}")
    
    # Visualize samples
    dataset.visualize_samples(num_samples=6)
    
    print("Synthetic data test completed!")
