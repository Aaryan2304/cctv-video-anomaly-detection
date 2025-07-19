"""
Video Dataset Management
========================

This module handles loading, preprocessing, and managing video datasets for
anomaly detection. It provides efficient data loading with preprocessing
pipelines optimized for surveillance video analysis.

Key Features:
- Efficient video frame extraction using OpenCV
- Preprocessing pipeline (grayscale, resize, normalize)
- Memory-efficient data loading with caching
- Support for both synthetic and real datasets
- UCSD Ped2 dataset integration
- Data augmentation for robust training
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Union
import warnings
from pathlib import Path
import pickle
from tqdm import tqdm

from .preprocessing import VideoPreprocessor


class VideoDataset(Dataset):
    """
    Custom Dataset class for loading and preprocessing video frames.
    
    This class handles the conversion of video files into preprocessed frames
    ready for neural network training. It supports both memory loading for
    small datasets and on-demand loading for large datasets.
    
    Features:
    - Efficient video processing with OpenCV
    - Configurable preprocessing pipeline
    - Memory vs disk trade-offs
    - Progress tracking for large datasets
    - Error handling for corrupted videos
    """
    
    def __init__(
        self,
        video_paths: List[str],
        frame_size: Tuple[int, int] = (64, 64),
        max_frames_per_video: Optional[int] = None,
        frame_skip: int = 1,
        load_in_memory: bool = True,
        cache_dir: Optional[str] = None,
        preprocessor: Optional[VideoPreprocessor] = None
    ):
        """
        Initialize the video dataset.
        
        Args:
            video_paths: List of paths to video files
            frame_size: Target size for frames (width, height)
            max_frames_per_video: Maximum frames to load per video
            frame_skip: Skip every N frames (1 = load all frames)
            load_in_memory: Whether to load all frames in memory
            cache_dir: Directory to cache preprocessed frames
            preprocessor: Custom preprocessing pipeline
        """
        self.video_paths = video_paths
        self.frame_size = frame_size
        self.max_frames_per_video = max_frames_per_video
        self.frame_skip = frame_skip
        self.load_in_memory = load_in_memory
        self.cache_dir = cache_dir
        
        # Initialize preprocessor
        self.preprocessor = preprocessor if preprocessor else VideoPreprocessor(frame_size)
        
        # Data storage
        self.frames = []
        self.frame_metadata = []  # Store video source, frame index, etc.
        
        # Performance tracking
        self.loading_stats = {
            'total_videos': len(video_paths),
            'total_frames': 0,
            'corrupted_videos': 0,
            'loading_time': 0
        }
        
        # Load data
        self._load_dataset()
    
    def _load_dataset(self):
        """Load and preprocess all video data."""
        print(f"Loading dataset with {len(self.video_paths)} videos...")
        
        if self.cache_dir and self._check_cache():
            print("✓ Loading from cache...")
            self._load_from_cache()
        else:
            print("✓ Processing videos...")
            self._process_videos()
            
            if self.cache_dir:
                self._save_to_cache()
        
        print(f"✓ Dataset loaded: {len(self.frames)} frames from {self.loading_stats['total_videos']} videos")
        if self.loading_stats['corrupted_videos'] > 0:
            print(f"⚠ Skipped {self.loading_stats['corrupted_videos']} corrupted videos")
    
    def _process_videos(self):
        """Process all videos and extract frames."""
        for video_idx, video_path in enumerate(tqdm(self.video_paths, desc="Processing videos")):
            try:
                frames = self._extract_frames_from_video(video_path, video_idx)
                
                if self.load_in_memory:
                    self.frames.extend(frames)
                else:
                    # Store paths instead of frames for on-demand loading
                    self.frame_metadata.extend([
                        {'video_path': video_path, 'frame_idx': i, 'video_idx': video_idx}
                        for i in range(len(frames))
                    ])
                
                self.loading_stats['total_frames'] += len(frames)
                
            except Exception as e:
                print(f"Warning: Failed to process {video_path}: {e}")
                self.loading_stats['corrupted_videos'] += 1
                continue
        
        # Convert frames to numpy array if loaded in memory
        if self.load_in_memory and self.frames:
            self.frames = np.array(self.frames, dtype=np.float32)
    
    def _extract_frames_from_video(self, video_path: str, video_idx: int) -> List[np.ndarray]:
        """
        Extract and preprocess frames from a single video or image file.
        
        Args:
            video_path: Path to video file or image file
            video_idx: Index of video in dataset
            
        Returns:
            List of preprocessed frames
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"File not found: {video_path}")
        
        # Check if this is an image file or video file
        image_extensions = {'.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        file_extension = os.path.splitext(video_path)[1].lower()
        
        if file_extension in image_extensions:
            # Handle individual image files
            return self._process_single_image(video_path)
        else:
            # Handle video files
            return self._process_video_file(video_path)
    
    def _process_single_image(self, image_path: str) -> List[np.ndarray]:
        """Process a single image file."""
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Preprocess frame
        processed_frame = self.preprocessor.process_frame(frame)
        return [processed_frame]
    
    def _process_video_file(self, video_path: str) -> List[np.ndarray]:
        """Process a video file."""
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frames = []
        frame_count = 0
        extracted_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if specified
                if frame_count % self.frame_skip != 0:
                    frame_count += 1
                    continue
                
                # Preprocess frame
                processed_frame = self.preprocessor.process_frame(frame)
                frames.append(processed_frame)
                
                extracted_count += 1
                frame_count += 1
                
                # Limit frames per video if specified
                if self.max_frames_per_video and extracted_count >= self.max_frames_per_video:
                    break
        
        finally:
            cap.release()
        
        return frames
    
    def _check_cache(self) -> bool:
        """Check if cached data exists and is valid."""
        if not self.cache_dir:
            return False
        
        cache_file = os.path.join(self.cache_dir, 'dataset_cache.pkl')
        metadata_file = os.path.join(self.cache_dir, 'dataset_metadata.pkl')
        
        return os.path.exists(cache_file) and os.path.exists(metadata_file)
    
    def _save_to_cache(self):
        """Save processed data to cache."""
        if not self.cache_dir:
            return
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        cache_file = os.path.join(self.cache_dir, 'dataset_cache.pkl')
        metadata_file = os.path.join(self.cache_dir, 'dataset_metadata.pkl')
        
        # Save frames
        with open(cache_file, 'wb') as f:
            pickle.dump(self.frames, f)
        
        # Save metadata
        cache_metadata = {
            'video_paths': self.video_paths,
            'frame_size': self.frame_size,
            'max_frames_per_video': self.max_frames_per_video,
            'frame_skip': self.frame_skip,
            'loading_stats': self.loading_stats,
            'frame_metadata': self.frame_metadata
        }
        
        with open(metadata_file, 'wb') as f:
            pickle.dump(cache_metadata, f)
        
        print(f"✓ Dataset cached to {self.cache_dir}")
    
    def _load_from_cache(self):
        """Load data from cache."""
        cache_file = os.path.join(self.cache_dir, 'dataset_cache.pkl')
        metadata_file = os.path.join(self.cache_dir, 'dataset_metadata.pkl')
        
        # Load frames
        with open(cache_file, 'rb') as f:
            self.frames = pickle.load(f)
        
        # Load metadata
        with open(metadata_file, 'rb') as f:
            cache_metadata = pickle.load(f)
            self.loading_stats = cache_metadata['loading_stats']
            self.frame_metadata = cache_metadata['frame_metadata']
    
    def __len__(self) -> int:
        """Return total number of frames in dataset."""
        if self.load_in_memory:
            return len(self.frames)
        else:
            return len(self.frame_metadata)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single frame by index.
        
        Args:
            idx: Frame index
            
        Returns:
            Tuple of (input_tensor, target_tensor) - same for autoencoder training
        """
        if self.load_in_memory:
            frame = self.frames[idx]
        else:
            # On-demand loading
            metadata = self.frame_metadata[idx]
            frame = self._load_frame_on_demand(metadata)
        
        # Add channel dimension and convert to tensor
        frame = np.expand_dims(frame, axis=0)  # (H, W) -> (1, H, W)
        frame_tensor = torch.from_numpy(frame).float()
        
        # For autoencoder, input and target are the same
        return frame_tensor, frame_tensor
    
    def _load_frame_on_demand(self, metadata: dict) -> np.ndarray:
        """Load and preprocess a single frame on demand."""
        video_path = metadata['video_path']
        frame_idx = metadata['frame_idx']
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx * self.frame_skip)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Could not read frame {frame_idx} from {video_path}")
        
        return self.preprocessor.process_frame(frame)
    
    def get_dataset_info(self) -> dict:
        """Get comprehensive dataset information."""
        return {
            'total_frames': len(self),
            'frame_size': self.frame_size,
            'video_count': self.loading_stats['total_videos'],
            'frames_per_video_avg': len(self) / max(1, self.loading_stats['total_videos']),
            'corrupted_videos': self.loading_stats['corrupted_videos'],
            'frame_skip': self.frame_skip,
            'max_frames_per_video': self.max_frames_per_video,
            'memory_usage_mb': self._estimate_memory_usage(),
            'load_in_memory': self.load_in_memory
        }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        if self.load_in_memory and len(self.frames) > 0:
            if isinstance(self.frames, np.ndarray):
                return self.frames.nbytes / (1024 * 1024)
            else:
                # Estimate based on frame size and count
                frame_size_bytes = self.frame_size[0] * self.frame_size[1] * 4  # float32
                return (len(self.frames) * frame_size_bytes) / (1024 * 1024)
        return 0.0


class UCSDPed2Dataset(VideoDataset):
    """
    Specialized dataset class for UCSD Ped2 dataset.
    
    This class handles the specific structure and format of the UCSD Ped2
    dataset, including ground truth labels for evaluation.
    """
    
    def __init__(
        self,
        dataset_root: str,
        subset: str = 'Train',
        **kwargs
    ):
        """
        Initialize UCSD Ped2 dataset.
        
        Args:
            dataset_root: Root directory of UCSD Ped2 dataset
            subset: 'Train' or 'Test'
            **kwargs: Additional arguments for VideoDataset
        """
        self.dataset_root = dataset_root
        self.subset = subset
        
        # Get video paths for the subset
        subset_dir = os.path.join(dataset_root, subset)
        if not os.path.exists(subset_dir):
            raise FileNotFoundError(f"UCSD Ped2 {subset} directory not found: {subset_dir}")
        
        video_paths = self._get_ucsd_video_paths(subset_dir)
        
        # Initialize parent class
        super().__init__(video_paths, **kwargs)
        
        # Load ground truth if test set
        self.ground_truth = None
        if subset == 'Test':
            self._load_ground_truth()
    
    def _get_ucsd_video_paths(self, subset_dir: str) -> List[str]:
        """Get video file paths for UCSD dataset."""
        video_paths = []
        
        # UCSD dataset has videos in numbered directories
        for item in os.listdir(subset_dir):
            item_path = os.path.join(subset_dir, item)
            
            if os.path.isdir(item_path):
                # Look for video files in the directory
                for file in os.listdir(item_path):
                    if file.endswith(('.avi', '.mp4', '.mov')):
                        video_paths.append(os.path.join(item_path, file))
            elif item.endswith(('.avi', '.mp4', '.mov')):
                video_paths.append(item_path)
        
        video_paths.sort()  # Ensure consistent ordering
        return video_paths
    
    def _load_ground_truth(self):
        """Load ground truth labels for test set."""
        gt_dir = os.path.join(self.dataset_root, 'Test_GT')
        if not os.path.exists(gt_dir):
            print(f"Warning: Ground truth directory not found: {gt_dir}")
            return
        
        # Implementation depends on UCSD ground truth format
        # This would need to be adapted based on actual GT format
        print("Ground truth loading not implemented yet")
    
    def get_ground_truth_labels(self) -> Optional[np.ndarray]:
        """Get ground truth labels if available."""
        return self.ground_truth


class SyntheticVideoDataset(Dataset):
    """
    Dataset for synthetically generated video data.
    
    This class generates artificial video frames for testing and demonstration
    purposes, creating predictable normal patterns and clear anomalies.
    """
    
    def __init__(
        self,
        num_normal_frames: int = 800,
        num_anomaly_frames: int = 80,
        frame_size: Tuple[int, int] = (64, 64),
        pattern_type: str = 'moving_circle',
        noise_level: float = 0.03
    ):
        """
        Initialize synthetic dataset.
        
        Args:
            num_normal_frames: Number of normal frames to generate
            num_anomaly_frames: Number of anomalous frames to generate
            frame_size: Size of generated frames
            pattern_type: Type of normal pattern ('moving_circle', 'bouncing_ball')
            noise_level: Amount of random noise to add
        """
        self.num_normal_frames = num_normal_frames
        self.num_anomaly_frames = num_anomaly_frames
        self.frame_size = frame_size
        self.pattern_type = pattern_type
        self.noise_level = noise_level
        
        # Generate data
        self.frames = []
        self.labels = []  # 0 = normal, 1 = anomaly
        
        self._generate_data()
    
    def _generate_data(self):
        """Generate synthetic video frames."""
        print(f"Generating {self.num_normal_frames} normal and {self.num_anomaly_frames} anomaly frames...")
        
        # Generate normal frames
        for i in range(self.num_normal_frames):
            frame = self._generate_normal_frame(i)
            self.frames.append(frame)
            self.labels.append(0)
        
        # Generate anomalous frames
        for i in range(self.num_anomaly_frames):
            frame = self._generate_anomaly_frame(i)
            self.frames.append(frame)
            self.labels.append(1)
        
        # Convert to numpy arrays
        self.frames = np.array(self.frames, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        
        print(f"✓ Generated {len(self.frames)} synthetic frames")
    
    def _generate_normal_frame(self, frame_idx: int) -> np.ndarray:
        """Generate a normal frame with predictable patterns."""
        frame = np.zeros(self.frame_size, dtype=np.float32)
        
        if self.pattern_type == 'moving_circle':
            # Moving circle with sinusoidal motion
            center_x = int(32 + 20 * np.sin(frame_idx * 0.1))
            center_y = int(32 + 15 * np.cos(frame_idx * 0.15))
            radius = 8 + int(3 * np.sin(frame_idx * 0.05))
            
            cv2.circle(frame, (center_x, center_y), radius, 1.0, -1)
        
        elif self.pattern_type == 'bouncing_ball':
            # Bouncing ball pattern
            x = int(10 + (frame_idx * 2) % 44)
            y = int(32 + 20 * np.sin(frame_idx * 0.2))
            
            cv2.circle(frame, (x, y), 6, 1.0, -1)
        
        # Add noise
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level, self.frame_size)
            frame = np.clip(frame + noise, 0, 1)
        
        return frame
    
    def _generate_anomaly_frame(self, frame_idx: int) -> np.ndarray:
        """Generate an anomalous frame with unusual patterns."""
        frame = np.zeros(self.frame_size, dtype=np.float32)
        
        anomaly_type = np.random.choice(['rectangle', 'multiple_circles', 'line'])
        
        if anomaly_type == 'rectangle':
            # Rectangle (never seen in normal data)
            x1 = np.random.randint(5, 35)
            y1 = np.random.randint(5, 35)
            x2 = np.random.randint(x1 + 10, 60)
            y2 = np.random.randint(y1 + 10, 60)
            cv2.rectangle(frame, (x1, y1), (x2, y2), 1.0, -1)
        
        elif anomaly_type == 'multiple_circles':
            # Multiple small circles
            for _ in range(np.random.randint(3, 7)):
                center_x = np.random.randint(10, 54)
                center_y = np.random.randint(10, 54)
                radius = np.random.randint(2, 6)
                cv2.circle(frame, (center_x, center_y), radius, 1.0, -1)
        
        elif anomaly_type == 'line':
            # Diagonal line
            pt1 = (np.random.randint(0, 32), np.random.randint(0, 32))
            pt2 = (np.random.randint(32, 64), np.random.randint(32, 64))
            cv2.line(frame, pt1, pt2, 1.0, 3)
        
        # Add noise
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level, self.frame_size)
            frame = np.clip(frame + noise, 0, 1)
        
        return frame
    
    def __len__(self) -> int:
        """Return total number of frames."""
        return len(self.frames)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get frame and label by index."""
        frame = self.frames[idx]
        
        # Add channel dimension
        frame = np.expand_dims(frame, axis=0)
        frame_tensor = torch.from_numpy(frame).float()
        
        # For autoencoder training, input and target are the same
        return frame_tensor, frame_tensor
    
    def get_labels(self) -> np.ndarray:
        """Get all labels for evaluation."""
        return self.labels
    
    def split_normal_anomaly(self) -> Tuple['SyntheticVideoDataset', 'SyntheticVideoDataset']:
        """Split dataset into normal and anomaly subsets."""
        normal_indices = np.where(self.labels == 0)[0]
        anomaly_indices = np.where(self.labels == 1)[0]
        
        # Create normal subset
        normal_dataset = SyntheticVideoDataset(0, 0, self.frame_size)
        normal_dataset.frames = self.frames[normal_indices]
        normal_dataset.labels = self.labels[normal_indices]
        
        # Create anomaly subset
        anomaly_dataset = SyntheticVideoDataset(0, 0, self.frame_size)
        anomaly_dataset.frames = self.frames[anomaly_indices]
        anomaly_dataset.labels = self.labels[anomaly_indices]
        
        return normal_dataset, anomaly_dataset


def create_data_loaders(
    dataset: Dataset,
    batch_size: int = 32,
    train_split: float = 0.8,
    val_split: float = 0.1,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        dataset: Dataset to split
        batch_size: Batch size for data loaders
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    dataset_size = len(dataset)
    indices = np.random.permutation(dataset_size)
    
    # Calculate split indices
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset classes
    print("Testing Video Dataset Classes...")
    
    # Test synthetic dataset
    print("\n1. Testing Synthetic Dataset:")
    synthetic_dataset = SyntheticVideoDataset(
        num_normal_frames=100,
        num_anomaly_frames=20,
        frame_size=(64, 64)
    )
    
    print(f"✓ Synthetic dataset: {len(synthetic_dataset)} frames")
    
    # Test data loader
    loader = DataLoader(synthetic_dataset, batch_size=8, shuffle=True)
    for batch_data, batch_target in loader:
        print(f"✓ Batch shape: {batch_data.shape}")
        break
    
    # Test dataset info
    sample_frame, _ = synthetic_dataset[0]
    print(f"✓ Sample frame shape: {sample_frame.shape}")
    print(f"✓ Labels distribution: Normal={np.sum(synthetic_dataset.labels == 0)}, Anomaly={np.sum(synthetic_dataset.labels == 1)}")
    
    print("\n✓ All dataset tests completed successfully!")