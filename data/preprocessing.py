"""
Video frame extraction and preprocessing pipeline.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List
import albumentations as A
from albumentations import Compose
import warnings


class VideoPreprocessor:
    """Frame extraction, resizing, and normalization for autoencoder input."""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (64, 64),
        convert_to_grayscale: bool = True,
        normalize_method: str = 'minmax',
        augmentation_config: Optional[Dict] = None,
        quality_threshold: float = 0.1
    ):
        """
        Initialize the video preprocessor.
        
        Args:
            target_size: Target frame size (width, height)
            convert_to_grayscale: Whether to convert frames to grayscale
            normalize_method: Normalization method ('minmax', 'zscore', 'none')
            augmentation_config: Configuration for data augmentation
            quality_threshold: Minimum quality threshold for frame acceptance
        """
        self.target_size = target_size
        self.convert_to_grayscale = convert_to_grayscale
        self.normalize_method = normalize_method
        self.quality_threshold = quality_threshold
        
        # Statistics for normalization
        self.normalization_stats = {
            'mean': 0.0,
            'std': 1.0,
            'min': 0.0,
            'max': 1.0
        }
        
        # Setup augmentation pipeline
        self.augmentation_pipeline = self._create_augmentation_pipeline(augmentation_config)
        
        # Processing statistics
        self.stats = {
            'frames_processed': 0,
            'frames_rejected': 0,
            'avg_processing_time': 0.0
        }
    
    def _create_augmentation_pipeline(self, config: Optional[Dict]) -> Optional[Compose]:
        """
        Create data augmentation pipeline using Albumentations.
        
        Args:
            config: Augmentation configuration dictionary
            
        Returns:
            Albumentations composition or None if no augmentation
        """
        if not config or not config.get('enabled', False):
            return None
        
        transforms = []
        
        # Geometric transforms (preserve anomaly structure)
        if config.get('rotation', False):
            transforms.append(A.Rotate(
                limit=config.get('rotation_limit', 5),
                p=config.get('rotation_prob', 0.3)
            ))
        
        if config.get('shift_scale', False):
            transforms.append(A.ShiftScaleRotate(
                shift_limit=config.get('shift_limit', 0.05),
                scale_limit=config.get('scale_limit', 0.05),
                rotate_limit=0,  # Handled separately
                p=config.get('shift_scale_prob', 0.3)
            ))
        
        # Intensity transforms (simulate lighting changes)
        if config.get('brightness_contrast', False):
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=config.get('brightness_limit', 0.1),
                contrast_limit=config.get('contrast_limit', 0.1),
                p=config.get('brightness_contrast_prob', 0.3)
            ))
        
        # Noise augmentation (simulate sensor noise)
        if config.get('gaussian_noise', False):
            transforms.append(A.GaussNoise(
                var_limit=config.get('noise_variance', (10, 50)),
                p=config.get('noise_prob', 0.2)
            ))
        
        if transforms:
            return Compose(transforms)
        return None
    
    def process_frame(
        self, 
        frame: np.ndarray, 
        apply_augmentation: bool = False,
        validate_quality: bool = True
    ) -> np.ndarray:
        """
        Process a single video frame through the preprocessing pipeline.
        
        Args:
            frame: Input frame as numpy array (BGR format from OpenCV)
            apply_augmentation: Whether to apply data augmentation
            validate_quality: Whether to validate frame quality
            
        Returns:
            Preprocessed frame ready for neural network input
            
        Raises:
            ValueError: If frame quality is below threshold
        """
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame: empty or None")
        
        # Quality validation
        if validate_quality and not self._validate_frame_quality(frame):
            self.stats['frames_rejected'] += 1
            raise ValueError("Frame quality below threshold")
        
        # Convert color space
        if self.convert_to_grayscale:
            if len(frame.shape) == 3:
                processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                processed_frame = frame
        else:
            processed_frame = frame.copy()
        
        # Resize frame
        processed_frame = self._resize_frame(processed_frame)
        
        # Apply augmentation if requested
        if apply_augmentation and self.augmentation_pipeline:
            if len(processed_frame.shape) == 2:
                # Add channel dimension for albumentations
                processed_frame = np.expand_dims(processed_frame, axis=2)
                augmented = self.augmentation_pipeline(image=processed_frame)
                processed_frame = augmented['image'].squeeze()
            else:
                augmented = self.augmentation_pipeline(image=processed_frame)
                processed_frame = augmented['image']
        
        # Normalize
        processed_frame = self._normalize_frame(processed_frame)
        
        self.stats['frames_processed'] += 1
        return processed_frame
    
    def _validate_frame_quality(self, frame: np.ndarray) -> bool:
        """
        Validate frame quality using various metrics.
        
        Args:
            frame: Input frame
            
        Returns:
            True if frame quality is acceptable, False otherwise
        """
        # Check for blank/corrupted frames
        if frame is None or frame.size == 0:
            return False
        
        # Convert to grayscale for analysis if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Calculate image variance (measure of content)
        variance = np.var(gray)
        normalized_variance = variance / (255.0 ** 2)
        
        # Check if frame has sufficient content
        if normalized_variance < self.quality_threshold:
            return False
        
        # Check for extreme brightness/darkness
        mean_intensity = np.mean(gray)
        if mean_intensity < 10 or mean_intensity > 245:
            return False
        
        return True
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize frame to target size with proper aspect ratio handling.
        
        Args:
            frame: Input frame
            
        Returns:
            Resized frame
        """
        current_height, current_width = frame.shape[:2]
        target_width, target_height = self.target_size
        
        # Calculate aspect ratios
        current_aspect = current_width / current_height
        target_aspect = target_width / target_height
        
        if abs(current_aspect - target_aspect) < 0.01:
            # Aspect ratios are similar, direct resize
            resized = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)
        else:
            # Aspect ratios differ, use padding to maintain ratio
            if current_aspect > target_aspect:
                # Frame is wider, fit to width
                new_width = target_width
                new_height = int(target_width / current_aspect)
            else:
                # Frame is taller, fit to height
                new_height = target_height
                new_width = int(target_height * current_aspect)
            
            # Resize to calculated size
            resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            # Pad to target size
            pad_x = (target_width - new_width) // 2
            pad_y = (target_height - new_height) // 2
            
            if len(frame.shape) == 2:
                padded = np.zeros((target_height, target_width), dtype=frame.dtype)
                padded[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized
            else:
                padded = np.zeros((target_height, target_width, frame.shape[2]), dtype=frame.dtype)
                padded[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized
            
            resized = padded
        
        return resized
    
    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Normalize frame values for neural network training.
        
        Args:
            frame: Input frame
            
        Returns:
            Normalized frame with values in appropriate range
        """
        frame = frame.astype(np.float32)
        
        if self.normalize_method == 'minmax':
            # Min-max normalization to [0, 1]
            frame = frame / 255.0
            
        elif self.normalize_method == 'zscore':
            # Z-score normalization
            mean = self.normalization_stats['mean']
            std = self.normalization_stats['std']
            frame = (frame - mean) / std
            
        elif self.normalize_method == 'none':
            # No normalization
            pass
        
        else:
            raise ValueError(f"Unknown normalization method: {self.normalize_method}")
        
        return frame
    
    def compute_normalization_stats(self, frames: List[np.ndarray]) -> Dict:
        """
        Compute normalization statistics from a collection of frames.
        
        Args:
            frames: List of frames to analyze
            
        Returns:
            Dictionary containing normalization statistics
        """
        if not frames:
            return self.normalization_stats
        
        # Convert frames to float32 and stack
        frame_stack = np.stack([frame.astype(np.float32) for frame in frames])
        
        # Compute statistics
        stats = {
            'mean': np.mean(frame_stack),
            'std': np.std(frame_stack),
            'min': np.min(frame_stack),
            'max': np.max(frame_stack)
        }
        
        # Update internal stats
        self.normalization_stats.update(stats)
        
        print(f"Computed normalization statistics:")
        print(f"  Mean: {stats['mean']:.3f}")
        print(f"  Std: {stats['std']:.3f}")
        print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        
        return stats
    
    def process_video_file(
        self, 
        video_path: str, 
        max_frames: Optional[int] = None,
        frame_skip: int = 1,
        apply_augmentation: bool = False
    ) -> List[np.ndarray]:
        """
        Process an entire video file and return preprocessed frames.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to process
            frame_skip: Skip every N frames (1 = process all frames)
            apply_augmentation: Whether to apply data augmentation
            
        Returns:
            List of preprocessed frames
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        frames = []
        frame_count = 0
        processed_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if specified
                if frame_count % frame_skip != 0:
                    frame_count += 1
                    continue
                
                try:
                    processed_frame = self.process_frame(
                        frame, 
                        apply_augmentation=apply_augmentation
                    )
                    frames.append(processed_frame)
                    processed_count += 1
                    
                    # Check frame limit
                    if max_frames and processed_count >= max_frames:
                        break
                        
                except ValueError as e:
                    # Skip low-quality frames
                    warnings.warn(f"Skipping frame {frame_count}: {e}")
                
                frame_count += 1
        
        finally:
            cap.release()
        
        print(f"Processed {processed_count} frames from {video_path}")
        return frames
    
    def get_preprocessing_stats(self) -> Dict:
        """Get preprocessing statistics."""
        total_frames = self.stats['frames_processed'] + self.stats['frames_rejected']
        rejection_rate = self.stats['frames_rejected'] / max(1, total_frames)
        
        return {
            'frames_processed': self.stats['frames_processed'],
            'frames_rejected': self.stats['frames_rejected'],
            'rejection_rate': rejection_rate,
            'target_size': self.target_size,
            'grayscale': self.convert_to_grayscale,
            'normalization_method': self.normalize_method,
            'augmentation_enabled': self.augmentation_pipeline is not None
        }


class TemporalPreprocessor:
    """
    Preprocessor for temporal sequences of video frames.
    
    This class handles preprocessing of frame sequences for temporal
    anomaly detection, maintaining temporal consistency across frames.
    """
    
    def __init__(
        self,
        sequence_length: int = 8,
        overlap: int = 4,
        frame_preprocessor: Optional[VideoPreprocessor] = None
    ):
        """
        Initialize temporal preprocessor.
        
        Args:
            sequence_length: Number of frames in each sequence
            overlap: Number of overlapping frames between sequences
            frame_preprocessor: Frame-level preprocessor
        """
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.frame_preprocessor = frame_preprocessor or VideoPreprocessor()
        
        if overlap >= sequence_length:
            raise ValueError("Overlap must be less than sequence length")
    
    def create_sequences(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Create overlapping sequences from frame list.
        
        Args:
            frames: List of individual frames
            
        Returns:
            List of frame sequences
        """
        if len(frames) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} frames")
        
        sequences = []
        step = self.sequence_length - self.overlap
        
        for i in range(0, len(frames) - self.sequence_length + 1, step):
            sequence = frames[i:i + self.sequence_length]
            sequences.append(np.stack(sequence))
        
        return sequences
    
    def process_frame_sequence(
        self, 
        frames: List[np.ndarray],
        apply_augmentation: bool = False
    ) -> np.ndarray:
        """
        Process a sequence of frames maintaining temporal consistency.
        
        Args:
            frames: List of frames in temporal order
            apply_augmentation: Whether to apply consistent augmentation
            
        Returns:
            Processed frame sequence as numpy array
        """
        if len(frames) != self.sequence_length:
            raise ValueError(f"Expected {self.sequence_length} frames, got {len(frames)}")
        
        processed_frames = []
        
        # Apply same augmentation parameters to all frames for consistency
        if apply_augmentation and self.frame_preprocessor.augmentation_pipeline:
            # Generate augmentation parameters once for the sequence
            sample_frame = frames[0]
            if len(sample_frame.shape) == 2:
                sample_frame = np.expand_dims(sample_frame, axis=2)
            
            augmentation_params = self.frame_preprocessor.augmentation_pipeline.get_params()
        
        for frame in frames:
            if apply_augmentation and self.frame_preprocessor.augmentation_pipeline:
                # Apply same augmentation to maintain temporal consistency
                if len(frame.shape) == 2:
                    frame = np.expand_dims(frame, axis=2)
                
                augmented = self.frame_preprocessor.augmentation_pipeline.apply_with_params(
                    augmentation_params, image=frame
                )
                frame = augmented['image'].squeeze()
            
            processed_frame = self.frame_preprocessor.process_frame(
                frame, 
                apply_augmentation=False  # Already handled above
            )
            processed_frames.append(processed_frame)
        
        return np.stack(processed_frames)


class BatchPreprocessor:
    """
    Efficient batch preprocessing for large datasets.
    
    This class optimizes preprocessing for batch operations,
    useful for processing large video datasets efficiently.
    """
    
    def __init__(
        self,
        preprocessor: VideoPreprocessor,
        batch_size: int = 32,
        num_workers: int = 4
    ):
        """
        Initialize batch preprocessor.
        
        Args:
            preprocessor: Frame preprocessor
            batch_size: Number of frames to process in each batch
            num_workers: Number of worker processes (not implemented)
        """
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def process_frame_batch(
        self, 
        frames: List[np.ndarray],
        apply_augmentation: bool = False
    ) -> List[np.ndarray]:
        """
        Process a batch of frames efficiently.
        
        Args:
            frames: List of frames to process
            apply_augmentation: Whether to apply augmentation
            
        Returns:
            List of processed frames
        """
        processed_frames = []
        
        for i in range(0, len(frames), self.batch_size):
            batch = frames[i:i + self.batch_size]
            
            batch_processed = []
            for frame in batch:
                try:
                    processed = self.preprocessor.process_frame(
                        frame,
                        apply_augmentation=apply_augmentation
                    )
                    batch_processed.append(processed)
                except ValueError:
                    # Skip invalid frames
                    continue
            
            processed_frames.extend(batch_processed)
        
        return processed_frames


# Utility functions for common preprocessing tasks

def extract_frames_from_video(
    video_path: str,
    output_dir: Optional[str] = None,
    frame_format: str = 'jpg',
    max_frames: Optional[int] = None
) -> List[str]:
    """
    Extract frames from video and save to directory.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save frames (if None, frames not saved)
        frame_format: Format for saved frames ('jpg', 'png')
        max_frames: Maximum number of frames to extract
        
    Returns:
        List of frame file paths (if saved) or empty list
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    frame_paths = []
    frame_count = 0
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if output_dir:
                frame_filename = f"frame_{frame_count:06d}.{frame_format}"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
            
            frame_count += 1
            
            if max_frames and frame_count >= max_frames:
                break
    
    finally:
        cap.release()
    
    print(f"Extracted {frame_count} frames from {video_path}")
    return frame_paths


def create_video_from_frames(
    frame_paths: List[str],
    output_path: str,
    fps: float = 30.0,
    codec: str = 'XVID'
) -> None:
    """
    Create video from list of frame images.
    
    Args:
        frame_paths: List of paths to frame images
        output_path: Path for output video
        fps: Frames per second for output video
        codec: Video codec to use
    """
    if not frame_paths:
        raise ValueError("No frame paths provided")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        raise ValueError(f"Cannot read first frame: {frame_paths[0]}")
    
    height, width = first_frame.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    try:
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            if frame is not None:
                out.write(frame)
    
    finally:
        out.release()
    
    print(f"Created video: {output_path}")


if __name__ == "__main__":
    # Test preprocessing pipeline
    print("Testing Video Preprocessing Pipeline...")
    
    # Create test frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test basic preprocessor
    preprocessor = VideoPreprocessor(
        target_size=(64, 64),
        convert_to_grayscale=True,
        normalize_method='minmax'
    )
    
    processed = preprocessor.process_frame(test_frame)
    print(f\"Processed frame shape: {processed.shape}\")
    print(f\"Processed frame range: [{processed.min():.3f}, {processed.max():.3f}]\")
    
    # Test augmentation
    aug_config = {
        'enabled': True,
        'rotation': True,
        'brightness_contrast': True,
        'gaussian_noise': True
    }
    
    aug_preprocessor = VideoPreprocessor(
        target_size=(64, 64),
        augmentation_config=aug_config
    )
    
    augmented = aug_preprocessor.process_frame(test_frame, apply_augmentation=True)
    print(f\"Augmented frame shape: {augmented.shape}\")
    
    # Test temporal preprocessor
    temporal_preprocessor = TemporalPreprocessor(
        sequence_length=4,
        overlap=2,
        frame_preprocessor=preprocessor
    )
    
    test_frames = [test_frame for _ in range(8)]
    sequences = temporal_preprocessor.create_sequences([
        preprocessor.process_frame(f) for f in test_frames
    ])
    
    print(f\"Created {len(sequences)} temporal sequences\")
    print(f\"Sequence shape: {sequences[0].shape}\")
    
    # Test statistics
    stats = preprocessor.get_preprocessing_stats()
    print(f\"Preprocessing stats: {stats}\")
    
    print(\"\\nAll preprocessing tests completed!\")
