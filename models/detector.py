"""
Anomaly Detection System
========================

This module implements the complete anomaly detection pipeline that coordinates
training, threshold establishment, and inference. It acts as the control center
that manages the entire anomaly detection workflow.

Key Components:
- Model training with early stopping and checkpointing
- Threshold calculation using statistical methods
- Real-time anomaly detection
- Performance monitoring and logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import os
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
import warnings

from .autoencoder import ConvolutionalAutoencoder


class EarlyStopping:
    """
    Early stopping utility to prevent overfitting during training.
    
    Monitors training loss and stops training when the model stops improving,
    helping to find the optimal point before overfitting occurs.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-6, verbose: bool = True):
        """
        Initialize early stopping monitor.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            verbose: Whether to print early stopping messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, loss: float) -> bool:
        """
        Check if training should stop early.
        
        Args:
            loss: Current epoch loss
            
        Returns:
            True if training should stop, False otherwise
        """
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.verbose:
                    print(f"Early stopping triggered after {self.counter} epochs without improvement")
                self.early_stop = True
        
        return self.early_stop


class AnomalyDetector:
    """
    Complete anomaly detection system that manages the entire pipeline.
    
    This class orchestrates:
    1. Training the autoencoder on normal data
    2. Establishing anomaly detection thresholds
    3. Detecting anomalies in new data
    4. Performance evaluation and monitoring
    
    The system is optimized for RTX 3050 performance with mixed precision training
    and efficient memory management.
    """
    
    def __init__(
        self,
        model: ConvolutionalAutoencoder,
        device: torch.device,
        config: object = None
    ):
        """
        Initialize the anomaly detection system.
        
        Args:
            model: Convolutional autoencoder model
            device: Computing device (CPU or GPU)
            config: Configuration object with hyperparameters
        """
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Training state
        self.is_trained = False
        self.training_history = {
            'losses': [],
            'epochs': [],
            'learning_rates': []
        }
        
        # Threshold state
        self.threshold = None
        self.normal_errors = []
        self.threshold_stats = {}
        
        # Performance monitoring
        self.training_time = 0
        self.inference_times = []
        
        # Mixed precision training for RTX 3050 optimization
        self.use_mixed_precision = getattr(config, 'MIXED_PRECISION', True) if config else True
        if self.use_mixed_precision and device.type == 'cuda':
            self.scaler = torch.amp.GradScaler('cuda')
            print("✓ Mixed precision training enabled for faster performance")
        else:
            self.scaler = None
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 50,
        learning_rate: float = 0.001,
        save_path: Optional[str] = None,
        early_stopping_patience: int = 10
    ) -> Dict:
        """
        Train the autoencoder on normal video frames.
        
        This method implements the core learning process where the network
        learns to reconstruct normal patterns. The better it becomes at
        reconstructing normal data, the worse it will be at abnormal data.
        
        Args:
            train_loader: DataLoader with normal training frames
            val_loader: Optional validation DataLoader
            num_epochs: Maximum number of training epochs
            learning_rate: Initial learning rate
            save_path: Path to save the best model
            early_stopping_patience: Epochs to wait before early stopping
            
        Returns:
            Dictionary containing training statistics
        """
        print("=" * 60)
        print("TRAINING ANOMALY DETECTION MODEL")
        print("=" * 60)
        
        start_time = time.time()
        
        # Setup optimization
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=getattr(self.config, 'WEIGHT_DECAY', 1e-5) if self.config else 1e-5
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=getattr(self.config, 'LR_PATIENCE', 5) if self.config else 5,
            factor=getattr(self.config, 'LR_FACTOR', 0.5) if self.config else 0.5,
            min_lr=getattr(self.config, 'LR_MIN', 1e-6) if self.config else 1e-6
        )
        
        # Loss function
        criterion = nn.MSELoss()
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=1e-6,
            verbose=True
        )
        
        # Training tracking
        best_loss = float('inf')
        best_model_state = None
        
        self.model.train()
        
        print(f"Training Configuration:")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Learning Rate: {learning_rate}")
        print(f"  Batch Size: {train_loader.batch_size}")
        print(f"  Training Samples: {len(train_loader.dataset)}")
        if val_loader:
            print(f"  Validation Samples: {len(val_loader.dataset)}")
        print(f"  Mixed Precision: {self.scaler is not None}")
        print()
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss = self._train_epoch(train_loader, optimizer, criterion, epoch)
            
            # Validation phase (if validation data provided)
            val_loss = None
            if val_loader:
                val_loss = self._validate_epoch(val_loader, criterion)
            
            # Learning rate scheduling
            scheduler_loss = val_loss if val_loss is not None else train_loss
            scheduler.step(scheduler_loss)
            
            # Record training history
            self.training_history['losses'].append(train_loss)
            self.training_history['epochs'].append(epoch + 1)
            self.training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # Check for best model
            monitor_loss = val_loss if val_loss is not None else train_loss
            if monitor_loss < best_loss:
                best_loss = monitor_loss
                best_model_state = self.model.state_dict().copy()
                
                # Save best model if path provided
                if save_path:
                    self._save_model(save_path, epoch, monitor_loss, optimizer)
            
            # Progress reporting
            epoch_time = time.time() - epoch_start_time
            self._print_epoch_progress(
                epoch + 1, num_epochs, train_loss, val_loss, 
                epoch_time, optimizer.param_groups[0]['lr']
            )
            
            # Early stopping check
            if early_stopping(monitor_loss):
                print(f"Early stopping at epoch {epoch + 1}")
                break
            
            # GPU memory management
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"✓ Loaded best model with loss: {best_loss:.6f}")
        
        # Training summary
        self.training_time = time.time() - start_time
        self.is_trained = True
        
        training_stats = {
            'total_time': self.training_time,
            'epochs_completed': len(self.training_history['losses']),
            'best_loss': best_loss,
            'final_learning_rate': optimizer.param_groups[0]['lr'],
            'early_stopped': early_stopping.early_stop
        }
        
        print(f"\n✓ Training completed in {self.training_time:.1f} seconds")
        print(f"✓ Best reconstruction loss: {best_loss:.6f}")
        
        return training_stats
    
    def _train_epoch(
        self, 
        train_loader: DataLoader, 
        optimizer: optim.Optimizer, 
        criterion: nn.Module,
        epoch: int
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer for gradient updates
            criterion: Loss function
            epoch: Current epoch number
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        # Progress bar for training batches
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False)
        
        for batch_idx, (data, target) in enumerate(pbar):
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    reconstruction = self.model(data)
                    loss = criterion(reconstruction, target)
                
                # Mixed precision backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                # Standard forward/backward pass
                reconstruction = self.model(data)
                loss = criterion(reconstruction, target)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'Loss': f"{loss.item():.6f}"})
            
            # Memory cleanup for large batches
            if batch_idx % 50 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return total_loss / num_batches
    
    def _validate_epoch(self, val_loader: DataLoader, criterion: nn.Module) -> float:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        reconstruction = self.model(data)
                        loss = criterion(reconstruction, target)
                else:
                    reconstruction = self.model(data)
                    loss = criterion(reconstruction, target)
                
                total_loss += loss.item()
        
        return total_loss / num_batches
    
    def _print_epoch_progress(
        self, 
        epoch: int, 
        total_epochs: int, 
        train_loss: float,
        val_loss: Optional[float], 
        epoch_time: float, 
        lr: float
    ):
        """Print progress for current epoch."""
        progress = f"Epoch {epoch:3d}/{total_epochs}"
        train_info = f"Train Loss: {train_loss:.6f}"
        val_info = f"Val Loss: {val_loss:.6f}" if val_loss is not None else ""
        time_info = f"Time: {epoch_time:.1f}s"
        lr_info = f"LR: {lr:.2e}"
        
        print(f"{progress} | {train_info} | {val_info} | {time_info} | {lr_info}")
    
    def establish_threshold(
        self,
        normal_data_loader: DataLoader,
        threshold_factor: float = 2.5,
        method: str = 'statistical'
    ) -> Dict:
        """
        Establish anomaly detection threshold based on normal data reconstruction errors.
        
        This critical step determines the sensitivity of anomaly detection.
        The threshold separates normal from anomalous reconstruction errors.
        
        Args:
            normal_data_loader: DataLoader with normal frames
            threshold_factor: Multiplier for standard deviation
            method: Threshold calculation method ('statistical' or 'percentile')
            
        Returns:
            Dictionary with threshold statistics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before establishing threshold")
        
        print("Establishing anomaly detection threshold...")
        
        self.model.eval()
        reconstruction_errors = []
        
        with torch.no_grad():
            for data, _ in tqdm(normal_data_loader, desc="Computing normal errors"):
                data = data.to(self.device, non_blocking=True)
                
                # Get reconstruction
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        reconstruction = self.model(data)
                else:
                    reconstruction = self.model(data)
                
                # Calculate per-frame reconstruction error
                errors = ((data - reconstruction) ** 2).mean(dim=[1, 2, 3])
                reconstruction_errors.extend(errors.cpu().numpy())
        
        self.normal_errors = np.array(reconstruction_errors)
        
        # Calculate threshold based on method
        if method == 'statistical':
            mean_error = np.mean(self.normal_errors)
            std_error = np.std(self.normal_errors)
            self.threshold = mean_error + threshold_factor * std_error
        elif method == 'percentile':
            percentile = getattr(self.config, 'PERCENTILE_THRESHOLD', 95) if self.config else 95
            self.threshold = np.percentile(self.normal_errors, percentile)
        else:
            raise ValueError(f"Unknown threshold method: {method}")
        
        # Calculate threshold statistics
        self.threshold_stats = {
            'method': method,
            'threshold': self.threshold,
            'mean_error': np.mean(self.normal_errors),
            'std_error': np.std(self.normal_errors),
            'min_error': np.min(self.normal_errors),
            'max_error': np.max(self.normal_errors),
            'threshold_factor': threshold_factor,
            'false_positive_rate': np.sum(self.normal_errors > self.threshold) / len(self.normal_errors)
        }
        
        print(f"✓ Threshold established using {method} method")
        print(f"  Threshold value: {self.threshold:.6f}")
        print(f"  Mean normal error: {self.threshold_stats['mean_error']:.6f}")
        print(f"  Std normal error: {self.threshold_stats['std_error']:.6f}")
        print(f"  False positive rate: {self.threshold_stats['false_positive_rate']:.4f}")
        
        return self.threshold_stats
    
    def detect_anomalies(self, test_data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies in test data.
        
        Args:
            test_data_loader: DataLoader with test frames
            
        Returns:
            Tuple of (reconstruction_errors, anomaly_flags)
        """
        if self.threshold is None:
            raise ValueError("Threshold not established. Call establish_threshold() first.")
        
        print("Detecting anomalies in test data...")
        
        self.model.eval()
        reconstruction_errors = []
        
        inference_start = time.time()
        
        with torch.no_grad():
            for data, _ in tqdm(test_data_loader, desc="Processing test frames"):
                batch_start = time.time()
                
                data = data.to(self.device, non_blocking=True)
                
                # Get reconstruction
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        reconstruction = self.model(data)
                else:
                    reconstruction = self.model(data)
                
                # Calculate reconstruction error
                errors = ((data - reconstruction) ** 2).mean(dim=[1, 2, 3])
                reconstruction_errors.extend(errors.cpu().numpy())
                
                # Track inference time
                batch_time = time.time() - batch_start
                self.inference_times.append(batch_time / len(data))  # Per-frame time
        
        reconstruction_errors = np.array(reconstruction_errors)
        anomaly_flags = reconstruction_errors > self.threshold
        
        # Performance statistics
        total_inference_time = time.time() - inference_start
        avg_inference_time = np.mean(self.inference_times)
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        
        anomaly_count = np.sum(anomaly_flags)
        total_count = len(reconstruction_errors)
        
        print(f"✓ Anomaly detection completed")
        print(f"  Detected {anomaly_count} anomalies out of {total_count} frames")
        print(f"  Anomaly rate: {anomaly_count/total_count:.2%}")
        print(f"  Average inference time: {avg_inference_time*1000:.2f}ms per frame")
        print(f"  Processing speed: {fps:.1f} FPS")
        
        return reconstruction_errors, anomaly_flags
    
    def _save_model(self, save_path: str, epoch: int, loss: float, optimizer: optim.Optimizer):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'threshold': self.threshold,
            'training_history': self.training_history,
            'threshold_stats': self.threshold_stats
        }
        
        torch.save(checkpoint, save_path)
    
    def load_model(self, load_path: str, load_optimizer: bool = False) -> Dict:
        """
        Load saved model checkpoint.
        
        Args:
            load_path: Path to saved checkpoint
            load_optimizer: Whether to load optimizer state
            
        Returns:
            Dictionary with loaded checkpoint information
        """
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.threshold = checkpoint.get('threshold')
        self.training_history = checkpoint.get('training_history', {})
        self.threshold_stats = checkpoint.get('threshold_stats', {})
        self.is_trained = True
        
        print(f"✓ Model loaded from {load_path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Loss: {checkpoint['loss']:.6f}")
        
        return checkpoint
    
    def get_performance_stats(self) -> Dict:
        """Get comprehensive performance statistics."""
        return {
            'training_time': self.training_time,
            'is_trained': self.is_trained,
            'threshold': self.threshold,
            'avg_inference_time': np.mean(self.inference_times) if self.inference_times else 0,
            'training_history': self.training_history,
            'threshold_stats': self.threshold_stats,
            'model_info': self.model.get_model_info()
        }


if __name__ == "__main__":
    # Test the detector
    from .autoencoder import ConvolutionalAutoencoder
    
    print("Testing Anomaly Detector...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvolutionalAutoencoder(latent_dim=128)
    detector = AnomalyDetector(model, device)
    
    print(f"✓ Detector initialized on {device}")
    
    # Create dummy data
    dummy_data = torch.randn(10, 1, 64, 64)
    dummy_dataset = torch.utils.data.TensorDataset(dummy_data, dummy_data)
    dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=4)
    
    # Quick training test
    stats = detector.train(dummy_loader, num_epochs=2)
    print(f"✓ Training test completed: {stats}")
    
    # Threshold test
    threshold_stats = detector.establish_threshold(dummy_loader)
    print(f"✓ Threshold test completed: {threshold_stats['threshold']:.6f}")
    
    print("\n✓ Detector test completed successfully!")