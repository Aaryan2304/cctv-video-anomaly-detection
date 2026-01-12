"""
Convolutional autoencoder for video frame reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class ConvolutionalAutoencoder(nn.Module):
    """
    Symmetric encoder-decoder for 64x64 grayscale frames.
    Encoder: 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4 -> latent_dim
    """
    
    def __init__(self, input_channels: int = 1, latent_dim: int = 256):
        super(ConvolutionalAutoencoder, self).__init__()
        
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        
        # Encoder: 64x64 -> 4x4 -> latent_dim
        self.encoder = nn.Sequential(
            # Block 1: 64x64 -> 32x32
            nn.Conv2d(input_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            
            # Block 2: 32x32 -> 16x16
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            
            # Block 3: 16x16 -> 8x8
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            
            # Block 4: 8x8 -> 4x4
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
        )
        
        # Bottleneck
        self.flatten = nn.Flatten()
        self.encode_fc = nn.Linear(4 * 4 * 256, latent_dim)
        self.decode_fc = nn.Linear(latent_dim, 4 * 4 * 256)
        
        # Decoder: latent_dim -> 4x4 -> 64x64
        self.decoder = nn.Sequential(
            # Block 1: 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            
            # Block 2: 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            
            # Block 3: 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            
            # Block 4: 32x32 -> 64x64
            nn.ConvTranspose2d(32, input_channels, 4, stride=2, padding=1),
            nn.Sigmoid()  # Output in [0,1] range
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for better training."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through autoencoder."""
        # Encode
        encoded = self.encoder(x)
        batch_size = encoded.size(0)
        
        # Bottleneck
        flattened = self.flatten(encoded)
        latent = self.encode_fc(flattened)
        
        # Decode
        decoded_flat = self.decode_fc(latent)
        decoded_spatial = decoded_flat.view(batch_size, 256, 4, 4)
        reconstruction = self.decoder(decoded_spatial)
        
        return reconstruction
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation."""
        encoded = self.encoder(x)
        flattened = self.flatten(encoded)
        latent = self.encode_fc(flattened)
        return latent
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode from latent representation."""
        batch_size = latent.size(0)
        decoded_flat = self.decode_fc(latent)
        decoded_spatial = decoded_flat.view(batch_size, 256, 4, 4)
        reconstruction = self.decoder(decoded_spatial)
        return reconstruction
    
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction error per sample."""
        reconstruction = self.forward(x)
        error = torch.mean((x - reconstruction) ** 2, dim=[1, 2, 3])
        return error
    
    def get_model_info(self) -> dict:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'latent_dimension': self.latent_dim,
            'input_channels': self.input_channels,
            'model_size_mb': total_params * 4 / (1024 * 1024)
        }


class LightweightAutoencoder(nn.Module):
    """Lightweight version for faster training and inference."""
    
    def __init__(self, input_channels: int = 1, latent_dim: int = 128):
        super(LightweightAutoencoder, self).__init__()
        
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        
        # Simpler encoder: 64x64 -> 8x8 -> latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, 4, stride=2, padding=1),  # 32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),  # 16x16
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 8x8
            nn.ReLU(inplace=True),
        )
        
        # Bottleneck
        self.flatten = nn.Flatten()
        self.encode_fc = nn.Linear(8 * 8 * 64, latent_dim)
        self.decode_fc = nn.Linear(latent_dim, 8 * 8 * 64)
        
        # Simpler decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 16x16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # 32x32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, input_channels, 4, stride=2, padding=1),  # 64x64
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        batch_size = encoded.size(0)
        
        flattened = self.flatten(encoded)
        latent = self.encode_fc(flattened)
        
        decoded_flat = self.decode_fc(latent)
        decoded_spatial = decoded_flat.view(batch_size, 64, 8, 8)
        reconstruction = self.decoder(decoded_spatial)
        
        return reconstruction
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        flattened = self.flatten(encoded)
        latent = self.encode_fc(flattened)
        return latent
    
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        reconstruction = self.forward(x)
        error = torch.mean((x - reconstruction) ** 2, dim=[1, 2, 3])
        return error
    
    def get_model_info(self) -> dict:
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'total_parameters': total_params,
            'trainable_parameters': total_params,
            'latent_dimension': self.latent_dim,
            'input_channels': self.input_channels,
            'model_size_mb': total_params * 4 / (1024 * 1024)
        }


def create_autoencoder(model_type: str = 'standard', **kwargs):
    """Factory function to create autoencoder models."""
    if model_type == 'standard':
        return ConvolutionalAutoencoder(**kwargs)
    elif model_type == 'lightweight':
        return LightweightAutoencoder(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test the autoencoder
    model = ConvolutionalAutoencoder()
    print("Testing autoencoder...")
    
    # Test forward pass
    x = torch.randn(4, 1, 64, 64)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test reconstruction error
    error = model.reconstruction_error(x)
    print(f"Reconstruction error shape: {error.shape}")
    
    # Model info
    info = model.get_model_info()
    print(f"Model info: {info}")
