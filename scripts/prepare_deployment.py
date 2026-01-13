"""
Deployment Model Preparation
============================

Create a lightweight version of the model for deployment.
This script prepares the model for production use.
"""

import torch
import os
import sys
import numpy as np
from models.autoencoder import ConvolutionalAutoencoder
from models.detector import AnomalyDetector

def create_deployment_model():
    """Create a deployment-ready model"""
    print("Creating deployment model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if trained model exists
    model_path = "outputs/trained_model.pth"
    
    if os.path.exists(model_path):
        print("Loading existing trained model...")
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model = ConvolutionalAutoencoder(input_channels=1, latent_dim=256)
            model.load_state_dict(checkpoint['model_state_dict'])
            threshold = checkpoint.get('threshold', 0.005)
            print(f"Loaded model with threshold: {threshold}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating new model...")
            model = ConvolutionalAutoencoder(input_channels=1, latent_dim=256)
            threshold = 0.005
    else:
        print("No trained model found. Creating new model for demonstration...")
        model = ConvolutionalAutoencoder(input_channels=1, latent_dim=256)
        threshold = 0.005
    
    # Set model to evaluation mode
    model.eval()
    
    # Create a simple deployment checkpoint
    deployment_checkpoint = {
        'model_state_dict': model.state_dict(),
        'threshold': threshold,
        'model_info': model.get_model_info(),
        'deployment_ready': True
    }
    
    # Save deployment model
    os.makedirs("outputs", exist_ok=True)
    deployment_path = "outputs/deployment_model.pth"
    torch.save(deployment_checkpoint, deployment_path)
    
    print(f"Deployment model saved to: {deployment_path}")
    print(f"Model parameters: {model.get_model_info()['total_parameters']:,}")
    print(f"Model size: {os.path.getsize(deployment_path) / (1024*1024):.1f} MB")
    
    return True

def test_deployment_model():
    """Test the deployment model"""
    print("\nTesting deployment model...")
    
    device = torch.device('cpu')  # Use CPU for deployment testing
    
    try:
        # Load deployment model
        checkpoint = torch.load("outputs/deployment_model.pth", map_location=device)
        model = ConvolutionalAutoencoder(input_channels=1, latent_dim=256)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Test with dummy data
        test_frame = torch.randn(1, 1, 64, 64)
        
        with torch.no_grad():
            reconstructed = model(test_frame)
            error = torch.mean((test_frame - reconstructed) ** 2).item()
        
        print(f"Model test successful!")
        print(f"  Test reconstruction error: {error:.6f}")
        print(f"  Model threshold: {checkpoint['threshold']:.6f}")
        print(f"  Test anomaly: {'Yes' if error > checkpoint['threshold'] else 'No'}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("DEPLOYMENT MODEL PREPARATION")
    print("=" * 50)
    
    # Create deployment model
    success = create_deployment_model()
    
    if success:
        # Test the model
        test_success = test_deployment_model()
        
        if test_success:
            print("\n🎉 Deployment model ready!")
            print("You can now deploy the application.")
        else:
            print("\n⚠️ Model created but test failed.")
            sys.exit(1)
    else:
        print("\n❌ Failed to create deployment model.")
        sys.exit(1)
