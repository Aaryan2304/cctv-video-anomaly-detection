"""
Configuration Management for Video Anomaly Detection
==================================================

This file centralizes all configuration parameters, making it easy to:
- Experiment with different hyperparameters
- Optimize for different hardware configurations
- Switch between datasets and experiments
- Maintain reproducible results

Your RTX 3050 optimized settings are included as defaults.
"""

import torch
import os

class Config:
    """
    Centralized configuration for the anomaly detection system.
    
    This class organizes all hyperparameters and settings in one place,
    making experimentation and reproducibility much easier.
    """
    
    # ====================================================================
    # HARDWARE CONFIGURATION (Optimized for RTX 3050)
    # ====================================================================
    
    # Device selection
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Memory optimization for RTX 3050 (4GB VRAM)
    BATCH_SIZE = 64  # Sweet spot for your VRAM
    NUM_WORKERS = 4  # Utilize your 12th gen i7 cores
    PIN_MEMORY = True  # Faster GPU transfers
    
    # Training efficiency
    MIXED_PRECISION = True  # Use half-precision for faster training
    
    # ====================================================================
    # MODEL ARCHITECTURE
    # ====================================================================
    
    # Input parameters
    INPUT_CHANNELS = 1  # Grayscale for efficiency
    FRAME_HEIGHT = 64
    FRAME_WIDTH = 64
    FRAME_SIZE = (FRAME_WIDTH, FRAME_HEIGHT)
    
    # Autoencoder architecture
    LATENT_DIM = 256  # Balanced for performance/accuracy
    
    # ====================================================================
    # TRAINING CONFIGURATION
    # ====================================================================
    
    # Training parameters
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    
    # Epochs (optimized for different scenarios)
    EPOCHS_SYNTHETIC = 25  # Quick demo
    EPOCHS_UCSD = 40      # Real dataset
    EPOCHS_FULL = 50      # Thorough training
    
    # Learning rate scheduling
    LR_PATIENCE = 5
    LR_FACTOR = 0.5
    LR_MIN = 1e-6
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 10
    EARLY_STOPPING_DELTA = 1e-6
    
    # ====================================================================
    # DATA CONFIGURATION
    # ====================================================================
    
    # Data splitting
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    
    # Video processing
    MAX_FRAMES_PER_VIDEO = 200  # Prevent VRAM overflow
    FRAME_SKIP = 1  # Process every nth frame (1 = all frames)
    
    # Data augmentation
    USE_AUGMENTATION = True
    NOISE_FACTOR = 0.05
    
    # ====================================================================
    # ANOMALY DETECTION
    # ====================================================================
    
    # Threshold calculation
    THRESHOLD_FACTOR = 2.5  # Multiplier for std deviation
    PERCENTILE_THRESHOLD = 95  # Alternative percentile-based threshold
    
    # Detection modes
    THRESHOLD_MODE = 'statistical'  # 'statistical' or 'percentile'
    
    # ====================================================================
    # PATHS AND DIRECTORIES
    # ====================================================================
    
    # Project structure
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    MODELS_DIR = os.path.join(PROJECT_ROOT, 'saved_models')
    OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')
    LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
    
    # UCSD dataset paths (update these for your setup)
    UCSD_ROOT = '/path/to/ucsd/ped2'  # UPDATE THIS PATH
    UCSD_TRAIN = os.path.join(UCSD_ROOT, 'Train')
    UCSD_TEST = os.path.join(UCSD_ROOT, 'Test')
    UCSD_GT = os.path.join(UCSD_ROOT, 'Test_GT')  # Ground truth labels
    
    # ====================================================================
    # SYNTHETIC DATA CONFIGURATION
    # ====================================================================
    
    # Dataset sizes
    SYNTHETIC_NORMAL_FRAMES = 800   # Reduced for faster demo
    SYNTHETIC_ANOMALY_FRAMES = 80   # 10% anomaly ratio
    
    # Pattern parameters
    CIRCLE_RADIUS_RANGE = (5, 12)
    MOVEMENT_AMPLITUDE = 20
    NOISE_LEVEL = 0.03
    
    # ====================================================================
    # VISUALIZATION SETTINGS
    # ====================================================================
    
    # Plot settings
    FIGURE_SIZE = (12, 8)
    DPI = 150
    SAVE_PLOTS = True
    
    # Demo visualization
    DEMO_EXAMPLES = 8
    
    # ====================================================================
    # LOGGING AND MONITORING
    # ====================================================================
    
    # Progress reporting
    LOG_INTERVAL = 10  # Log every N batches
    SAVE_INTERVAL = 5  # Save model every N epochs
    
    # Monitoring
    MONITOR_GPU = True
    VERBOSE = True
    
    # ====================================================================
    # EXPERIMENTAL SETTINGS
    # ====================================================================
    
    # Random seed for reproducibility
    RANDOM_SEED = 42
    
    # Model variations to try
    ARCHITECTURES = {
        'small': {'latent_dim': 128, 'epochs': 20},
        'medium': {'latent_dim': 256, 'epochs': 30},
        'large': {'latent_dim': 512, 'epochs': 40}
    }
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist."""
        directories = [
            cls.DATA_DIR,
            cls.MODELS_DIR, 
            cls.OUTPUTS_DIR,
            cls.LOGS_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
    @classmethod
    def get_device_info(cls):
        """Return detailed information about the computing device."""
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            return {
                'device': 'GPU',
                'name': device_name,
                'memory_gb': total_memory,
                'mixed_precision_available': True
            }
        else:
            return {
                'device': 'CPU',
                'name': 'CPU',
                'memory_gb': 'N/A',
                'mixed_precision_available': False
            }
    
    @classmethod
    def optimize_for_hardware(cls):
        """
        Automatically optimize settings based on available hardware.
        This ensures optimal performance on your RTX 3050.
        """
        device_info = cls.get_device_info()
        
        if device_info['device'] == 'GPU':
            memory_gb = device_info['memory_gb']
            
            # Optimize batch size based on VRAM
            if memory_gb >= 8:
                cls.BATCH_SIZE = 128
            elif memory_gb >= 6:
                cls.BATCH_SIZE = 96
            elif memory_gb >= 4:  # Your RTX 3050
                cls.BATCH_SIZE = 64
            else:
                cls.BATCH_SIZE = 32
                
            print(f"✓ Optimized for {device_info['name']} ({memory_gb:.1f}GB)")
            print(f"✓ Batch size set to: {cls.BATCH_SIZE}")
        else:
            # CPU optimization
            cls.BATCH_SIZE = 16
            cls.MIXED_PRECISION = False
            print("✓ Optimized for CPU processing")
    
    @classmethod
    def set_ucsd_path(cls, path):
        """Update UCSD dataset path."""
        cls.UCSD_ROOT = path
        cls.UCSD_TRAIN = os.path.join(path, 'Train')
        cls.UCSD_TEST = os.path.join(path, 'Test') 
        cls.UCSD_GT = os.path.join(path, 'Test_GT')
        print(f"✓ UCSD dataset path updated: {path}")
    
    @classmethod
    def print_config(cls):
        """Print current configuration for verification."""
        print("=" * 60)
        print("ANOMALY DETECTION CONFIGURATION")
        print("=" * 60)
        
        print(f"Device: {cls.DEVICE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Frame Size: {cls.FRAME_SIZE}")
        print(f"Latent Dimension: {cls.LATENT_DIM}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Threshold Factor: {cls.THRESHOLD_FACTOR}")
        print(f"Mixed Precision: {cls.MIXED_PRECISION}")
        
        print("\nPaths:")
        print(f"  Models: {cls.MODELS_DIR}")
        print(f"  Outputs: {cls.OUTPUTS_DIR}")
        print(f"  UCSD Root: {cls.UCSD_ROOT}")
        
        print("=" * 60)


# ====================================================================
# SPECIALIZED CONFIGURATIONS
# ====================================================================

class QuickDemoConfig(Config):
    """Fast configuration for quick demonstrations."""
    EPOCHS_SYNTHETIC = 15
    SYNTHETIC_NORMAL_FRAMES = 400
    SYNTHETIC_ANOMALY_FRAMES = 40
    LATENT_DIM = 128

class HighQualityConfig(Config):
    """High-quality configuration for best results."""
    EPOCHS_SYNTHETIC = 50
    EPOCHS_UCSD = 60
    LATENT_DIM = 512
    BATCH_SIZE = 32  # Larger model needs smaller batches

class ProductionConfig(Config):
    """Optimized configuration for production deployment."""
    LATENT_DIM = 256
    FRAME_SIZE = (32, 32)  # Smaller for speed
    BATCH_SIZE = 128
    MIXED_PRECISION = True


# Initialize configuration
def get_config(mode='default'):
    """
    Get configuration based on mode.
    
    Args:
        mode: 'default', 'quick', 'high_quality', or 'production'
    """
    configs = {
        'default': Config,
        'quick': QuickDemoConfig,
        'high_quality': HighQualityConfig,
        'production': ProductionConfig
    }
    
    config_class = configs.get(mode, Config)
    config_class.optimize_for_hardware()
    config_class.create_directories()
    
    return config_class


if __name__ == "__main__":
    # Test configuration
    config = get_config('default')
    config.print_config()
    
    device_info = config.get_device_info()
    print(f"\nDetected Hardware: {device_info}")
