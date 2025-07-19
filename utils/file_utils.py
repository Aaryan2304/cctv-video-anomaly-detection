"""
File Management Utilities
=========================

This module provides utilities for file operations, directory management,
and result saving for the anomaly detection system.
"""

import os
import json
import pickle
import time
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
import torch


def setup_experiment_directory(base_dir: str, experiment_name: Optional[str] = None) -> str:
    """
    Create a new experiment directory with timestamp.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Name for experiment (auto-generated if None)
        
    Returns:
        Path to created experiment directory
    """
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"experiment_{timestamp}"
    
    experiment_dir = os.path.join(base_dir, experiment_name)
    
    # Create directory structure
    subdirs = ['models', 'plots', 'logs', 'data']
    for subdir in subdirs:
        os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)
    
    print(f"✓ Experiment directory created: {experiment_dir}")
    return experiment_dir


def save_results(results: Dict[str, Any], filepath: str):
    """
    Save results dictionary to JSON file.
    
    Args:
        results: Dictionary containing results
        filepath: Path to save file
    """
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            json_results[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            json_results[key] = value.item()
        else:
            json_results[key] = value
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"✓ Results saved to {filepath}")


def load_results(filepath: str) -> Dict[str, Any]:
    """
    Load results from JSON file.
    
    Args:
        filepath: Path to results file
        
    Returns:
        Dictionary containing loaded results
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    return results


def save_model_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)
    print(f"✓ Model checkpoint saved to {filepath}")


def load_model_checkpoint(filepath, model, optimizer=None):
"""
File Management Utilities
=========================

This module provides utilities for file operations, directory management,
and result saving for the anomaly detection system.
"""

import os
import json
import pickle
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np
import torch
import shutil
import glob


def setup_experiment_directory(base_dir: str, experiment_name: Optional[str] = None) -> str:
    """
    Create a new experiment directory with timestamp.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Name for experiment (auto-generated if None)
        
    Returns:
        Path to created experiment directory
    """
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"experiment_{timestamp}"
    
    experiment_dir = os.path.join(base_dir, experiment_name)
    
    # Create directory structure
    subdirs = ['models', 'plots', 'logs', 'data', 'cache']
    for subdir in subdirs:
        os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)
    
    print(f"✓ Experiment directory created: {experiment_dir}")
    return experiment_dir


def save_results(results: Dict[str, Any], filepath: str):
    """
    Save results dictionary to JSON file.
    
    Args:
        results: Dictionary containing results
        filepath: Path to save file
    """
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            json_results[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            json_results[key] = value.item()
        elif isinstance(value, dict):
            # Recursively handle nested dictionaries
            json_results[key] = _convert_dict_for_json(value)
        else:
            json_results[key] = value
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"✓ Results saved to {filepath}")


def _convert_dict_for_json(d: Dict) -> Dict:
    """Recursively convert dictionary for JSON serialization."""
    converted = {}
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            converted[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            converted[key] = value.item()
        elif isinstance(value, dict):
            converted[key] = _convert_dict_for_json(value)
        else:
            converted[key] = value
    return converted


def load_results(filepath: str) -> Dict[str, Any]:
    """
    Load results from JSON file.
    
    Args:
        filepath: Path to results file
        
    Returns:
        Dictionary containing loaded results
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    return results


def save_model_checkpoint(model, optimizer, epoch, loss, filepath, **kwargs):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
        **kwargs: Additional data to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat(),
        **kwargs
    }
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)
    print(f"✓ Model checkpoint saved to {filepath}")


def load_model_checkpoint(filepath, model, optimizer=None, device='cpu'):
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: PyTorch model to load state into
        optimizer: Optimizer to load state into (optional)
        device: Device to map tensors to
        
    Returns:
        Dictionary with checkpoint information
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"✓ Model checkpoint loaded from {filepath}")
    return checkpoint


def save_numpy_arrays(arrays_dict: Dict[str, np.ndarray], directory: str):
    """
    Save multiple numpy arrays to directory.
    
    Args:
        arrays_dict: Dictionary mapping names to numpy arrays
        directory: Directory to save arrays
    """
    os.makedirs(directory, exist_ok=True)
    
    for name, array in arrays_dict.items():
        filepath = os.path.join(directory, f"{name}.npy")
        np.save(filepath, array)
    
    print(f"✓ Saved {len(arrays_dict)} arrays to {directory}")


def load_numpy_arrays(directory: str) -> Dict[str, np.ndarray]:
    """
    Load all numpy arrays from directory.
    
    Args:
        directory: Directory containing .npy files
        
    Returns:
        Dictionary mapping names to loaded arrays
    """
    arrays_dict = {}
    
    for filepath in glob.glob(os.path.join(directory, "*.npy")):
        name = os.path.splitext(os.path.basename(filepath))[0]
        arrays_dict[name] = np.load(filepath)
    
    print(f"✓ Loaded {len(arrays_dict)} arrays from {directory}")
    return arrays_dict


def get_file_size_mb(filepath: str) -> float:
    """Get file size in MB."""
    if os.path.exists(filepath):
        return os.path.getsize(filepath) / (1024 * 1024)
    return 0.0


def get_directory_size_mb(directory: str) -> float:
    """Get total size of directory in MB."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)


def cleanup_old_files(directory: str, max_files: int = 10, pattern: str = "*.pth"):
    """
    Clean up old files keeping only the most recent ones.
    
    Args:
        directory: Directory to clean
        max_files: Maximum number of files to keep
        pattern: File pattern to match
    """
    if not os.path.exists(directory):
        return
    
    files = glob.glob(os.path.join(directory, pattern))
    if len(files) <= max_files:
        return
    
    # Sort by modification time (newest first)
    files.sort(key=os.path.getmtime, reverse=True)
    
    # Remove oldest files
    for filepath in files[max_files:]:
        try:
            os.remove(filepath)
            print(f"Removed old file: {os.path.basename(filepath)}")
        except OSError as e:
            print(f"Error removing {filepath}: {e}")


def backup_directory(source_dir: str, backup_dir: str):
    """
    Create backup of directory.
    
    Args:
        source_dir: Source directory to backup
        backup_dir: Destination backup directory
    """
    if os.path.exists(source_dir):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{timestamp}"
        backup_path = os.path.join(backup_dir, backup_name)
        
        shutil.copytree(source_dir, backup_path)
        print(f"✓ Backup created: {backup_path}")
    else:
        print(f"Source directory does not exist: {source_dir}")


def create_experiment_summary(experiment_dir: str, results: Dict[str, Any]):
    """
    Create a text summary of the experiment.
    
    Args:
        experiment_dir: Experiment directory
        results: Results dictionary
    """
    summary_path = os.path.join(experiment_dir, "experiment_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("ANOMALY DETECTION EXPERIMENT SUMMARY\n")
        f.write("=" * 40 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Experiment Directory: {experiment_dir}\n\n")
        
        # Performance metrics
        if 'evaluation_results' in results and results['evaluation_results']:
            eval_results = results['evaluation_results']
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"AUC Score: {eval_results.get('auc_score', 'N/A'):.4f}\n")
            f.write(f"Precision: {eval_results.get('precision', 'N/A'):.4f}\n")
            f.write(f"Recall: {eval_results.get('recall', 'N/A'):.4f}\n")
            f.write(f"F1-Score: {eval_results.get('f1_score', 'N/A'):.4f}\n")
            f.write(f"Accuracy: {eval_results.get('accuracy', 'N/A'):.4f}\n\n")
        
        # Training info
        if 'training_stats' in results and results['training_stats']:
            train_stats = results['training_stats']
            f.write("TRAINING INFORMATION\n")
            f.write("-" * 20 + "\n")
            f.write(f"Training Time: {train_stats.get('total_time', 'N/A'):.1f} seconds\n")
            f.write(f"Epochs Completed: {train_stats.get('epochs_completed', 'N/A')}\n")
            f.write(f"Best Loss: {train_stats.get('best_loss', 'N/A'):.6f}\n")
            f.write(f"Early Stopped: {train_stats.get('early_stopped', 'N/A')}\n\n")
        
        # Dataset info
        if 'dataset_info' in results:
            dataset_info = results['dataset_info']
            f.write("DATASET INFORMATION\n")
            f.write("-" * 20 + "\n")
            for key, value in dataset_info.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
        
        # Experiment config
        if 'experiment_config' in results:
            config = results['experiment_config']
            f.write("EXPERIMENT CONFIGURATION\n")
            f.write("-" * 25 + "\n")
            for key, value in config.items():
                f.write(f"{key}: {value}\n")
        
        # File information
        f.write(f"\nFILE INFORMATION\n")
        f.write("-" * 15 + "\n")
        total_size = get_directory_size_mb(experiment_dir)
        f.write(f"Total Size: {total_size:.2f} MB\n")
        
        # List key files
        key_files = ['training_analysis.png', 'roc_curve.png', 'results.json', 'model_best.pth']
        for filename in key_files:
            filepath = os.path.join(experiment_dir, filename)
            if os.path.exists(filepath):
                size = get_file_size_mb(filepath)
                f.write(f"{filename}: {size:.2f} MB\n")
    
    print(f"✓ Experiment summary saved to {summary_path}")


def list_experiments(base_dir: str) -> List[Dict]:
    """
    List all experiments in base directory.
    
    Args:
        base_dir: Base directory containing experiments
        
    Returns:
        List of experiment information dictionaries
    """
    experiments = []
    
    if not os.path.exists(base_dir):
        return experiments
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            # Check if it looks like an experiment directory
            results_file = os.path.join(item_path, 'results.json')
            summary_file = os.path.join(item_path, 'experiment_summary.txt')
            
            if os.path.exists(results_file) or os.path.exists(summary_file):
                exp_info = {
                    'name': item,
                    'path': item_path,
                    'created': datetime.fromtimestamp(os.path.getctime(item_path)),
                    'modified': datetime.fromtimestamp(os.path.getmtime(item_path)),
                    'size_mb': get_directory_size_mb(item_path),
                    'has_results': os.path.exists(results_file),
                    'has_summary': os.path.exists(summary_file)
                }
                experiments.append(exp_info)
    
    # Sort by modification time (newest first)
    experiments.sort(key=lambda x: x['modified'], reverse=True)
    
    return experiments


def clean_experiment_directory(experiment_dir: str, keep_essentials: bool = True):
    """
    Clean up experiment directory, removing temporary files.
    
    Args:
        experiment_dir: Experiment directory to clean
        keep_essentials: Whether to keep essential files (results, plots, models)
    """
    if not os.path.exists(experiment_dir):
        return
    
    # Files to always remove
    temp_patterns = ['*.tmp', '*.temp', '*.log', '*~', '.DS_Store']
    
    # Additional files to remove if not keeping essentials
    if not keep_essentials:
        temp_patterns.extend(['*.npy', '*.pkl', 'cache/*'])
    
    removed_count = 0
    for pattern in temp_patterns:
        files = glob.glob(os.path.join(experiment_dir, '**', pattern), recursive=True)
        for filepath in files:
            try:
                os.remove(filepath)
                removed_count += 1
            except OSError:
                pass
    
    print(f"✓ Cleaned experiment directory: removed {removed_count} temporary files")


def export_experiment_results(experiment_dir: str, export_format: str = 'zip'):
    """
    Export experiment results in specified format.
    
    Args:
        experiment_dir: Experiment directory to export
        export_format: Export format ('zip', 'tar')
    """
    if not os.path.exists(experiment_dir):
        print(f"Experiment directory does not exist: {experiment_dir}")
        return
    
    experiment_name = os.path.basename(experiment_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if export_format == 'zip':
        import zipfile
        export_path = f"{experiment_name}_{timestamp}.zip"
        
        with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(experiment_dir):
                for file in files:
                    filepath = os.path.join(root, file)
                    arcname = os.path.relpath(filepath, experiment_dir)
                    zipf.write(filepath, arcname)
        
    elif export_format == 'tar':
        import tarfile
        export_path = f"{experiment_name}_{timestamp}.tar.gz"
        
        with tarfile.open(export_path, 'w:gz') as tarf:
            tarf.add(experiment_dir, arcname=experiment_name)
    
    else:
        print(f"Unsupported export format: {export_format}")
        return
    
    print(f"✓ Experiment exported to: {export_path}")


def validate_experiment_directory(experiment_dir: str) -> Dict[str, bool]:
    """
    Validate experiment directory structure and contents.
    
    Args:
        experiment_dir: Experiment directory to validate
        
    Returns:
        Dictionary with validation results
    """
    validation = {
        'directory_exists': os.path.exists(experiment_dir),
        'has_results': False,
        'has_plots': False,
        'has_models': False,
        'has_logs': False
    }
    
    if not validation['directory_exists']:
        return validation
    
    # Check for key files
    validation['has_results'] = os.path.exists(os.path.join(experiment_dir, 'results.json'))
    validation['has_plots'] = len(glob.glob(os.path.join(experiment_dir, '*.png'))) > 0
    validation['has_models'] = len(glob.glob(os.path.join(experiment_dir, '*.pth'))) > 0
    validation['has_logs'] = os.path.exists(os.path.join(experiment_dir, 'logs'))
    
    return validation


if __name__ == "__main__":
    # Test file utilities
    print("Testing file utilities...")
    
    # Test directory creation
    test_dir = setup_experiment_directory("test_outputs", "test_experiment")
    
    # Test results saving
    test_results = {
        'auc_score': 0.85,
        'precision': 0.78,
        'test_array': np.array([1, 2, 3, 4, 5]),
        'nested_dict': {
            'metric1': 0.92,
            'metric2': np.array([0.1, 0.2, 0.3])
        }
    }
    
    results_path = os.path.join(test_dir, "test_results.json")
    save_results(test_results, results_path)
    
    # Test results loading
    loaded_results = load_results(results_path)
    print(f"Loaded results keys: {list(loaded_results.keys())}")
    
    # Test numpy arrays saving/loading
    arrays_dict = {
        'errors': np.random.rand(100),
        'labels': np.random.randint(0, 2, 100)
    }
    save_numpy_arrays(arrays_dict, os.path.join(test_dir, 'arrays'))
    loaded_arrays = load_numpy_arrays(os.path.join(test_dir, 'arrays'))
    print(f"Loaded arrays: {list(loaded_arrays.keys())}")
    
    # Test experiment summary
    create_experiment_summary(test_dir, {'evaluation_results': test_results})
    
    # Test directory validation
    validation = validate_experiment_directory(test_dir)
    print(f"Validation results: {validation}")
    
    # Test directory listing
    experiments = list_experiments("test_outputs")
    print(f"Found {len(experiments)} experiments")
    
    print("✓ File utilities test completed!")

