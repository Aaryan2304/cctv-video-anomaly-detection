"""
GPU Management Utilities
========================

This module provides utilities for GPU memory management, monitoring,
and optimization for the anomaly detection system.
"""

import torch
import time
from typing import Dict, Optional, Tuple
import subprocess
import platform


class GPUManager:
    """
    GPU management utilities for memory monitoring and optimization.
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu_available = torch.cuda.is_available()
        
        if self.gpu_available:
            self.device_count = torch.cuda.device_count()
            self.current_device = torch.cuda.current_device()
        else:
            self.device_count = 0
            self.current_device = None
    
    def get_gpu_info(self) -> Dict:
        """Get comprehensive GPU information."""
        if not self.gpu_available:
            return {'gpu_available': False, 'device': 'CPU'}
        
        props = torch.cuda.get_device_properties(self.current_device)
        
        return {
            'gpu_available': True,
            'device_count': self.device_count,
            'current_device': self.current_device,
            'device_name': props.name,
            'total_memory_gb': props.total_memory / (1024**3),
            'major_version': props.major,
            'minor_version': props.minor,
            'multi_processor_count': props.multi_processor_count
        }
    
    def get_memory_usage(self) -> Dict:
        """Get current GPU memory usage."""
        if not self.gpu_available:
            return {'gpu_available': False}
        
        allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'utilization_percent': (allocated / total) * 100,
            'free_gb': total - reserved
        }
    
    def clear_memory(self):
        """Clear GPU memory cache."""
        if self.gpu_available:
            torch.cuda.empty_cache()
            print("GPU memory cache cleared")
    
    def optimize_memory_usage(self):
        """Apply memory optimization settings."""
        if not self.gpu_available:
            return
        
        # Enable memory fraction to prevent OOM
        torch.cuda.set_per_process_memory_fraction(0.8)
        
        # Enable memory efficient attention if available
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
        
        print("GPU memory optimization applied")
    
    def monitor_memory_during_training(self, check_interval: int = 10):
        """
        Memory monitoring context manager for training.
        
        Args:
            check_interval: Seconds between memory checks
        """
        return MemoryMonitor(self, check_interval)


class MemoryMonitor:
    """Context manager for monitoring GPU memory during training."""
    
    def __init__(self, gpu_manager: GPUManager, check_interval: int = 10):
        self.gpu_manager = gpu_manager
        self.check_interval = check_interval
        self.start_time = None
        self.peak_memory = 0
        
    def __enter__(self):
        self.start_time = time.time()
        if self.gpu_manager.gpu_available:
            torch.cuda.reset_peak_memory_stats()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.gpu_manager.gpu_available:
            peak_allocated = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"Peak GPU memory usage: {peak_allocated:.2f} GB")
    
    def check_memory(self):
        """Check current memory usage."""
        if not self.gpu_manager.gpu_available:
            return
        
        usage = self.gpu_manager.get_memory_usage()
        print(f"GPU Memory - Allocated: {usage['allocated_gb']:.2f}GB, "
              f"Reserved: {usage['reserved_gb']:.2f}GB, "
              f"Utilization: {usage['utilization_percent']:.1f}%")


def monitor_gpu_usage():
    """Simple function to print current GPU usage."""
    manager = GPUManager()
    
    if not manager.gpu_available:
        print("GPU not available - using CPU")
        return
    
    info = manager.get_gpu_info()
    usage = manager.get_memory_usage()
    
    print(f"GPU: {info['device_name']}")
    print(f"Memory: {usage['allocated_gb']:.2f}GB / {usage['total_gb']:.1f}GB "
          f"({usage['utilization_percent']:.1f}% utilized)")


def get_optimal_batch_size(model, input_shape: Tuple[int, ...], max_memory_gb: float = 4.0) -> int:
    """
    Estimate optimal batch size based on available GPU memory.
    
    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (excluding batch dimension)
        max_memory_gb: Maximum memory to use in GB
        
    Returns:
        Recommended batch size
    """
    if not torch.cuda.is_available():
        return 32  # Conservative CPU batch size
    
    device = torch.device('cuda')
    model = model.to(device)
    
    # Test with small batch to estimate memory per sample
    test_batch_size = 4
    test_input = torch.randn(test_batch_size, *input_shape).to(device)
    
    # Clear memory before test
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Forward pass to measure memory
    with torch.no_grad():
        _ = model(test_input)
    
    memory_per_sample = torch.cuda.max_memory_allocated() / test_batch_size / (1024**3)
    
    # Calculate optimal batch size with safety margin
    safety_factor = 0.8  # Use 80% of available memory
    available_memory = max_memory_gb * safety_factor
    optimal_batch_size = int(available_memory / memory_per_sample)
    
    # Ensure reasonable bounds
    optimal_batch_size = max(1, min(optimal_batch_size, 256))
    
    print(f"Estimated memory per sample: {memory_per_sample*1024:.1f}MB")
    print(f"Recommended batch size: {optimal_batch_size}")
    
    return optimal_batch_size


def check_cuda_installation():
    """Check CUDA installation and compatibility."""
    print("CUDA Installation Check:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name} ({props.total_memory/1024**3:.1f}GB)")
    else:
        print("CUDA not available - will use CPU")


def get_system_info() -> Dict:
    """Get comprehensive system information."""
    info = {
        'platform': platform.system(),
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info.update({
            'cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(),
            'gpu_count': torch.cuda.device_count(),
        })
        
        # Get GPU details
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info[f'gpu_{i}_name'] = props.name
            info[f'gpu_{i}_memory_gb'] = props.total_memory / (1024**3)
    
    return info


def benchmark_gpu_performance(model, input_shape: Tuple[int, ...], batch_size: int = 32, num_iterations: int = 100):
    """
    Benchmark GPU performance for the given model.
    
    Args:
        model: PyTorch model to benchmark
        input_shape: Input tensor shape (excluding batch dimension)
        batch_size: Batch size for benchmarking
        num_iterations: Number of iterations to run
    """
    if not torch.cuda.is_available():
        print("GPU not available for benchmarking")
        return
    
    device = torch.device('cuda')
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, *input_shape).to(device)
    
    # Warmup
    print("Warming up GPU...")
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    torch.cuda.synchronize()
    
    # Benchmark
    print(f"Benchmarking {num_iterations} iterations...")
    start_time = time.time()
    
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(dummy_input)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    # Calculate metrics
    total_time = end_time - start_time
    avg_time_per_batch = total_time / num_iterations
    samples_per_second = (batch_size * num_iterations) / total_time
    
    print(f"Benchmark Results:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average time per batch: {avg_time_per_batch*1000:.2f}ms")
    print(f"  Throughput: {samples_per_second:.1f} samples/second")
    print(f"  GPU utilization: {torch.cuda.utilization()}%")


if __name__ == "__main__":
    # Test GPU utilities
    print("Testing GPU utilities...")
    
    # Check CUDA installation
    check_cuda_installation()
    
    # Test GPU manager
    manager = GPUManager()
    info = manager.get_gpu_info()
    print(f"\nGPU Info: {info}")
    
    if manager.gpu_available:
        usage = manager.get_memory_usage()
        print(f"Memory Usage: {usage}")
        
        # Test memory monitoring
        monitor_gpu_usage()
    
    # System info
    sys_info = get_system_info()
    print(f"\nSystem Info: {sys_info}")
    
    print("\\nGPU utilities test completed!")
