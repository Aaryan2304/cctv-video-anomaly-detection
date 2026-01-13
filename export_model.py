"""
ONNX model export for cross-platform inference optimization.

Phase 4 feature: Export PyTorch autoencoder to ONNX format for:
- Cross-platform deployment (ONNX Runtime)
- Hardware-specific optimizations (TensorRT, OpenVINO)
- Reduced inference latency in production

Usage:
    python export_model.py --output model.onnx
    python export_model.py --output model.onnx --optimize
    python export_model.py --validate  # Test exported model
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from models.autoencoder import ConvolutionalAutoencoder
from settings import get_settings


def load_pytorch_model(model_path: Path, device: torch.device) -> Tuple[ConvolutionalAutoencoder, float]:
    """
    Load trained PyTorch model from checkpoint.
    
    Returns (model, threshold).
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    settings = get_settings()
    model = ConvolutionalAutoencoder(input_channels=1, latent_dim=settings.latent_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    threshold = checkpoint.get("threshold", 0.005069)
    
    return model, threshold


def export_to_onnx(
    model: ConvolutionalAutoencoder,
    output_path: Path,
    batch_size: int = 1,
    dynamic_batch: bool = True,
    opset_version: int = 17,
) -> None:
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: Trained ConvolutionalAutoencoder
        output_path: Path for exported ONNX file
        batch_size: Static batch size (ignored if dynamic_batch=True)
        dynamic_batch: Enable dynamic batch dimension
        opset_version: ONNX opset version (17 recommended for PyTorch 2.x)
    """
    # Create dummy input matching model's expected shape
    # Shape: (batch, channels=1, height=64, width=64)
    dummy_input = torch.randn(batch_size, 1, 64, 64)
    
    # Dynamic axes for variable batch size
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )
    
    print(f"Exported ONNX model to: {output_path}")
    print(f"  Opset version: {opset_version}")
    print(f"  Dynamic batch: {dynamic_batch}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")


def optimize_onnx(input_path: Path, output_path: Optional[Path] = None) -> Path:
    """
    Optimize ONNX model using onnxruntime graph optimizations.
    
    Applies:
    - Constant folding
    - Redundant node elimination
    - Operator fusion
    
    Returns path to optimized model.
    """
    try:
        import onnx
        from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
    except ImportError:
        print("Warning: onnx/onnxruntime not installed. Skipping optimization.")
        return input_path
    
    if output_path is None:
        output_path = input_path.with_suffix(".optimized.onnx")
    
    # Load and check model
    onnx_model = onnx.load(str(input_path))
    onnx.checker.check_model(onnx_model)
    
    # Apply graph optimizations via inference session save
    options = SessionOptions()
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    options.optimized_model_filepath = str(output_path)
    
    # Create session to trigger optimization
    _ = InferenceSession(str(input_path), options)
    
    print(f"Optimized ONNX model saved to: {output_path}")
    print(f"  Original size: {input_path.stat().st_size / 1024:.1f} KB")
    print(f"  Optimized size: {output_path.stat().st_size / 1024:.1f} KB")
    
    return output_path


def validate_onnx(onnx_path: Path, pytorch_model: ConvolutionalAutoencoder, atol: float = 1e-5) -> bool:
    """
    Validate ONNX model output matches PyTorch model.
    
    Args:
        onnx_path: Path to ONNX model
        pytorch_model: Original PyTorch model
        atol: Absolute tolerance for comparison
        
    Returns:
        True if outputs match within tolerance.
    """
    try:
        import onnx
        from onnxruntime import InferenceSession
    except ImportError:
        print("Error: onnx/onnxruntime required for validation. Install with:")
        print("  pip install onnx onnxruntime")
        return False
    
    # Check ONNX model
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    
    # Create inference session
    session = InferenceSession(str(onnx_path))
    
    # Test with random input
    test_input = np.random.randn(4, 1, 64, 64).astype(np.float32)
    
    # ONNX inference
    onnx_output = session.run(None, {"input": test_input})[0]
    
    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(torch.from_numpy(test_input)).numpy()
    
    # Compare
    max_diff = np.abs(onnx_output - pytorch_output).max()
    mean_diff = np.abs(onnx_output - pytorch_output).mean()
    
    passed = max_diff < atol
    
    print(f"Validation {'PASSED' if passed else 'FAILED'}:")
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")
    print(f"  Tolerance: {atol:.2e}")
    
    return passed


def benchmark_inference(onnx_path: Path, batch_size: int = 1, iterations: int = 100) -> dict:
    """
    Benchmark ONNX Runtime inference latency.
    
    Returns dict with latency statistics.
    """
    try:
        from onnxruntime import InferenceSession
    except ImportError:
        print("Error: onnxruntime required for benchmarking.")
        return {}
    
    import time
    
    session = InferenceSession(str(onnx_path))
    test_input = np.random.randn(batch_size, 1, 64, 64).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        _ = session.run(None, {"input": test_input})
    
    # Benchmark
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = session.run(None, {"input": test_input})
        latencies.append((time.perf_counter() - start) * 1000)
    
    latencies = np.array(latencies)
    
    stats = {
        "batch_size": batch_size,
        "iterations": iterations,
        "mean_ms": float(latencies.mean()),
        "std_ms": float(latencies.std()),
        "min_ms": float(latencies.min()),
        "max_ms": float(latencies.max()),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
    }
    
    print(f"Inference Benchmark (batch_size={batch_size}):")
    print(f"  Mean: {stats['mean_ms']:.2f} ms")
    print(f"  Std:  {stats['std_ms']:.2f} ms")
    print(f"  P95:  {stats['p95_ms']:.2f} ms")
    print(f"  P99:  {stats['p99_ms']:.2f} ms")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX format")
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/trained_model.pth",
        help="Path to PyTorch checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/model.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Apply graph optimizations after export",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate ONNX output matches PyTorch",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run inference benchmark",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for benchmarking",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    output_path = Path(args.output)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("ONNX MODEL EXPORT")
    print("=" * 60)
    
    # Load PyTorch model
    print(f"\nLoading PyTorch model from: {model_path}")
    device = torch.device("cpu")  # Export on CPU for compatibility
    model, threshold = load_pytorch_model(model_path, device)
    print(f"  Threshold: {threshold:.6f}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Export to ONNX
    print(f"\nExporting to ONNX...")
    export_to_onnx(model, output_path, opset_version=args.opset)
    
    # Optimize
    if args.optimize:
        print(f"\nOptimizing ONNX model...")
        output_path = optimize_onnx(output_path)
    
    # Validate
    if args.validate:
        print(f"\nValidating ONNX model...")
        if not validate_onnx(output_path, model):
            sys.exit(1)
    
    # Benchmark
    if args.benchmark:
        print(f"\nRunning inference benchmark...")
        benchmark_inference(output_path, batch_size=args.batch_size)
    
    print("\nExport complete.")
    
    # Save metadata
    metadata_path = output_path.with_suffix(".json")
    import json
    metadata = {
        "source_model": str(model_path),
        "onnx_path": str(output_path),
        "threshold": threshold,
        "opset_version": args.opset,
        "input_shape": [None, 1, 64, 64],
        "output_shape": [None, 1, 64, 64],
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
