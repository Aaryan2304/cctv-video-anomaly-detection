"""
Training and evaluation pipeline for video anomaly detection.
"""

import argparse
import os
import sys
import time
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader

# Import project modules
from config import Config
from models.autoencoder import ConvolutionalAutoencoder, LightweightAutoencoder
from models.detector import AnomalyDetector
from data.dataset import SyntheticVideoDataset, VideoDataset
from data.preprocessing import VideoPreprocessor
from evaluation.metrics import PerformanceEvaluator
import matplotlib.pyplot as plt


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Video Anomaly Detection System')
    
    parser.add_argument('--mode', choices=['synthetic', 'ucsd'], default='synthetic',
                       help='Dataset mode: synthetic (demo) or ucsd (real dataset)')
    parser.add_argument('--dataset_name', choices=['ped1', 'ped2'], default='ped2',
                       help='UCSD dataset variant: ped1 or ped2')
    parser.add_argument('--data_path', type=str, 
                       help='Path to UCSD dataset (required for ucsd mode)')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--model_type', choices=['standard', 'lightweight'], default='standard',
                       help='Model architecture type')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for results')
    parser.add_argument('--quick', action='store_true',
                       help='Quick demo with reduced parameters')
    
    return parser.parse_args()


def create_synthetic_dataset():
    """Create synthetic dataset for demonstration."""
    print("Creating synthetic dataset...")
    
    dataset = SyntheticVideoDataset(
        num_normal_frames=800,
        num_anomaly_frames=80,
        frame_size=(64, 64)
    )
    
    # Split into train (normal only) and test (mixed)
    labels = dataset.get_labels()
    normal_indices = np.where(labels == 0)[0]
    
    # Use 80% of normal data for training
    split_idx = int(0.8 * len(normal_indices))
    train_indices = normal_indices[:split_idx]
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = dataset  # Full dataset for testing
    test_labels = labels
    
    print(f"Created synthetic dataset: {len(train_dataset)} train, {len(test_dataset)} test")
    return train_dataset, test_dataset, test_labels


def create_ucsd_dataset(data_path):
    """Create UCSD dataset with ground truth labels."""
    print(f"Loading UCSD dataset from {data_path}...")
    
    # Check if path exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"UCSD dataset not found at: {data_path}")
    
    # Get video files
    train_path = os.path.join(data_path, 'Train')
    test_path = os.path.join(data_path, 'Test')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("UCSD dataset should have 'Train' and 'Test' directories")
    
    # Get training files (no ground truth needed for training)
    train_videos = []
    for root, dirs, files in os.walk(train_path):
        for file in files:
            if file.endswith(('.avi', '.mp4', '.mov', '.tif', '.tiff', '.jpg', '.jpeg', '.png')):
                train_videos.append(os.path.join(root, file))
    
    # Get test files AND their corresponding ground truth
    test_videos = []
    test_labels = []
    
    # Process each Test folder
    test_folders = [d for d in os.listdir(test_path) if d.startswith('Test') and not d.endswith('_gt')]
    test_folders.sort()  # Ensure consistent order
    
    print(f"Found {len(test_folders)} test sequences")
    
    for test_folder in test_folders:
        test_video_path = os.path.join(test_path, test_folder)
        test_gt_path = os.path.join(test_path, test_folder + '_gt')
        
        if not os.path.exists(test_gt_path):
            print(f"Warning: No ground truth found for {test_folder}")
            continue
        
        # Get video files from this test folder
        video_files = []
        gt_files = []
        
        for file in sorted(os.listdir(test_video_path)):
            if file.endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png')):
                video_file = os.path.join(test_video_path, file)
                # Corresponding ground truth file (change extension to .bmp)
                gt_filename = os.path.splitext(file)[0] + '.bmp'
                gt_file = os.path.join(test_gt_path, gt_filename)
                
                if os.path.exists(gt_file):
                    video_files.append(video_file)
                    gt_files.append(gt_file)
        
        print(f"  {test_folder}: {len(video_files)} frames with ground truth")
        test_videos.extend(video_files)
        
        # Load ground truth labels for this sequence
        for gt_file in gt_files:
            # Read ground truth mask
            gt_mask = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
            if gt_mask is None:
                print(f"Warning: Could not read ground truth {gt_file}")
                test_labels.append(0)  # Default to normal
                continue
            
            # Convert mask to frame-level label
            # If any pixel is anomalous (white), the entire frame is anomalous
            has_anomaly = np.any(gt_mask > 127)  # Threshold for white pixels
            test_labels.append(1 if has_anomaly else 0)
    
    if not train_videos or not test_videos:
        raise FileNotFoundError("No video/image files found in UCSD dataset directories")
    
    print(f"Loaded {len(test_videos)} test frames with labels")
    print(f"  Normal frames: {test_labels.count(0)}")
    print(f"  Anomalous frames: {test_labels.count(1)}")
    print(f"  Anomaly rate: {test_labels.count(1)/len(test_labels):.1%}")
    
    # Create datasets
    from data.preprocessing import VideoPreprocessor
    preprocessor = VideoPreprocessor(
        target_size=(64, 64),
        quality_threshold=0.001  # More lenient threshold for image files
    )
    
    train_dataset = VideoDataset(
        train_videos, 
        frame_size=(64, 64),
        max_frames_per_video=200,
        preprocessor=preprocessor
    )
    
    test_dataset = VideoDataset(
        test_videos,
        frame_size=(64, 64), 
        max_frames_per_video=200,
        preprocessor=preprocessor
    )
    
    # Convert labels to numpy array
    test_labels = np.array(test_labels, dtype=np.int64)
    
    print(f"Created UCSD dataset: {len(train_dataset)} train, {len(test_dataset)} test")
    return train_dataset, test_dataset, test_labels


def create_model(model_type, device):
    """Create autoencoder model."""
    print(f"Creating {model_type} model...")
    
    if model_type == 'standard':
        model = ConvolutionalAutoencoder(input_channels=1, latent_dim=256)
    else:
        model = LightweightAutoencoder(input_channels=1, latent_dim=128)
    
    model = model.to(device)
    info = model.get_model_info()
    
    print(f"Model created: {info['total_parameters']:,} parameters")
    return model


def train_model(detector, train_dataset, epochs, batch_size):
    """Train the anomaly detection model."""
    print(f"\nTraining model for {epochs} epochs...")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    start_time = time.time()
    
    # Train the detector
    training_stats = detector.train(
        train_loader=train_loader,
        num_epochs=epochs,
        learning_rate=0.001
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.1f} seconds")
    
    return training_stats


def evaluate_model(detector, train_dataset, test_dataset, test_labels, batch_size):
    """Evaluate the trained model."""
    print(f"\nEvaluating model...")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Establish threshold
    print("Establishing detection threshold...")
    threshold_stats = detector.establish_threshold(train_loader, threshold_factor=2.0)
    
    # Detect anomalies
    print("Detecting anomalies...")
    reconstruction_errors, anomaly_flags = detector.detect_anomalies(test_loader)
    
    # Evaluate if labels available
    results = {}
    if test_labels is not None:
        evaluator = PerformanceEvaluator()
        results = evaluator.evaluate_detection(reconstruction_errors, test_labels)
        
        print(f"\nPerformance Results:")
        print(f"  AUC Score: {results['auc_score']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  F1-Score: {results['f1_score']:.4f}")
    else:
        print(f"Anomaly detection completed (no ground truth for evaluation)")
    
    anomaly_count = np.sum(anomaly_flags)
    total_count = len(anomaly_flags)
    print(f"  Detected {anomaly_count} anomalies out of {total_count} frames ({anomaly_count/total_count:.1%})")
    
    return results, reconstruction_errors, anomaly_flags


def create_validation_samples(reconstruction_errors, anomaly_flags, output_dir, num_samples=10):
    """Create validation samples showing normal vs anomalous frames."""
    print(f"\nCreating validation samples...")
    
    # Get indices of normal and anomalous frames
    normal_indices = np.where(anomaly_flags == False)[0]
    anomaly_indices = np.where(anomaly_flags == True)[0]
    
    # Create validation report
    validation_report = []
    
    # Sample normal frames (lowest errors)
    normal_errors = reconstruction_errors[normal_indices]
    normal_sorted = normal_indices[np.argsort(normal_errors)]
    
    # Sample anomalous frames (highest errors)
    anomaly_errors = reconstruction_errors[anomaly_indices]
    anomaly_sorted = anomaly_indices[np.argsort(anomaly_errors)[::-1]]
    
    validation_report.append("VALIDATION SAMPLE ANALYSIS")
    validation_report.append("=" * 30)
    validation_report.append("")
    
    validation_report.append("TOP NORMAL FRAMES (Lowest Reconstruction Errors):")
    validation_report.append("-" * 50)
    for i, idx in enumerate(normal_sorted[:num_samples]):
        validation_report.append(f"Frame {idx}: Error = {reconstruction_errors[idx]:.6f}")
    
    validation_report.append("")
    validation_report.append("TOP ANOMALOUS FRAMES (Highest Reconstruction Errors):")
    validation_report.append("-" * 55)
    for i, idx in enumerate(anomaly_sorted[:num_samples]):
        validation_report.append(f"Frame {idx}: Error = {reconstruction_errors[idx]:.6f}")
    
    validation_report.append("")
    validation_report.append("INTERPRETATION GUIDE:")
    validation_report.append("-" * 20)
    validation_report.append("• Lower errors = More 'normal' according to the model")
    validation_report.append("• Higher errors = More 'anomalous' according to the model")
    validation_report.append("• Review frames manually to validate model decisions")
    
    # Save validation report
    with open(os.path.join(output_dir, 'validation_samples.txt'), 'w') as f:
        f.write('\n'.join(validation_report))
    
    print(f"Validation samples saved to validation_samples.txt")
    return validation_report


def create_threshold_analysis(detector, reconstruction_errors, output_dir):
    """Analyze threshold sensitivity."""
    print(f"Creating threshold sensitivity analysis...")
    
    # Test different threshold multipliers
    threshold_factors = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    analysis_results = []
    
    mean_error = detector.threshold_stats['mean_error']
    std_error = detector.threshold_stats['std_error']
    
    analysis_results.append("THRESHOLD SENSITIVITY ANALYSIS")
    analysis_results.append("=" * 35)
    analysis_results.append("")
    analysis_results.append("Factor | Threshold | Anomaly Rate | Description")
    analysis_results.append("-" * 50)
    
    for factor in threshold_factors:
        test_threshold = mean_error + factor * std_error
        test_anomalies = np.sum(reconstruction_errors > test_threshold)
        anomaly_rate = test_anomalies / len(reconstruction_errors)
        
        if anomaly_rate > 0.8:
            desc = "Very Sensitive"
        elif anomaly_rate > 0.5:
            desc = "Sensitive"
        elif anomaly_rate > 0.2:
            desc = "Moderate"
        elif anomaly_rate > 0.05:
            desc = "Conservative"
        else:
            desc = "Very Conservative"
        
        analysis_results.append(f"{factor:4.1f}   | {test_threshold:.6f} | {anomaly_rate:8.2%}   | {desc}")
    
    analysis_results.append("")
    analysis_results.append("RECOMMENDATIONS:")
    analysis_results.append("-" * 15)
    analysis_results.append("• Current factor: 2.5")
    analysis_results.append("• For fewer false alarms: Increase factor (3.0-4.0)")
    analysis_results.append("• For higher sensitivity: Decrease factor (1.5-2.0)")
    analysis_results.append("• Ideal range for surveillance: 10-30% anomaly rate")
    
    # Save analysis
    with open(os.path.join(output_dir, 'threshold_analysis.txt'), 'w') as f:
        f.write('\n'.join(analysis_results))
    
    print(f"Threshold analysis saved to threshold_analysis.txt")
    return analysis_results


def create_visualizations(detector, results, reconstruction_errors, test_labels, output_dir):
    """Create and save visualizations."""
    print(f"\nCreating visualizations...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Training loss plot
    plt.figure(figsize=(10, 6))
    
    if hasattr(detector, 'training_history') and detector.training_history['losses']:
        plt.subplot(1, 2, 1)
        plt.plot(detector.training_history['losses'])
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.grid(True)
    
    # Error distribution
    plt.subplot(1, 2, 2)
    if hasattr(detector, 'normal_errors') and len(detector.normal_errors) > 0:
        plt.hist(detector.normal_errors, bins=50, alpha=0.7, label='Normal', density=True)
        plt.axvline(detector.threshold, color='red', linestyle='--', label=f'Threshold: {detector.threshold:.4f}')
    
    if test_labels is not None:
        normal_errors = reconstruction_errors[test_labels == 0]
        anomaly_errors = reconstruction_errors[test_labels == 1]
        
        plt.hist(normal_errors, bins=50, alpha=0.7, label='Normal (Test)', density=True)
        plt.hist(anomaly_errors, bins=50, alpha=0.7, label='Anomaly (Test)', density=True)
    
    plt.title('Reconstruction Error Distribution')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_analysis.png'), dpi=150, bbox_inches='tight')
    print(f"Saved training analysis plot")
    
    # ROC curve (if labels available)
    if test_labels is not None and 'fpr' in results:
        plt.figure(figsize=(8, 6))
        plt.plot(results['fpr'], results['tpr'], linewidth=2, 
                label=f'ROC Curve (AUC = {results["auc_score"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Anomaly Detection Performance')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=150, bbox_inches='tight')
        print(f"Saved ROC curve")
    
    plt.close('all')


def save_results(results, reconstruction_errors, anomaly_flags, output_dir, detector=None, training_stats=None):
    """Save results to files."""
    print(f"Saving results to {output_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save numerical results
    np.save(os.path.join(output_dir, 'reconstruction_errors.npy'), reconstruction_errors)
    np.save(os.path.join(output_dir, 'anomaly_flags.npy'), anomaly_flags)
    
    # Create detailed analysis file
    with open(os.path.join(output_dir, 'analysis_report.txt'), 'w') as f:
        f.write("ANOMALY DETECTION ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Dataset info
        f.write("DATASET INFORMATION\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total frames analyzed: {len(reconstruction_errors)}\n")
        f.write(f"Anomalies detected: {np.sum(anomaly_flags)}\n")
        f.write(f"Anomaly rate: {np.sum(anomaly_flags)/len(anomaly_flags):.2%}\n\n")
        
        # Threshold info
        if detector and hasattr(detector, 'threshold_stats'):
            f.write("THRESHOLD INFORMATION\n")
            f.write("-" * 20 + "\n")
            f.write(f"Detection threshold: {detector.threshold:.6f}\n")
            f.write(f"Mean normal error: {detector.threshold_stats['mean_error']:.6f}\n")
            f.write(f"Std normal error: {detector.threshold_stats['std_error']:.6f}\n")
            f.write(f"False positive rate: {detector.threshold_stats['false_positive_rate']:.4f}\n\n")
        
        # Reconstruction error statistics
        f.write("RECONSTRUCTION ERROR STATISTICS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Mean error: {np.mean(reconstruction_errors):.6f}\n")
        f.write(f"Std error: {np.std(reconstruction_errors):.6f}\n")
        f.write(f"Min error: {np.min(reconstruction_errors):.6f}\n")
        f.write(f"Max error: {np.max(reconstruction_errors):.6f}\n")
        f.write(f"Median error: {np.median(reconstruction_errors):.6f}\n\n")
        
        # Percentile analysis
        f.write("ERROR PERCENTILES\n")
        f.write("-" * 15 + "\n")
        for p in [25, 50, 75, 90, 95, 99]:
            percentile_val = np.percentile(reconstruction_errors, p)
            f.write(f"{p}th percentile: {percentile_val:.6f}\n")
        f.write("\n")
        
        # Training info
        if training_stats:
            f.write("TRAINING INFORMATION\n")
            f.write("-" * 20 + "\n")
            f.write(f"Training time: {training_stats.get('total_time', 0):.1f} seconds\n")
            f.write(f"Epochs completed: {training_stats.get('epochs_completed', 0)}\n")
            f.write(f"Final loss: {training_stats.get('best_loss', 0):.6f}\n")
            f.write(f"Early stopped: {training_stats.get('early_stopped', False)}\n\n")
        
        # Performance metrics (if available)
        if results and 'auc_score' in results:
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 18 + "\n")
            f.write(f"AUC Score: {results['auc_score']:.4f}\n")
            f.write(f"Precision: {results['precision']:.4f}\n")
            f.write(f"Recall: {results['recall']:.4f}\n")
            f.write(f"F1-Score: {results['f1_score']:.4f}\n\n")
        else:
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 18 + "\n")
            f.write("No ground truth labels available for validation.\n")
            f.write("Consider using synthetic data for performance evaluation.\n\n")
    
    print(f"Results saved to {output_dir}")
    print(f"Detailed analysis saved to analysis_report.txt")


def main():
    """Main execution function."""
    args = parse_args()
    
    print("=" * 60)
    print("VIDEO ANOMALY DETECTION SYSTEM")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Adjust parameters for quick demo
    if args.quick:
        args.epochs = 15
        args.batch_size = 32
        print("Quick demo mode: reduced parameters for faster execution")
    
    try:
        # Create dataset
        if args.mode == 'synthetic':
            train_dataset, test_dataset, test_labels = create_synthetic_dataset()
        else:
            if not args.data_path:
                raise ValueError("--data_path required for UCSD mode")
            train_dataset, test_dataset, test_labels = create_ucsd_dataset(args.data_path)
        
        # Create model and detector
        model = create_model(args.model_type, device)
        detector = AnomalyDetector(model, device)
        
        # Train model
        training_stats = train_model(detector, train_dataset, args.epochs, args.batch_size)
        
        # Save the trained model
        model_save_path = os.path.join(args.output_dir, 'trained_model.pth')
        torch.save({
            'model_state_dict': detector.model.state_dict(),
            'threshold': detector.threshold if hasattr(detector, 'threshold') else None,
            'training_stats': training_stats,
            'model_info': detector.model.get_model_info()
        }, model_save_path)
        print(f"Model saved to: {model_save_path}")
        
        # Evaluate model
        results, reconstruction_errors, anomaly_flags = evaluate_model(
            detector, train_dataset, test_dataset, test_labels, args.batch_size
        )
        
        # Create visualizations
        create_visualizations(detector, results, reconstruction_errors, test_labels, args.output_dir)
        
        # Create validation analysis
        create_validation_samples(reconstruction_errors, anomaly_flags, args.output_dir)
        create_threshold_analysis(detector, reconstruction_errors, args.output_dir)
        
        # Save results
        save_results(results, reconstruction_errors, anomaly_flags, args.output_dir, detector, training_stats)
        
        # Final summary
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        if results and 'auc_score' in results:
            auc = results['auc_score']
            print(f"🎯 Final AUC Score: {auc:.4f}")
            
            if auc > 0.9:
                print("🏆 Excellent performance!")
            elif auc > 0.8:
                print("✅ Good performance!")
            elif auc > 0.7:
                print("📈 Fair performance")
            else:
                print("⚠️ Performance could be improved")
        
        print(f"📁 Results saved to: {args.output_dir}")
        print(f"📊 Check visualizations: {args.output_dir}/training_analysis.png")
        
        if args.mode == 'synthetic':
            print(f"\n💡 Next step: Try real dataset with --mode ucsd --data_path /path/to/ucsd")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
