"""
Results Visualization for Anomaly Detection
===========================================

This module creates comprehensive visualizations for understanding and
presenting anomaly detection results. It generates publication-quality
plots for training analysis, performance evaluation, and detection examples.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import os
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")


class ResultsVisualizer:
    """
    Comprehensive visualization suite for anomaly detection results.
    
    This class creates various plots to help understand:
    - Training progress and convergence
    - Performance metrics and ROC curves
    - Reconstruction examples and error patterns
    - Detection results and threshold analysis
    """
    
    def __init__(self, output_dir: str = "outputs", figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save plots
            figsize: Default figure size for plots
        """
        self.output_dir = output_dir
        self.figsize = figsize
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot settings
        self.dpi = 150
        self.save_format = 'png'
    
    def plot_training_progress(self, training_history: Dict, save_name: str = "training_progress.png"):
        """
        Plot training progress including loss curves and learning rate.
        
        Args:
            training_history: Dictionary with training metrics
            save_name: Name for saved plot
        """
        if not training_history or 'losses' not in training_history:
            print("No training history available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Training Progress Analysis', fontsize=16, fontweight='bold')
        
        losses = training_history['losses']
        epochs = range(1, len(losses) + 1)
        
        # Training loss
        axes[0, 0].plot(epochs, losses, 'b-', linewidth=2, label='Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('MSE Loss')
        axes[0, 0].set_title('Training Loss Over Time')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Loss improvement (derivative)
        if len(losses) > 1:
            loss_diff = np.diff(losses)
            axes[0, 1].plot(epochs[1:], loss_diff, 'r-', linewidth=2, label='Loss Change')
            axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss Change')
            axes[0, 1].set_title('Loss Improvement Rate')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
        
        # Learning rate (if available)
        if 'learning_rates' in training_history:
            lr_history = training_history['learning_rates']
            axes[1, 0].plot(epochs, lr_history, 'g-', linewidth=2, label='Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
        
        # Convergence analysis
        if len(losses) > 10:
            # Moving average for smoother curve
            window = min(10, len(losses) // 4)
            moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
            moving_epochs = epochs[window-1:]
            
            axes[1, 1].plot(epochs, losses, 'b-', alpha=0.6, label='Raw Loss')
            axes[1, 1].plot(moving_epochs, moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window})')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('MSE Loss')
            axes[1, 1].set_title('Loss Convergence Analysis')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', format=self.save_format)
        plt.show()
        
        print(f"Training progress plot saved to {save_path}")
    
    def plot_error_distributions(
        self, 
        normal_errors: np.ndarray, 
        anomaly_errors: np.ndarray = None,
        threshold: float = None,
        save_name: str = "error_distributions.png"
    ):
        """
        Plot reconstruction error distributions for normal and anomalous data.
        
        Args:
            normal_errors: Reconstruction errors for normal data
            anomaly_errors: Reconstruction errors for anomalous data (optional)
            threshold: Detection threshold to display
            save_name: Name for saved plot
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        fig.suptitle('Reconstruction Error Analysis', fontsize=16, fontweight='bold')
        
        # Error distribution
        axes[0].hist(normal_errors, bins=50, alpha=0.7, density=True, 
                    color='blue', label='Normal Data', edgecolor='black', linewidth=0.5)
        
        if anomaly_errors is not None:
            axes[0].hist(anomaly_errors, bins=50, alpha=0.7, density=True,
                        color='red', label='Anomalous Data', edgecolor='black', linewidth=0.5)
        
        if threshold is not None:
            axes[0].axvline(threshold, color='green', linestyle='--', linewidth=2,
                           label=f'Threshold: {threshold:.4f}')
        
        axes[0].set_xlabel('Reconstruction Error')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Error Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot comparison
        if anomaly_errors is not None:
            data_to_plot = [normal_errors, anomaly_errors]
            labels = ['Normal', 'Anomaly']
            
            box_plot = axes[1].boxplot(data_to_plot, labels=labels, patch_artist=True)
            box_plot['boxes'][0].set_facecolor('lightblue')
            if len(box_plot['boxes']) > 1:
                box_plot['boxes'][1].set_facecolor('lightcoral')
            
            if threshold is not None:
                axes[1].axhline(threshold, color='green', linestyle='--', linewidth=2,
                               label=f'Threshold: {threshold:.4f}')
                axes[1].legend()
        else:
            # Just plot normal errors statistics
            stats_text = f"""Normal Data Statistics:
Mean: {np.mean(normal_errors):.4f}
Std: {np.std(normal_errors):.4f}
Min: {np.min(normal_errors):.4f}
Max: {np.max(normal_errors):.4f}"""
            
            axes[1].text(0.1, 0.5, stats_text, transform=axes[1].transAxes,
                        fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
        
        axes[1].set_ylabel('Reconstruction Error')
        axes[1].set_title('Error Statistics')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', format=self.save_format)
        plt.show()
        
        print(f"Error distribution plot saved to {save_path}")
    
    def plot_roc_curve(self, results: Dict, save_name: str = "roc_curve.png"):
        """
        Plot ROC curve for anomaly detection performance.
        
        Args:
            results: Dictionary containing evaluation results
            save_name: Name for saved plot
        """
        if 'fpr' not in results or 'tpr' not in results:
            print("ROC curve data not available in results")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        fig.suptitle('Performance Evaluation', fontsize=16, fontweight='bold')
        
        # ROC Curve
        fpr = results['fpr']
        tpr = results['tpr']
        auc_score = results.get('auc_score', 0)
        
        axes[0].plot(fpr, tpr, linewidth=3, label=f'ROC Curve (AUC = {auc_score:.3f})')
        axes[0].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.5)')
        axes[0].fill_between(fpr, tpr, alpha=0.3)
        
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curve')
        axes[0].legend(loc="lower right")
        axes[0].grid(True, alpha=0.3)
        
        # Performance metrics bar chart
        metrics = ['AUC', 'Precision', 'Recall', 'F1-Score', 'Accuracy']
        values = [
            results.get('auc_score', 0),
            results.get('precision', 0),
            results.get('recall', 0),
            results.get('f1_score', 0),
            results.get('accuracy', 0)
        ]
        
        bars = axes[1].bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink'])
        axes[1].set_ylim([0, 1])
        axes[1].set_ylabel('Score')
        axes[1].set_title('Performance Metrics Summary')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', format=self.save_format)
        plt.show()
        
        print(f"ROC curve plot saved to {save_path}")
    
    def plot_reconstruction_examples(
        self,
        detector,
        test_dataset,
        test_labels: np.ndarray = None,
        num_examples: int = 8,
        save_name: str = "reconstruction_examples.png"
    ):
        """
        Plot examples of original vs reconstructed frames.
        
        Args:
            detector: Trained anomaly detector
            test_dataset: Test dataset
            test_labels: Test labels (optional)
            num_examples: Number of examples to show
            save_name: Name for saved plot
        """
        # Get sample data
        data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=num_examples, shuffle=True)
        
        try:
            sample_batch, _ = next(iter(data_loader))
        except:
            print("Could not load sample data for reconstruction examples")
            return
        
        # Get reconstructions
        detector.model.eval()
        with torch.no_grad():
            sample_batch = sample_batch.to(detector.device)
            reconstructions = detector.model(sample_batch)
            errors = detector.model.reconstruction_error(sample_batch)
        
        # Move to CPU for plotting
        originals = sample_batch.cpu().numpy()
        reconstructions = reconstructions.cpu().numpy()
        errors = errors.cpu().numpy()
        
        # Create plot
        fig, axes = plt.subplots(3, num_examples, figsize=(2*num_examples, 6))
        if num_examples == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle('Reconstruction Examples', fontsize=16, fontweight='bold')
        
        for i in range(min(num_examples, originals.shape[0])):
            # Original frame
            axes[0, i].imshow(originals[i, 0], cmap='gray', vmin=0, vmax=1)
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # Reconstructed frame
            axes[1, i].imshow(reconstructions[i, 0], cmap='gray', vmin=0, vmax=1)
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')
            
            # Error map
            error_map = np.abs(originals[i, 0] - reconstructions[i, 0])
            im = axes[2, i].imshow(error_map, cmap='hot', vmin=0, vmax=error_map.max())
            axes[2, i].set_title(f'Error: {errors[i]:.4f}')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', format=self.save_format)
        plt.show()
        
        print(f"Reconstruction examples plot saved to {save_path}")
    
    def plot_detection_timeline(
        self,
        reconstruction_errors: np.ndarray,
        anomaly_flags: np.ndarray,
        threshold: float,
        true_labels: np.ndarray = None,
        save_name: str = "detection_timeline.png"
    ):
        """
        Plot anomaly detection results over time.
        
        Args:
            reconstruction_errors: Array of reconstruction errors
            anomaly_flags: Binary anomaly predictions
            threshold: Detection threshold
            true_labels: Ground truth labels (optional)
            save_name: Name for saved plot
        """
        fig, axes = plt.subplots(2, 1, figsize=self.figsize, sharex=True)
        fig.suptitle('Anomaly Detection Timeline', fontsize=16, fontweight='bold')
        
        frames = range(len(reconstruction_errors))
        
        # Reconstruction error over time
        axes[0].plot(frames, reconstruction_errors, 'b-', linewidth=1, alpha=0.7, label='Reconstruction Error')
        axes[0].axhline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.4f}')
        
        # Highlight detected anomalies
        anomaly_frames = np.where(anomaly_flags)[0]
        if len(anomaly_frames) > 0:
            axes[0].scatter(anomaly_frames, reconstruction_errors[anomaly_frames], 
                           color='red', s=50, alpha=0.8, label='Detected Anomalies', zorder=5)
        
        axes[0].set_ylabel('Reconstruction Error')
        axes[0].set_title('Reconstruction Error vs Threshold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Detection results
        axes[1].fill_between(frames, 0, anomaly_flags, alpha=0.6, color='red', label='Detected Anomalies')
        
        if true_labels is not None:
            # Show ground truth
            axes[1].fill_between(frames, 1.1, 1.1 + true_labels, alpha=0.6, color='green', 
                               label='True Anomalies')
            axes[1].set_ylim(-0.1, 2.2)
        else:
            axes[1].set_ylim(-0.1, 1.1)
        
        axes[1].set_xlabel('Frame Number')
        axes[1].set_ylabel('Anomaly Detection')
        axes[1].set_title('Detection Results Over Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', format=self.save_format)
        plt.show()
        
        print(f"Detection timeline plot saved to {save_path}")
    
    def create_summary_report(
        self,
        results: Dict,
        training_stats: Dict = None,
        dataset_info: Dict = None,
        save_name: str = "summary_report.png"
    ):
        """
        Create a comprehensive summary report visualization.
        
        Args:
            results: Evaluation results
            training_stats: Training statistics
            dataset_info: Dataset information
            save_name: Name for saved plot
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Anomaly Detection System - Summary Report', fontsize=20, fontweight='bold')
        
        # Performance metrics (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        if 'auc_score' in results:
            metrics = ['AUC', 'Precision', 'Recall', 'F1']
            values = [results.get('auc_score', 0), results.get('precision', 0),
                     results.get('recall', 0), results.get('f1_score', 0)]
            
            bars = ax1.bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow'])
            ax1.set_ylim(0, 1)
            ax1.set_title('Performance Metrics')
            ax1.grid(True, alpha=0.3, axis='y')
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # ROC Curve (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        if 'fpr' in results and 'tpr' in results:
            fpr, tpr = results['fpr'], results['tpr']
            auc = results.get('auc_score', 0)
            ax2.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.3f}')
            ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            ax2.set_title('ROC Curve')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Dataset Information (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        info_text = "Dataset Information\n" + "="*20 + "\n"
        
        if dataset_info:
            for key, value in dataset_info.items():
                info_text += f"{key}: {value}\n"
        
        if results:
            info_text += f"\nResults Summary\n" + "="*15 + "\n"
            info_text += f"Total Samples: {results.get('num_samples', 'N/A')}\n"
            info_text += f"Anomalies: {results.get('num_anomalies', 'N/A')}\n"
            info_text += f"Anomaly Rate: {results.get('anomaly_rate', 0):.2%}\n"
        
        ax3.text(0.05, 0.95, info_text, transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        # Training Progress (bottom left)
        ax4 = fig.add_subplot(gs[1, :])
        if training_stats and 'training_history' in training_stats:
            history = training_stats['training_history']
            if 'losses' in history:
                epochs = range(1, len(history['losses']) + 1)
                ax4.plot(epochs, history['losses'], 'b-', linewidth=2, label='Training Loss')
                ax4.set_xlabel('Epoch')
                ax4.set_ylabel('Loss')
                ax4.set_title('Training Progress')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
        
        # Confusion Matrix (bottom left)
        ax5 = fig.add_subplot(gs[2, 0])
        if all(key in results for key in ['true_positives', 'false_positives', 'true_negatives', 'false_negatives']):
            cm = np.array([[results['true_negatives'], results['false_positives']],
                          [results['false_negatives'], results['true_positives']]])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5,
                       xticklabels=['Normal', 'Anomaly'],
                       yticklabels=['Normal', 'Anomaly'])
            ax5.set_title('Confusion Matrix')
            ax5.set_xlabel('Predicted')
            ax5.set_ylabel('Actual')
        
        # System Information (bottom right)
        ax6 = fig.add_subplot(gs[2, 1:])
        ax6.axis('off')
        
        system_text = "System Configuration\n" + "="*20 + "\n"
        if training_stats:
            system_text += f"Training Time: {training_stats.get('total_time', 'N/A'):.1f}s\n"
            system_text += f"Epochs: {training_stats.get('epochs_completed', 'N/A')}\n"
            system_text += f"Best Loss: {training_stats.get('best_loss', 'N/A'):.6f}\n"
        
        if results.get('threshold'):
            system_text += f"Detection Threshold: {results['threshold']:.4f}\n"
        
        ax6.text(0.05, 0.95, system_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # Save comprehensive report
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', format=self.save_format)
        plt.show()
        
        print(f"Summary report saved to {save_path}")


def plot_simple_results(results: Dict, output_dir: str = "outputs"):
    """
    Simple function to create basic visualizations without the full class.
    
    Args:
        results: Dictionary containing evaluation results
        output_dir: Directory to save plots
    """
    visualizer = ResultsVisualizer(output_dir)
    
    # Plot ROC curve if available
    if 'fpr' in results and 'tpr' in results:
        visualizer.plot_roc_curve(results)
    
    # Create summary if enough data available
    if 'auc_score' in results:
        visualizer.create_summary_report(results)


if __name__ == "__main__":
    # Test visualization with dummy data
    print("Testing visualization components...")
    
    # Create dummy results
    dummy_results = {
        'auc_score': 0.85,
        'precision': 0.78,
        'recall': 0.82,
        'f1_score': 0.80,
        'accuracy': 0.88,
        'fpr': np.linspace(0, 1, 100),
        'tpr': np.power(np.linspace(0, 1, 100), 0.5),  # Dummy ROC curve
        'num_samples': 1000,
        'num_anomalies': 100,
        'anomaly_rate': 0.1,
        'threshold': 0.025,
        'true_positives': 82,
        'false_positives': 23,
        'true_negatives': 877,
        'false_negatives': 18
    }
    
    # Create visualizer
    viz = ResultsVisualizer("test_outputs")
    
    # Test ROC curve
    viz.plot_roc_curve(dummy_results)
    
    # Test error distributions
    normal_errors = np.random.normal(0.01, 0.005, 800)
    anomaly_errors = np.random.normal(0.03, 0.01, 100)
    viz.plot_error_distributions(normal_errors, anomaly_errors, threshold=0.025)
    
    print("Visualization test completed!")
