"""
Performance Evaluation Metrics
===============================

This module implements comprehensive evaluation metrics for anomaly detection
systems. It provides standard metrics used in anomaly detection research
and practical deployment scenarios.

Key Metrics:
- Area Under ROC Curve (AUC) - Primary metric for anomaly detection
- Precision, Recall, F1-Score - Classification performance
- Equal Error Rate (EER) - Optimal threshold point
- Precision-Recall curves - Performance across thresholds
- Frame-level and video-level evaluation
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix,
    precision_score, recall_score, f1_score
)
from typing import Dict, List, Tuple, Optional, Union
import warnings


class PerformanceEvaluator:
    """
    Comprehensive performance evaluation for anomaly detection systems.
    
    This class provides a complete suite of evaluation metrics commonly
    used in anomaly detection research and applications. It handles both
    frame-level and video-level evaluation scenarios.
    """
    
    def __init__(self, positive_label: int = 1):
        """
        Initialize the performance evaluator.
        
        Args:
            positive_label: Label value indicating anomaly (typically 1)
        """
        self.positive_label = positive_label
        self.evaluation_cache = {}  # Cache computed metrics
    
    def evaluate_detection(
        self,
        anomaly_scores: np.ndarray,
        true_labels: np.ndarray,
        threshold: Optional[float] = None
    ) -> Dict:
        """
        Comprehensive evaluation of anomaly detection performance.
        
        Args:
            anomaly_scores: Anomaly scores (higher = more anomalous)
            true_labels: Ground truth binary labels (0=normal, 1=anomaly)
            threshold: Detection threshold (computed automatically if None)
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        # Input validation
        if len(anomaly_scores) != len(true_labels):
            raise ValueError("Anomaly scores and labels must have same length")
        
        if len(np.unique(true_labels)) < 2:
            warnings.warn("Only one class present in true labels")
        
        # Convert to numpy arrays
        scores = np.asarray(anomaly_scores)
        labels = np.asarray(true_labels)
        
        # Compute ROC curve and AUC
        fpr, tpr, roc_thresholds = roc_curve(labels, scores, pos_label=self.positive_label)
        auc_score = roc_auc_score(labels, scores)
        
        # Compute Precision-Recall curve
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
            labels, scores, pos_label=self.positive_label
        )
        avg_precision = average_precision_score(labels, scores, pos_label=self.positive_label)
        
        # Find optimal threshold if not provided
        if threshold is None:
            threshold = self._find_optimal_threshold(fpr, tpr, roc_thresholds)
        
        # Compute classification metrics at threshold
        predicted_labels = (scores >= threshold).astype(int)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predicted_labels)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, len(labels))
        
        # Classification metrics
        precision = precision_score(labels, predicted_labels, pos_label=self.positive_label, zero_division=0)
        recall = recall_score(labels, predicted_labels, pos_label=self.positive_label, zero_division=0)
        f1 = f1_score(labels, predicted_labels, pos_label=self.positive_label, zero_division=0)
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0
        
        # Equal Error Rate
        eer, eer_threshold = self._compute_equal_error_rate(fpr, tpr, roc_thresholds)
        
        # Detection rate at different false positive rates
        detection_rates = self._compute_detection_rates(fpr, tpr, [0.01, 0.05, 0.1])
        
        # Package all results
        results = {
            # Primary metrics
            'auc_score': auc_score,
            'average_precision': avg_precision,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'specificity': specificity,
            
            # Threshold analysis
            'threshold': threshold,
            'eer': eer,
            'eer_threshold': eer_threshold,
            
            # Confusion matrix
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            
            # Curves for plotting
            'fpr': fpr,
            'tpr': tpr,
            'roc_thresholds': roc_thresholds,
            'precision_curve': precision_curve,
            'recall_curve': recall_curve,
            'pr_thresholds': pr_thresholds,
            
            # Detection rates
            'detection_rates': detection_rates,
            
            # Summary statistics
            'num_samples': len(labels),
            'num_anomalies': np.sum(labels == self.positive_label),
            'anomaly_rate': np.mean(labels == self.positive_label),
        }
        
        return results
    
    def _find_optimal_threshold(
        self, 
        fpr: np.ndarray, 
        tpr: np.ndarray, 
        thresholds: np.ndarray,
        method: str = 'youden'
    ) -> float:
        """
        Find optimal threshold using various criteria.
        
        Args:
            fpr: False positive rates
            tpr: True positive rates  
            thresholds: Threshold values
            method: Optimization method ('youden', 'f1', 'closest_to_corner')
            
        Returns:
            Optimal threshold value
        """
        if method == 'youden':
            # Youden's J statistic: maximize (TPR - FPR)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            
        elif method == 'closest_to_corner':
            # Point closest to (0, 1) corner
            distances = np.sqrt(fpr**2 + (1 - tpr)**2)
            optimal_idx = np.argmin(distances)
            
        elif method == 'f1':
            # Maximize F1 score (requires computing F1 for each threshold)
            f1_scores = []
            for threshold in thresholds:
                # This is approximate - for exact F1 optimization, 
                # we'd need the original scores and labels
                f1_scores.append(2 * tpr[len(f1_scores)] / (2 * tpr[len(f1_scores)] + fpr[len(f1_scores)] + 1))
            optimal_idx = np.argmax(f1_scores)
            
        else:
            raise ValueError(f"Unknown threshold optimization method: {method}")
        
        return thresholds[optimal_idx]
    
    def _compute_equal_error_rate(
        self, 
        fpr: np.ndarray, 
        tpr: np.ndarray, 
        thresholds: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute Equal Error Rate (EER) where FPR = FNR.
        
        Args:
            fpr: False positive rates
            tpr: True positive rates
            thresholds: Threshold values
            
        Returns:
            Tuple of (EER value, EER threshold)
        """
        # False Negative Rate = 1 - True Positive Rate
        fnr = 1 - tpr
        
        # Find point where FPR ≈ FNR
        diff = np.abs(fpr - fnr)
        eer_idx = np.argmin(diff)
        
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        eer_threshold = thresholds[eer_idx]
        
        return eer, eer_threshold
    
    def _compute_detection_rates(
        self, 
        fpr: np.ndarray, 
        tpr: np.ndarray, 
        target_fprs: List[float]
    ) -> Dict[float, float]:
        """
        Compute detection rates at specific false positive rates.
        
        Args:
            fpr: False positive rates
            tpr: True positive rates
            target_fprs: Target false positive rates
            
        Returns:
            Dictionary mapping FPR to detection rate (TPR)
        """
        detection_rates = {}
        
        for target_fpr in target_fprs:
            # Find closest FPR
            idx = np.argmin(np.abs(fpr - target_fpr))
            detection_rates[target_fpr] = tpr[idx]
        
        return detection_rates
    
    def evaluate_video_level(
        self,
        frame_scores: List[np.ndarray],
        video_labels: np.ndarray,
        aggregation_method: str = 'max'
    ) -> Dict:
        """
        Evaluate performance at video level by aggregating frame scores.
        
        Args:
            frame_scores: List of frame score arrays (one per video)
            video_labels: Binary labels for each video
            aggregation_method: How to aggregate frame scores ('max', 'mean', 'percentile')
            
        Returns:
            Video-level evaluation results
        """
        if len(frame_scores) != len(video_labels):
            raise ValueError("Number of videos must match number of labels")
        
        # Aggregate frame scores to video scores
        video_scores = []
        
        for scores in frame_scores:
            if len(scores) == 0:
                video_scores.append(0.0)
                continue
                
            if aggregation_method == 'max':
                video_score = np.max(scores)
            elif aggregation_method == 'mean':
                video_score = np.mean(scores)
            elif aggregation_method == 'percentile':
                video_score = np.percentile(scores, 95)  # 95th percentile
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation_method}")
            
            video_scores.append(video_score)
        
        video_scores = np.array(video_scores)
        
        # Evaluate at video level
        results = self.evaluate_detection(video_scores, video_labels)
        results['aggregation_method'] = aggregation_method
        results['num_videos'] = len(video_labels)
        
        return results
    
    def compare_methods(
        self,
        methods_results: Dict[str, Dict],
        metrics: List[str] = None
    ) -> pd.DataFrame:
        """
        Compare multiple anomaly detection methods.
        
        Args:
            methods_results: Dictionary mapping method names to their results
            metrics: List of metrics to compare (None for all)
            
        Returns:
            DataFrame comparing methods across metrics
        """
        if metrics is None:
            metrics = ['auc_score', 'average_precision', 'precision', 'recall', 'f1_score', 'eer']
        
        comparison_data = {}
        
        for method_name, results in methods_results.items():
            comparison_data[method_name] = {
                metric: results.get(metric, np.nan) for metric in metrics
            }
        
        df = pd.DataFrame(comparison_data).T
        return df
    
    def statistical_significance_test(
        self,
        scores1: np.ndarray,
        scores2: np.ndarray,
        labels: np.ndarray,
        test: str = 'delong'
    ) -> Dict:
        """
        Test statistical significance between two sets of anomaly scores.
        
        Args:
            scores1: First set of anomaly scores
            scores2: Second set of anomaly scores  
            labels: Ground truth labels
            test: Statistical test to use ('delong', 'bootstrap')
            
        Returns:
            Dictionary with test results
        """
        if test == 'delong':
            # DeLong test for comparing AUCs
            return self._delong_test(scores1, scores2, labels)
        elif test == 'bootstrap':
            # Bootstrap test
            return self._bootstrap_test(scores1, scores2, labels)
        else:
            raise ValueError(f"Unknown statistical test: {test}")
    
    def _delong_test(self, scores1: np.ndarray, scores2: np.ndarray, labels: np.ndarray) -> Dict:
        """
        DeLong test for comparing two AUC scores.
        
        This is a simplified implementation. For production use,
        consider using specialized libraries like scipy or statsmodels.
        """
        auc1 = roc_auc_score(labels, scores1)
        auc2 = roc_auc_score(labels, scores2)
        
        # Simplified DeLong test (actual implementation is more complex)
        # This is a placeholder for educational purposes
        auc_diff = auc1 - auc2
        
        return {
            'auc1': auc1,
            'auc2': auc2,
            'auc_difference': auc_diff,
            'p_value': 0.05,  # Placeholder
            'significant': abs(auc_diff) > 0.05  # Simplified criterion
        }
    
    def _bootstrap_test(self, scores1: np.ndarray, scores2: np.ndarray, labels: np.ndarray, n_bootstrap: int = 1000) -> Dict:
        """
        Bootstrap test for comparing two methods.
        """
        auc1 = roc_auc_score(labels, scores1)
        auc2 = roc_auc_score(labels, scores2)
        
        # Bootstrap resampling
        bootstrap_diffs = []
        n_samples = len(labels)
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            
            boot_labels = labels[indices]
            boot_scores1 = scores1[indices]
            boot_scores2 = scores2[indices]
            
            # Compute AUCs for bootstrap sample
            try:
                boot_auc1 = roc_auc_score(boot_labels, boot_scores1)
                boot_auc2 = roc_auc_score(boot_labels, boot_scores2)
                bootstrap_diffs.append(boot_auc1 - boot_auc2)
            except ValueError:
                # Skip if bootstrap sample has only one class
                continue
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Compute confidence interval
        ci_lower = np.percentile(bootstrap_diffs, 2.5)
        ci_upper = np.percentile(bootstrap_diffs, 97.5)
        
        # Check if difference is significant
        significant = not (ci_lower <= 0 <= ci_upper)
        
        return {
            'auc1': auc1,
            'auc2': auc2,
            'auc_difference': auc1 - auc2,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant': significant,
            'n_bootstrap': len(bootstrap_diffs)
        }
    
    def generate_performance_report(self, results: Dict) -> str:
        """
        Generate a human-readable performance report.
        
        Args:
            results: Evaluation results dictionary
            
        Returns:
            Formatted performance report string
        """
        report = []
        report.append("=" * 60)
        report.append("ANOMALY DETECTION PERFORMANCE REPORT")
        report.append("=" * 60)
        
        # Dataset summary
        report.append(f"\nDataset Summary:")
        report.append(f"  Total samples: {results['num_samples']:,}")
        report.append(f"  Anomalies: {results['num_anomalies']:,}")
        report.append(f"  Anomaly rate: {results['anomaly_rate']:.2%}")
        
        # Primary metrics
        report.append(f"\nPrimary Metrics:")
        report.append(f"  AUC Score: {results['auc_score']:.4f}")
        report.append(f"  Average Precision: {results['average_precision']:.4f}")
        
        # Classification metrics
        report.append(f"\nClassification Performance (Threshold: {results['threshold']:.4f}):")
        report.append(f"  Precision: {results['precision']:.4f}")
        report.append(f"  Recall: {results['recall']:.4f}")
        report.append(f"  F1-Score: {results['f1_score']:.4f}")
        report.append(f"  Accuracy: {results['accuracy']:.4f}")
        report.append(f"  Specificity: {results['specificity']:.4f}")
        
        # Confusion matrix
        report.append(f"\nConfusion Matrix:")
        report.append(f"  True Positives: {results['true_positives']}")
        report.append(f"  False Positives: {results['false_positives']}")
        report.append(f"  True Negatives: {results['true_negatives']}")
        report.append(f"  False Negatives: {results['false_negatives']}")
        
        # Additional metrics
        report.append(f"\nAdditional Metrics:")
        report.append(f"  Equal Error Rate: {results['eer']:.4f}")
        report.append(f"  EER Threshold: {results['eer_threshold']:.4f}")
        
        # Detection rates
        report.append(f"\nDetection Rates at Fixed False Positive Rates:")
        for fpr, dr in results['detection_rates'].items():
            report.append(f"  FPR {fpr:.2%}: Detection Rate {dr:.4f}")
        
        # Performance assessment
        auc = results['auc_score']
        if auc >= 0.95:
            assessment = "Excellent"
        elif auc >= 0.90:
            assessment = "Very Good"
        elif auc >= 0.80:
            assessment = "Good"
        elif auc >= 0.70:
            assessment = "Fair"
        else:
            assessment = "Poor"
        
        report.append(f"\nOverall Assessment: {assessment}")
        
        if auc < 0.70:
            report.append("\nRecommendations:")
            report.append("  - Consider collecting more training data")
            report.append("  - Try different model architectures")
            report.append("  - Adjust preprocessing parameters")
            report.append("  - Experiment with threshold selection methods")
        
        report.append("=" * 60)
        
        return "\n".join(report)


if __name__ == "__main__":
    # Test the performance evaluator
    print("Testing Performance Evaluator...")
    
    # Generate synthetic test data
    np.random.seed(42)
    n_samples = 1000
    
    # Normal samples (lower scores)
    normal_scores = np.random.normal(0.3, 0.1, 800)
    normal_labels = np.zeros(800)
    
    # Anomaly samples (higher scores)
    anomaly_scores = np.random.normal(0.7, 0.15, 200)
    anomaly_labels = np.ones(200)
    
    # Combine
    all_scores = np.concatenate([normal_scores, anomaly_scores])
    all_labels = np.concatenate([normal_labels, anomaly_labels])
    
    # Shuffle
    indices = np.random.permutation(len(all_scores))
    all_scores = all_scores[indices]
    all_labels = all_labels[indices]
    
    # Evaluate
    evaluator = PerformanceEvaluator()
    results = evaluator.evaluate_detection(all_scores, all_labels)
    
    # Print results
    print(f"✓ AUC Score: {results['auc_score']:.4f}")
    print(f"✓ Precision: {results['precision']:.4f}")
    print(f"✓ Recall: {results['recall']:.4f}")
    print(f"✓ F1-Score: {results['f1_score']:.4f}")
    
    # Generate report
    report = evaluator.generate_performance_report(results)
    print("\n" + report)
    
    print("\n✓ Performance evaluator test completed successfully!")
