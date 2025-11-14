"""
Confidence Calibration for Classification Detection

This module implements calibration techniques to transform raw confidence scores
into well-calibrated probabilities that accurately reflect prediction certainty.

Calibration Methods:
- Platt Scaling: Fits a sigmoid to map scores to probabilities
- Temperature Scaling: Scales logits before softmax (simple, effective)
- Isotonic Regression: Non-parametric calibration (flexible)

Why Calibration Matters:
Raw confidence scores from pattern matching may not accurately reflect true
prediction probabilities. Calibration ensures that when a prediction has 80%
confidence, it's actually correct ~80% of the time.

Author: Classmark Development Team
Date: 2025-11-10
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


@dataclass
class CalibrationMetrics:
    """Metrics for calibration quality"""
    method: str
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    brier_score: float  # Brier score (lower is better)
    log_loss: float  # Log loss (lower is better)
    accuracy: float  # Classification accuracy
    num_bins: int = 10


class CalibratedScorer:
    """
    Confidence calibration using Platt scaling, temperature scaling,
    or isotonic regression
    """

    def __init__(
        self,
        method: str = 'platt',
        output_dir: str = './models/calibration',
        num_bins: int = 10
    ):
        """
        Initialize calibrated scorer

        Args:
            method: Calibration method ('platt', 'temperature', 'isotonic')
            output_dir: Directory to save calibration models
            num_bins: Number of bins for calibration curve (default 10)
        """
        self.method = method.lower()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_bins = num_bins

        # Calibration models
        self.platt_model = None  # LogisticRegression for Platt scaling
        self.temperature = 1.0  # Temperature parameter for temperature scaling
        self.isotonic_model = None  # IsotonicRegression model

        self._fitted = False

    def fit(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        verbose: bool = True
    ):
        """
        Fit calibration model on validation data

        Args:
            scores: Raw confidence scores (0-1)
            labels: True labels (0 or 1)
            verbose: Whether to print progress
        """
        if verbose:
            print(f"\nFitting {self.method} calibration model...")
            print(f"  Training examples: {len(scores)}")
            print(f"  Positive examples: {np.sum(labels)}")
            print(f"  Negative examples: {len(labels) - np.sum(labels)}")

        scores = np.array(scores).reshape(-1, 1)
        labels = np.array(labels)

        if self.method == 'platt':
            # Platt scaling: fit logistic regression to scores
            self.platt_model = LogisticRegression(solver='lbfgs', max_iter=1000)
            self.platt_model.fit(scores, labels)

            if verbose:
                coef = self.platt_model.coef_[0][0]
                intercept = self.platt_model.intercept_[0]
                print(f"  Platt parameters: coef={coef:.4f}, intercept={intercept:.4f}")

        elif self.method == 'temperature':
            # Temperature scaling: find optimal temperature using log loss
            from scipy.optimize import minimize_scalar

            def log_loss_fn(T):
                calibrated = self._apply_temperature(scores.flatten(), T)
                # Clip to avoid log(0)
                calibrated = np.clip(calibrated, 1e-10, 1 - 1e-10)
                return -np.mean(
                    labels * np.log(calibrated) +
                    (1 - labels) * np.log(1 - calibrated)
                )

            result = minimize_scalar(log_loss_fn, bounds=(0.1, 10.0), method='bounded')
            self.temperature = result.x

            if verbose:
                print(f"  Optimal temperature: {self.temperature:.4f}")

        elif self.method == 'isotonic':
            # Isotonic regression: monotonic calibration
            self.isotonic_model = IsotonicRegression(out_of_bounds='clip')
            self.isotonic_model.fit(scores.flatten(), labels)

            if verbose:
                print(f"  Isotonic model fitted with {len(self.isotonic_model.X_thresholds_)} thresholds")

        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        self._fitted = True

        if verbose:
            print("  ✓ Calibration model fitted")

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """
        Apply calibration to raw scores

        Args:
            scores: Raw confidence scores (0-1)

        Returns:
            Calibrated probabilities (0-1)
        """
        if not self._fitted:
            raise RuntimeError("Calibration model not fitted. Call fit() first.")

        scores = np.array(scores)
        original_shape = scores.shape
        scores = scores.flatten()

        if self.method == 'platt':
            # Apply Platt scaling
            calibrated = self.platt_model.predict_proba(scores.reshape(-1, 1))[:, 1]

        elif self.method == 'temperature':
            # Apply temperature scaling
            calibrated = self._apply_temperature(scores, self.temperature)

        elif self.method == 'isotonic':
            # Apply isotonic regression
            calibrated = self.isotonic_model.predict(scores)

        return calibrated.reshape(original_shape)

    def _apply_temperature(self, scores: np.ndarray, temperature: float) -> np.ndarray:
        """
        Apply temperature scaling to scores

        For binary classification, we map:
        score -> logit -> scaled_logit -> probability

        Args:
            scores: Raw scores (0-1)
            temperature: Temperature parameter (T>1 = softer, T<1 = sharper)

        Returns:
            Calibrated probabilities
        """
        # Convert probabilities to logits
        scores = np.clip(scores, 1e-10, 1 - 1e-10)
        logits = np.log(scores / (1 - scores))

        # Scale logits
        scaled_logits = logits / temperature

        # Convert back to probabilities
        calibrated = 1 / (1 + np.exp(-scaled_logits))

        return calibrated

    def evaluate(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        verbose: bool = True
    ) -> CalibrationMetrics:
        """
        Evaluate calibration quality

        Args:
            scores: Confidence scores (0-1)
            labels: True labels (0 or 1)
            verbose: Whether to print results

        Returns:
            CalibrationMetrics with evaluation results
        """
        scores = np.array(scores).flatten()
        labels = np.array(labels)

        # Get calibrated probabilities
        if self._fitted:
            calibrated_scores = self.predict_proba(scores)
        else:
            calibrated_scores = scores

        # Calculate metrics
        ece = self._expected_calibration_error(calibrated_scores, labels)
        mce = self._maximum_calibration_error(calibrated_scores, labels)
        brier = self._brier_score(calibrated_scores, labels)
        log_loss_val = self._log_loss(calibrated_scores, labels)

        # Calculate accuracy
        predictions = (calibrated_scores >= 0.5).astype(int)
        accuracy = np.mean(predictions == labels)

        metrics = CalibrationMetrics(
            method=self.method if self._fitted else 'uncalibrated',
            ece=ece,
            mce=mce,
            brier_score=brier,
            log_loss=log_loss_val,
            accuracy=accuracy,
            num_bins=self.num_bins
        )

        if verbose:
            print(f"\nCalibration Metrics ({metrics.method}):")
            print(f"  Expected Calibration Error (ECE): {ece:.4f}")
            print(f"  Maximum Calibration Error (MCE): {mce:.4f}")
            print(f"  Brier Score: {brier:.4f}")
            print(f"  Log Loss: {log_loss_val:.4f}")
            print(f"  Accuracy: {accuracy:.4f}")

        return metrics

    def _expected_calibration_error(
        self,
        scores: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE)

        ECE measures the difference between predicted confidence and actual accuracy
        """
        bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
        ece = 0.0

        for i in range(self.num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            # Find scores in this bin
            in_bin = (scores >= bin_lower) & (scores < bin_upper)

            if i == self.num_bins - 1:  # Last bin includes upper boundary
                in_bin = (scores >= bin_lower) & (scores <= bin_upper)

            if np.sum(in_bin) > 0:
                # Average confidence in bin
                bin_confidence = np.mean(scores[in_bin])

                # Average accuracy in bin
                bin_accuracy = np.mean(labels[in_bin])

                # Weight by proportion of samples in bin
                bin_weight = np.sum(in_bin) / len(scores)

                ece += bin_weight * np.abs(bin_confidence - bin_accuracy)

        return ece

    def _maximum_calibration_error(
        self,
        scores: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """Calculate Maximum Calibration Error (MCE)"""
        bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
        mce = 0.0

        for i in range(self.num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            in_bin = (scores >= bin_lower) & (scores < bin_upper)
            if i == self.num_bins - 1:
                in_bin = (scores >= bin_lower) & (scores <= bin_upper)

            if np.sum(in_bin) > 0:
                bin_confidence = np.mean(scores[in_bin])
                bin_accuracy = np.mean(labels[in_bin])
                mce = max(mce, np.abs(bin_confidence - bin_accuracy))

        return mce

    def _brier_score(self, scores: np.ndarray, labels: np.ndarray) -> float:
        """Calculate Brier score (mean squared error of probabilities)"""
        return np.mean((scores - labels) ** 2)

    def _log_loss(self, scores: np.ndarray, labels: np.ndarray) -> float:
        """Calculate log loss (cross-entropy)"""
        scores = np.clip(scores, 1e-10, 1 - 1e-10)
        return -np.mean(
            labels * np.log(scores) + (1 - labels) * np.log(1 - scores)
        )

    def plot_calibration_curve(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        output_path: Optional[str] = None,
        title: str = "Calibration Curve"
    ):
        """
        Plot calibration curve showing predicted vs actual probabilities

        Args:
            scores: Confidence scores
            labels: True labels
            output_path: Path to save plot (if None, displays plot)
            title: Plot title
        """
        scores = np.array(scores).flatten()
        labels = np.array(labels)

        # Get calibrated scores if model is fitted
        if self._fitted:
            calibrated_scores = self.predict_proba(scores)
        else:
            calibrated_scores = scores

        # Calculate calibration curve
        prob_true, prob_pred = calibration_curve(
            labels,
            calibrated_scores,
            n_bins=self.num_bins,
            strategy='uniform'
        )

        # Create plot
        plt.figure(figsize=(10, 8))

        # Plot calibration curve
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label=f'{self.method} calibrated')

        # Plot perfect calibration
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration', color='gray')

        # Plot before calibration (if fitted)
        if self._fitted:
            prob_true_uncal, prob_pred_uncal = calibration_curve(
                labels,
                scores,
                n_bins=self.num_bins,
                strategy='uniform'
            )
            plt.plot(prob_pred_uncal, prob_true_uncal, marker='s', linewidth=2,
                    label='Uncalibrated', alpha=0.6)

        plt.xlabel('Predicted Probability', fontsize=12)
        plt.ylabel('True Probability', fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300)
            print(f"  ✓ Calibration curve saved to {output_path}")
        else:
            plt.show()

        plt.close()

    def plot_reliability_diagram(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        output_path: Optional[str] = None
    ):
        """
        Plot reliability diagram with histogram

        Args:
            scores: Confidence scores
            labels: True labels
            output_path: Path to save plot
        """
        scores = np.array(scores).flatten()
        labels = np.array(labels)

        if self._fitted:
            calibrated_scores = self.predict_proba(scores)
        else:
            calibrated_scores = scores

        # Calculate calibration curve
        prob_true, prob_pred = calibration_curve(
            labels,
            calibrated_scores,
            n_bins=self.num_bins,
            strategy='uniform'
        )

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [3, 1]})

        # Plot calibration curve
        ax1.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Calibrated')
        ax1.plot([0, 1], [0, 1], linestyle='--', label='Perfect', color='gray')
        ax1.set_xlabel('Predicted Probability')
        ax1.set_ylabel('True Probability')
        ax1.set_title('Reliability Diagram')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Plot histogram
        ax2.hist(calibrated_scores, bins=self.num_bins, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Predicted Probability')
        ax2.set_ylabel('Count')
        ax2.set_title('Confidence Distribution')
        ax2.grid(alpha=0.3, axis='y')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300)
            print(f"  ✓ Reliability diagram saved to {output_path}")
        else:
            plt.show()

        plt.close()

    def save(self, filename: Optional[str] = None):
        """
        Save calibration model to disk

        Args:
            filename: Output filename (default: {method}_calibration.pkl)
        """
        if not self._fitted:
            raise RuntimeError("Calibration model not fitted. Call fit() first.")

        if filename is None:
            filename = f"{self.method}_calibration.pkl"

        output_path = self.output_dir / filename

        model_data = {
            'method': self.method,
            'fitted': self._fitted,
            'num_bins': self.num_bins,
            'platt_model': self.platt_model,
            'temperature': self.temperature,
            'isotonic_model': self.isotonic_model
        }

        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✓ Calibration model saved to {output_path}")

    def load(self, filename: Optional[str] = None):
        """
        Load calibration model from disk

        Args:
            filename: Input filename (default: {method}_calibration.pkl)
        """
        if filename is None:
            filename = f"{self.method}_calibration.pkl"

        input_path = self.output_dir / filename

        if not input_path.exists():
            raise FileNotFoundError(f"Calibration model not found: {input_path}")

        with open(input_path, 'rb') as f:
            model_data = pickle.load(f)

        self.method = model_data['method']
        self._fitted = model_data['fitted']
        self.num_bins = model_data['num_bins']
        self.platt_model = model_data['platt_model']
        self.temperature = model_data['temperature']
        self.isotonic_model = model_data['isotonic_model']

        print(f"✓ Calibration model loaded from {input_path}")


def main():
    """Example usage and testing"""
    print("=" * 70)
    print("Confidence Calibration Example")
    print("=" * 70)
    print()

    # Generate synthetic data for demonstration
    np.random.seed(42)

    # Simulate overconfident classifier
    n_samples = 1000
    true_labels = np.random.binomial(1, 0.5, n_samples)

    # Raw scores are biased toward extremes (overconfident)
    raw_scores = np.random.beta(2, 2, n_samples)
    raw_scores = np.where(true_labels == 1, raw_scores * 0.7 + 0.3, raw_scores * 0.7)

    print(f"Generated {n_samples} synthetic examples")
    print(f"  Positive: {np.sum(true_labels)}")
    print(f"  Negative: {n_samples - np.sum(true_labels)}")
    print()

    # Split into calibration and validation
    cal_scores, val_scores, cal_labels, val_labels = train_test_split(
        raw_scores, true_labels, test_size=0.7, random_state=42
    )

    print(f"Split: {len(cal_scores)} calibration, {len(val_scores)} validation")
    print()

    # Test each calibration method
    for method in ['platt', 'temperature', 'isotonic']:
        print(f"\n{'='*70}")
        print(f"Testing {method.upper()} calibration")
        print(f"{'='*70}")

        # Create and fit calibrator
        calibrator = CalibratedScorer(method=method, num_bins=10)
        calibrator.fit(cal_scores, cal_labels)

        # Evaluate before and after
        print("\nBefore calibration:")
        calibrator._fitted = False  # Temporarily unfitted to get uncalibrated metrics
        uncal_metrics = calibrator.evaluate(val_scores, val_labels, verbose=True)

        print("\nAfter calibration:")
        calibrator._fitted = True
        cal_metrics = calibrator.evaluate(val_scores, val_labels, verbose=True)

        # Show improvement
        print("\nImprovement:")
        print(f"  ECE: {uncal_metrics.ece:.4f} → {cal_metrics.ece:.4f} ({((uncal_metrics.ece - cal_metrics.ece) / uncal_metrics.ece * 100):.1f}% reduction)")
        print(f"  Brier: {uncal_metrics.brier_score:.4f} → {cal_metrics.brier_score:.4f}")

        # Save model
        calibrator.save()

        # Plot calibration curve
        calibrator.plot_calibration_curve(
            val_scores,
            val_labels,
            output_path=f"./models/calibration/{method}_calibration_curve.png",
            title=f"{method.capitalize()} Calibration Curve"
        )

    print("\n" + "=" * 70)
    print("Calibration demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
