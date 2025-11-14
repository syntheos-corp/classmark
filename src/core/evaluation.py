"""
Classification Marking Detection - Evaluation Framework

This module provides comprehensive evaluation capabilities for the classification
marking detection system, including:
- Ground truth annotation management
- Precision, recall, F1, accuracy metrics
- Per-level classification accuracy
- Processing time benchmarks
- Calibration curve generation
- Detailed error analysis

Author: Classmark Development Team
Date: 2025-11-10
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

# Try to import sklearn for metrics, make it optional
try:
    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        accuracy_score,
        confusion_matrix,
        classification_report
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn>=1.3.0")


@dataclass
class GroundTruth:
    """Ground truth annotation for a single document"""
    file_path: str
    has_classification: bool
    classification_level: Optional[str] = None  # TOP SECRET, SECRET, CONFIDENTIAL, CUI, None
    control_markings: Optional[List[str]] = None  # NOFORN, SCI, etc.
    has_portion_marks: bool = False
    has_banner: bool = False
    has_authority_block: bool = False
    is_declassified: bool = False
    expected_confidence_range: Optional[Tuple[float, float]] = None  # (min, max)
    notes: Optional[str] = None


@dataclass
class EvaluationResult:
    """Result from evaluating a single document"""
    file_path: str
    ground_truth: GroundTruth
    predicted_has_classification: bool
    predicted_level: Optional[str]
    predicted_confidence: float
    processing_time: float
    is_correct: bool
    error_type: Optional[str] = None  # 'false_positive', 'false_negative', 'wrong_level'


@dataclass
class EvaluationMetrics:
    """Aggregate evaluation metrics"""
    total_documents: int
    precision: float
    recall: float
    f1_score: float
    accuracy: float

    # Per-level metrics
    level_accuracy: Dict[str, float]

    # Timing metrics
    avg_processing_time: float
    median_processing_time: float
    p95_processing_time: float
    p99_processing_time: float

    # Confusion matrix
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    # Detailed breakdowns
    false_positive_examples: List[str]
    false_negative_examples: List[str]
    wrong_level_examples: List[Tuple[str, str, str]]  # (file, expected, predicted)


class GroundTruthManager:
    """Manages ground truth annotations for test documents"""

    def __init__(self, ground_truth_path: str):
        """
        Initialize ground truth manager

        Args:
            ground_truth_path: Path to ground truth JSON file
        """
        self.ground_truth_path = Path(ground_truth_path)
        self.annotations: Dict[str, GroundTruth] = {}
        self.load()

    def load(self):
        """Load ground truth annotations from JSON file"""
        if not self.ground_truth_path.exists():
            print(f"Ground truth file not found: {self.ground_truth_path}")
            print("Creating new ground truth file...")
            self.annotations = {}
            return

        with open(self.ground_truth_path, 'r') as f:
            data = json.load(f)

        # Convert JSON to GroundTruth objects
        self.annotations = {}
        for file_path, annotation in data.items():
            # Handle expected_confidence_range (can be None, list, or tuple)
            conf_range = annotation.get('expected_confidence_range')
            if conf_range is not None and not isinstance(conf_range, tuple):
                conf_range = tuple(conf_range)

            self.annotations[file_path] = GroundTruth(
                file_path=file_path,
                has_classification=annotation['has_classification'],
                classification_level=annotation.get('classification_level'),
                control_markings=annotation.get('control_markings', []),
                has_portion_marks=annotation.get('has_portion_marks', False),
                has_banner=annotation.get('has_banner', False),
                has_authority_block=annotation.get('has_authority_block', False),
                is_declassified=annotation.get('is_declassified', False),
                expected_confidence_range=conf_range,
                notes=annotation.get('notes')
            )

        print(f"Loaded {len(self.annotations)} ground truth annotations")

    def save(self):
        """Save ground truth annotations to JSON file"""
        data = {}
        for file_path, gt in self.annotations.items():
            annotation = asdict(gt)
            # Convert Path to string if necessary
            annotation['file_path'] = str(annotation['file_path'])
            data[file_path] = annotation

        with open(self.ground_truth_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Saved {len(self.annotations)} ground truth annotations to {self.ground_truth_path}")

    def add_annotation(self, annotation: GroundTruth):
        """Add or update a ground truth annotation"""
        self.annotations[annotation.file_path] = annotation

    def get_annotation(self, file_path: str) -> Optional[GroundTruth]:
        """Get ground truth annotation for a file"""
        return self.annotations.get(file_path)

    def get_all_files(self) -> List[str]:
        """Get list of all annotated files"""
        return list(self.annotations.keys())


class ClassificationEvaluator:
    """
    Comprehensive evaluation framework for classification marking detection

    Features:
    - Precision, recall, F1, accuracy metrics
    - Per-level classification accuracy
    - Processing time benchmarks
    - Detailed error analysis
    - Confidence calibration data collection
    """

    def __init__(self, ground_truth_manager: GroundTruthManager):
        """
        Initialize evaluator

        Args:
            ground_truth_manager: Manager for ground truth annotations
        """
        self.gt_manager = ground_truth_manager
        self.results: List[EvaluationResult] = []

    def evaluate_file(self, scanner, file_path: str) -> Optional[EvaluationResult]:
        """
        Evaluate scanner on a single file

        Args:
            scanner: ClassificationScanner instance
            file_path: Path to file to evaluate

        Returns:
            EvaluationResult or None if no ground truth available
        """
        # Get ground truth
        gt = self.gt_manager.get_annotation(file_path)
        if gt is None:
            print(f"Warning: No ground truth for {file_path}")
            return None

        # Run scanner
        start_time = time.time()
        try:
            result = scanner.scan_file(file_path)
            processing_time = time.time() - start_time
        except Exception as e:
            print(f"Error scanning {file_path}: {e}")
            return None

        # Extract predictions
        predicted_has_classification = result.has_markings
        predicted_level = str(result.classification_level.value) if result.has_markings and result.classification_level else None
        predicted_confidence = result.overall_confidence

        # Determine correctness
        is_correct = (predicted_has_classification == gt.has_classification)
        error_type = None

        if not is_correct:
            if predicted_has_classification and not gt.has_classification:
                error_type = 'false_positive'
            elif not predicted_has_classification and gt.has_classification:
                error_type = 'false_negative'
        elif is_correct and gt.has_classification:
            # Check if level is correct
            if predicted_level != gt.classification_level:
                error_type = 'wrong_level'
                is_correct = False

        return EvaluationResult(
            file_path=file_path,
            ground_truth=gt,
            predicted_has_classification=predicted_has_classification,
            predicted_level=predicted_level,
            predicted_confidence=predicted_confidence,
            processing_time=processing_time,
            is_correct=is_correct,
            error_type=error_type
        )

    def evaluate_dataset(self, scanner, file_paths: Optional[List[str]] = None) -> EvaluationMetrics:
        """
        Evaluate scanner on entire dataset

        Args:
            scanner: ClassificationScanner instance
            file_paths: Optional list of files to evaluate. If None, evaluates all annotated files.

        Returns:
            EvaluationMetrics with comprehensive statistics
        """
        if file_paths is None:
            file_paths = self.gt_manager.get_all_files()

        print(f"Evaluating {len(file_paths)} documents...")

        self.results = []
        for i, file_path in enumerate(file_paths, 1):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(file_paths)}")

            result = self.evaluate_file(scanner, file_path)
            if result:
                self.results.append(result)

        print(f"Evaluated {len(self.results)} documents")

        return self.calculate_metrics()

    def calculate_metrics(self) -> EvaluationMetrics:
        """
        Calculate comprehensive evaluation metrics from results

        Returns:
            EvaluationMetrics object with all statistics
        """
        if not self.results:
            raise ValueError("No evaluation results available. Run evaluate_dataset first.")

        # Collect predictions and ground truth
        y_true = [int(r.ground_truth.has_classification) for r in self.results]
        y_pred = [int(r.predicted_has_classification) for r in self.results]

        # Calculate basic metrics
        if SKLEARN_AVAILABLE:
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            accuracy = accuracy_score(y_true, y_pred)

            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        else:
            # Manual calculation
            tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
            fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
            fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
            tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

        # Per-level accuracy
        level_accuracy = self._calculate_level_accuracy()

        # Timing metrics
        times = [r.processing_time for r in self.results]
        times.sort()
        avg_time = sum(times) / len(times)
        median_time = times[len(times) // 2]
        p95_time = times[int(len(times) * 0.95)] if len(times) > 20 else times[-1]
        p99_time = times[int(len(times) * 0.99)] if len(times) > 100 else times[-1]

        # Error examples
        false_positives = [r.file_path for r in self.results if r.error_type == 'false_positive']
        false_negatives = [r.file_path for r in self.results if r.error_type == 'false_negative']
        wrong_levels = [
            (r.file_path, r.ground_truth.classification_level, r.predicted_level)
            for r in self.results if r.error_type == 'wrong_level'
        ]

        return EvaluationMetrics(
            total_documents=len(self.results),
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy,
            level_accuracy=level_accuracy,
            avg_processing_time=avg_time,
            median_processing_time=median_time,
            p95_processing_time=p95_time,
            p99_processing_time=p99_time,
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            false_positive_examples=false_positives[:10],  # Limit to 10 examples
            false_negative_examples=false_negatives[:10],
            wrong_level_examples=wrong_levels[:10]
        )

    def _calculate_level_accuracy(self) -> Dict[str, float]:
        """Calculate per-level classification accuracy"""
        level_counts = defaultdict(lambda: {'correct': 0, 'total': 0})

        for result in self.results:
            if result.ground_truth.has_classification:
                level = result.ground_truth.classification_level
                level_counts[level]['total'] += 1
                if result.predicted_level == level:
                    level_counts[level]['correct'] += 1

        level_accuracy = {}
        for level, counts in level_counts.items():
            accuracy = counts['correct'] / counts['total'] if counts['total'] > 0 else 0.0
            level_accuracy[level] = accuracy

        return level_accuracy

    def generate_report(self, metrics: EvaluationMetrics, output_path: Optional[str] = None) -> str:
        """
        Generate detailed evaluation report

        Args:
            metrics: EvaluationMetrics object
            output_path: Optional path to save report

        Returns:
            Report string
        """
        report_lines = [
            "=" * 80,
            "CLASSIFICATION MARKING DETECTION - EVALUATION REPORT",
            "=" * 80,
            "",
            f"Total Documents Evaluated: {metrics.total_documents}",
            "",
            "=" * 80,
            "DETECTION METRICS",
            "=" * 80,
            f"Precision:  {metrics.precision:.4f}  ({metrics.precision * 100:.2f}%)",
            f"Recall:     {metrics.recall:.4f}  ({metrics.recall * 100:.2f}%)",
            f"F1 Score:   {metrics.f1_score:.4f}  ({metrics.f1_score * 100:.2f}%)",
            f"Accuracy:   {metrics.accuracy:.4f}  ({metrics.accuracy * 100:.2f}%)",
            "",
            "Confusion Matrix:",
            f"  True Positives:  {metrics.true_positives}",
            f"  False Positives: {metrics.false_positives}",
            f"  True Negatives:  {metrics.true_negatives}",
            f"  False Negatives: {metrics.false_negatives}",
            "",
            "=" * 80,
            "PER-LEVEL ACCURACY",
            "=" * 80,
        ]

        # Sort level_accuracy, handling None keys
        sorted_levels = sorted(
            metrics.level_accuracy.items(),
            key=lambda x: (x[0] is None, x[0] or "")
        )
        for level, accuracy in sorted_levels:
            level_str = level if level is not None else "(None/Unknown)"
            report_lines.append(f"{level_str:20s}: {accuracy:.4f}  ({accuracy * 100:.2f}%)")

        report_lines.extend([
            "",
            "=" * 80,
            "PERFORMANCE METRICS",
            "=" * 80,
            f"Average Processing Time:  {metrics.avg_processing_time * 1000:.2f} ms",
            f"Median Processing Time:   {metrics.median_processing_time * 1000:.2f} ms",
            f"95th Percentile Time:     {metrics.p95_processing_time * 1000:.2f} ms",
            f"99th Percentile Time:     {metrics.p99_processing_time * 1000:.2f} ms",
            "",
            "=" * 80,
            "ERROR ANALYSIS",
            "=" * 80,
        ])

        if metrics.false_positive_examples:
            report_lines.append("\nFalse Positives (flagged but should not be):")
            for fp in metrics.false_positive_examples:
                report_lines.append(f"  - {fp}")

        if metrics.false_negative_examples:
            report_lines.append("\nFalse Negatives (missed classification markings):")
            for fn in metrics.false_negative_examples:
                report_lines.append(f"  - {fn}")

        if metrics.wrong_level_examples:
            report_lines.append("\nWrong Classification Level:")
            for file_path, expected, predicted in metrics.wrong_level_examples:
                report_lines.append(f"  - {file_path}")
                report_lines.append(f"    Expected: {expected}, Predicted: {predicted}")

        report_lines.append("")
        report_lines.append("=" * 80)

        report = "\n".join(report_lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"Report saved to: {output_path}")

        return report

    def export_calibration_data(self, output_path: str):
        """
        Export data for confidence calibration

        Args:
            output_path: Path to save calibration data (JSON)
        """
        calibration_data = []

        for result in self.results:
            calibration_data.append({
                'file_path': result.file_path,
                'predicted_confidence': result.predicted_confidence,
                'actual_has_classification': result.ground_truth.has_classification,
                'is_correct': result.is_correct
            })

        with open(output_path, 'w') as f:
            json.dump(calibration_data, f, indent=2)

        print(f"Calibration data exported to: {output_path}")


if __name__ == "__main__":
    # Example usage
    print("Classification Marking Detection - Evaluation Framework")
    print("=" * 60)
    print("\nThis module provides evaluation capabilities for the")
    print("classification marking detection system.")
    print("\nUsage:")
    print("  from evaluation import ClassificationEvaluator, GroundTruthManager")
    print("  gt_manager = GroundTruthManager('ground_truth.json')")
    print("  evaluator = ClassificationEvaluator(gt_manager)")
    print("  metrics = evaluator.evaluate_dataset(scanner)")
    print("  report = evaluator.generate_report(metrics)")
