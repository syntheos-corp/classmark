"""
Baseline Evaluation Script

Calculates baseline metrics for the current classification marking detection system
before applying SOTA enhancements.

Usage:
    python run_baseline_evaluation.py

Author: Classmark Development Team
Date: 2025-11-10
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import evaluation framework
from src.core.evaluation import GroundTruthManager, ClassificationEvaluator

# Import the current classification scanner
from src.core.classification_scanner import ClassificationScanner


def run_baseline_evaluation():
    """Run baseline evaluation on current system"""

    print("=" * 80)
    print("BASELINE EVALUATION - Current System Performance")
    print("=" * 80)
    print()

    # Initialize ground truth manager
    print("Loading ground truth annotations...")
    gt_manager = GroundTruthManager("documents/ground_truth.json")
    print(f"Loaded {len(gt_manager.annotations)} annotated documents")
    print()

    # Initialize scanner with config dictionary
    print("Initializing classification scanner...")
    config = {
        'sensitivity_threshold': 0.5,  # Medium sensitivity (50%)
        'fuzzy_matching': True,
        'fuzzy_threshold': 85,
        'use_llm': False,  # Disable LLM for faster baseline
        'llm_model': 'qwen3:8b'
    }
    scanner = ClassificationScanner(config)
    print("Scanner initialized (medium sensitivity 50%, fuzzy matching ON, LLM OFF)")
    print()

    # Initialize evaluator
    evaluator = ClassificationEvaluator(gt_manager)

    # Run evaluation on test_data directory (synthetic documents only for now)
    print("=" * 80)
    print("PHASE 1: Evaluating Synthetic Test Documents")
    print("=" * 80)
    print()

    # Get only test_data files
    test_data_files = [f for f in gt_manager.get_all_files()
                       if 'test_data' in f]

    print(f"Evaluating {len(test_data_files)} synthetic test documents...")
    print()

    metrics = evaluator.evaluate_dataset(scanner, test_data_files)

    # Generate and print report
    print()
    print("=" * 80)
    print("BASELINE METRICS - Synthetic Test Documents")
    print("=" * 80)
    report = evaluator.generate_report(metrics, "evaluation_baseline_synthetic.txt")
    print(report)

    # Export calibration data
    evaluator.export_calibration_data("calibration_data_baseline_synthetic.json")

    # Run evaluation on real documents (if any exist on disk)
    print()
    print("=" * 80)
    print("PHASE 2: Evaluating Real Declassified Documents")
    print("=" * 80)
    print()

    real_doc_files = [f for f in gt_manager.get_all_files()
                      if 'test_data' not in f and Path(f).exists()]

    if real_doc_files:
        print(f"Evaluating {len(real_doc_files)} real declassified documents...")
        print()

        metrics_real = evaluator.evaluate_dataset(scanner, real_doc_files)

        print()
        print("=" * 80)
        print("BASELINE METRICS - Real Declassified Documents")
        print("=" * 80)
        report_real = evaluator.generate_report(metrics_real, "evaluation_baseline_real.txt")
        print(report_real)

        evaluator.export_calibration_data("calibration_data_baseline_real.json")
    else:
        print("No real documents found on disk. Skipping real document evaluation.")
        print("(PDFs may need to be present in documents/ folder)")

    # Summary
    print()
    print("=" * 80)
    print("BASELINE EVALUATION COMPLETE")
    print("=" * 80)
    print()
    print("Files generated:")
    print("  - evaluation_baseline_synthetic.txt (detailed report)")
    print("  - calibration_data_baseline_synthetic.json (calibration data)")
    if real_doc_files:
        print("  - evaluation_baseline_real.txt (real docs report)")
        print("  - calibration_data_baseline_real.json (real docs calibration data)")
    print()
    print("Key Findings:")
    print(f"  Precision:  {metrics.precision:.4f} ({metrics.precision * 100:.2f}%)")
    print(f"  Recall:     {metrics.recall:.4f} ({metrics.recall * 100:.2f}%)")
    print(f"  F1 Score:   {metrics.f1_score:.4f} ({metrics.f1_score * 100:.2f}%)")
    print(f"  Accuracy:   {metrics.accuracy:.4f} ({metrics.accuracy * 100:.2f}%)")
    print(f"  Avg Time:   {metrics.avg_processing_time * 1000:.2f} ms")
    print()
    print("These metrics represent the BEFORE state for comparison after SOTA enhancements.")
    print("=" * 80)


if __name__ == "__main__":
    try:
        run_baseline_evaluation()
    except Exception as e:
        print(f"\nERROR: Baseline evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
