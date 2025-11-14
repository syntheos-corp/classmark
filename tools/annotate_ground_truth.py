"""
Ground Truth Annotation Script

Automatically generates ground truth annotations for test documents
based on filename patterns and content analysis.

Usage:
    python annotate_ground_truth.py

Author: Classmark Development Team
Date: 2025-11-10
"""

import json
import re
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.evaluation import GroundTruth, GroundTruthManager


def infer_ground_truth_from_filename(file_path: str) -> GroundTruth:
    """
    Infer ground truth annotation from filename patterns

    Filename patterns:
    - ts_XX_* = TOP SECRET
    - s_XX_* = SECRET
    - c_XX_* = CONFIDENTIAL
    - cui_XX_* = CUI
    - false_positive_* = should NOT be flagged
    - declassified_* = declassified document
    - mixed_* = contains multiple classification levels
    - errors_* = documents with marking errors
    """
    filename = Path(file_path).name.lower()

    # Default annotation
    gt = GroundTruth(
        file_path=file_path,
        has_classification=False,
        classification_level=None,
        control_markings=[],
        has_portion_marks=False,
        has_banner=False,
        has_authority_block=False,
        is_declassified=False,
        expected_confidence_range=None,
        notes=""
    )

    # TOP SECRET documents
    if filename.startswith('ts_') or 'top_secret' in filename:
        gt.has_classification = True
        gt.classification_level = "TOP SECRET"
        gt.has_banner = True
        gt.has_portion_marks = True
        gt.has_authority_block = True
        gt.expected_confidence_range = (0.85, 1.0)
        gt.notes = "TOP SECRET document with proper markings"

    # SECRET documents
    elif filename.startswith('s_') or (filename.startswith('secret') and not filename.startswith('secret_')):
        gt.has_classification = True
        gt.classification_level = "SECRET"
        gt.has_banner = True
        gt.has_portion_marks = True
        gt.has_authority_block = True
        gt.expected_confidence_range = (0.80, 1.0)
        gt.notes = "SECRET document with proper markings"

    # CONFIDENTIAL documents
    elif filename.startswith('c_') or 'confidential' in filename:
        gt.has_classification = True
        gt.classification_level = "CONFIDENTIAL"
        gt.has_banner = True
        gt.has_portion_marks = True
        gt.has_authority_block = True
        gt.expected_confidence_range = (0.75, 1.0)
        gt.notes = "CONFIDENTIAL document with proper markings"

    # CUI documents
    elif filename.startswith('cui_') or 'controlled_unclassified' in filename:
        gt.has_classification = True
        gt.classification_level = "CUI"
        gt.has_banner = True
        gt.has_portion_marks = False  # CUI typically doesn't use portion marks
        gt.has_authority_block = False  # CUI has simpler marking
        gt.expected_confidence_range = (0.70, 0.95)
        gt.notes = "CUI document"

    # Legacy CUI (FOUO, SBU, LES)
    elif any(x in filename for x in ['fouo', 'sbu', 'les', 'limdis']):
        gt.has_classification = True
        gt.classification_level = "CUI"
        gt.has_banner = True
        gt.has_portion_marks = False
        gt.has_authority_block = False
        gt.expected_confidence_range = (0.65, 0.90)
        gt.notes = "Legacy CUI marking (FOUO/SBU/LES)"

    # Declassified documents
    elif filename.startswith('declassified') or 'declassified' in filename:
        gt.has_classification = True  # Has markings but is declassified
        gt.is_declassified = True
        gt.has_banner = True
        gt.has_authority_block = True
        gt.expected_confidence_range = (0.70, 0.95)
        gt.notes = "Declassified document with original markings visible"

        # Try to infer original classification level
        if 'top_secret' in filename or '_ts_' in filename:
            gt.classification_level = "TOP SECRET"
        elif 'secret' in filename:
            gt.classification_level = "SECRET"
        elif 'confidential' in filename:
            gt.classification_level = "CONFIDENTIAL"

    # Documents with marking errors
    elif filename.startswith('errors_') or 'marking_errors' in filename:
        gt.has_classification = True
        gt.has_banner = False  # May be missing banners
        gt.has_portion_marks = False  # May be missing portion marks
        gt.has_authority_block = False  # May be missing authority block
        gt.expected_confidence_range = (0.50, 0.85)
        gt.notes = "Document with marking errors (missing required elements)"

    # Mixed classification levels
    elif filename.startswith('mixed_'):
        gt.has_classification = True
        gt.classification_level = "TOP SECRET"  # Highest level wins
        gt.has_banner = True
        gt.has_portion_marks = True
        gt.has_authority_block = True
        gt.expected_confidence_range = (0.75, 1.0)
        gt.notes = "Document with multiple classification levels (highest: TOP SECRET)"

    # False positive test cases
    elif filename.startswith('false_positive') or 'no_classification' in filename:
        gt.has_classification = False
        gt.classification_level = None
        gt.expected_confidence_range = (0.0, 0.3)
        gt.notes = "Should NOT be flagged - contains casual usage or no markings"

    # Unclassified documents
    elif filename.startswith('unclassified') or filename.startswith('u_'):
        gt.has_classification = False
        gt.classification_level = None
        gt.expected_confidence_range = (0.0, 0.2)
        gt.notes = "Unclassified document"

    return gt


def analyze_file_content(file_path: str, gt: GroundTruth) -> GroundTruth:
    """
    Analyze file content to refine ground truth annotation

    Looks for:
    - Control markings (NOFORN, SCI, etc.)
    - Portion markings ((TS), (S), (C))
    - Authority blocks (CLASSIFIED BY, DERIVED FROM)
    """
    try:
        # Read file content
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Check for control markings
        control_markings = []
        if re.search(r'\bNOFORN\b', content, re.IGNORECASE):
            control_markings.append('NOFORN')
        if re.search(r'\b//SCI\b', content, re.IGNORECASE):
            control_markings.append('SCI')
        if re.search(r'\bORCON\b', content, re.IGNORECASE):
            control_markings.append('ORCON')
        if re.search(r'\bIMCON\b', content, re.IGNORECASE):
            control_markings.append('IMCON')
        if re.search(r'\bFISA\b', content, re.IGNORECASE):
            control_markings.append('FISA')

        gt.control_markings = control_markings

        # Check for portion markings
        if re.search(r'\((TS|S|C|U)\)', content):
            gt.has_portion_marks = True

        # Check for authority blocks
        if re.search(r'CLASSIFIED BY:', content, re.IGNORECASE):
            gt.has_authority_block = True
        if re.search(r'DERIVED FROM:', content, re.IGNORECASE):
            gt.has_authority_block = True

        # Check for declassification indicators
        if re.search(r'DECLASSIFIED', content, re.IGNORECASE):
            gt.is_declassified = True

    except Exception as e:
        print(f"Warning: Could not analyze {file_path}: {e}")

    return gt


def generate_ground_truth_for_directory(directory: str, output_file: str):
    """
    Generate ground truth annotations for all files in a directory

    Args:
        directory: Path to directory containing test documents
        output_file: Path to output JSON file
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"Error: Directory not found: {directory}")
        return

    print(f"Scanning directory: {directory}")
    print("=" * 60)

    # Find all text and PDF files
    files = list(dir_path.glob('*.txt')) + list(dir_path.glob('*.pdf'))
    print(f"Found {len(files)} files")

    annotations = {}

    for file_path in sorted(files):
        print(f"\nProcessing: {file_path.name}")

        # Infer ground truth from filename
        gt = infer_ground_truth_from_filename(str(file_path))

        # Analyze content for refinement
        gt = analyze_file_content(str(file_path), gt)

        # Store annotation (use path relative to project root for portability)
        try:
            rel_path = str(file_path.relative_to(Path.cwd()))
        except ValueError:
            # If file_path is already relative, use it as-is
            rel_path = str(file_path)

        annotations[rel_path] = {
            'has_classification': gt.has_classification,
            'classification_level': gt.classification_level,
            'control_markings': gt.control_markings,
            'has_portion_marks': gt.has_portion_marks,
            'has_banner': gt.has_banner,
            'has_authority_block': gt.has_authority_block,
            'is_declassified': gt.is_declassified,
            'expected_confidence_range': list(gt.expected_confidence_range) if gt.expected_confidence_range else None,
            'notes': gt.notes
        }

        print(f"  Classification: {gt.classification_level or 'None'}")
        print(f"  Has Markings: {gt.has_classification}")
        print(f"  Control Markings: {', '.join(gt.control_markings) if gt.control_markings else 'None'}")

    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Generated {len(annotations)} annotations")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    print("Ground Truth Annotation Generator")
    print("=" * 60)

    # Generate for test_data directory
    test_data_dir = "documents/test_data"
    output_file = "documents/ground_truth.json"

    generate_ground_truth_for_directory(test_data_dir, output_file)

    print("\n" + "=" * 60)
    print("Ground truth generation complete!")
    print("\nNext steps:")
    print("1. Review the generated ground_truth.json file")
    print("2. Make any manual corrections if needed")
    print("3. Run evaluation: python -c \"from evaluation import *; ...\"")
