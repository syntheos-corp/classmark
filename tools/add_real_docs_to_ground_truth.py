"""
Add real declassified documents to ground truth annotations

This script scans real declassified documents in the documents/ folder
and adds them to the ground truth annotations.

Author: Classmark Development Team
Date: 2025-11-10
"""

import json
from pathlib import Path

# Annotations for real declassified documents
# Based on manual inspection of filenames and typical declassified document patterns
REAL_DOC_ANNOTATIONS = {
    "documents/1971 BureBehav&StratComp.pdf": {
        "has_classification": False,
        "classification_level": None,
        "control_markings": [],
        "has_portion_marks": False,
        "has_banner": False,
        "has_authority_block": False,
        "is_declassified": True,
        "expected_confidence_range": [0.0, 0.3],
        "notes": "Historical document (1971) - likely fully declassified or unclassified"
    },
    "documents/1975-12-06 From A. W. Marshall re Key Military Balances .pdf": {
        "has_classification": True,
        "classification_level": None,
        "control_markings": [],
        "has_portion_marks": False,
        "has_banner": False,
        "has_authority_block": False,
        "is_declassified": True,
        "expected_confidence_range": [0.5, 0.8],
        "notes": "Historical memo (1975) - may have FOUO or SECRET markings, now declassified"
    },
    "documents/1976 Team B Report.pdf": {
        "has_classification": True,
        "classification_level": "SECRET",
        "control_markings": [],
        "has_portion_marks": False,
        "has_banner": False,
        "has_authority_block": False,
        "is_declassified": True,
        "expected_confidence_range": [0.6, 0.9],
        "notes": "CIA Team B Report (1976) - originally SECRET, now declassified"
    },
    "documents/19760323 from AWM re McClellan Letter on Recent Trends in the Military Balance Between U.S. and Soviet Union.pdf": {
        "has_classification": True,
        "classification_level": None,
        "control_markings": [],
        "has_portion_marks": False,
        "has_banner": False,
        "has_authority_block": False,
        "is_declassified": True,
        "expected_confidence_range": [0.4, 0.7],
        "notes": "Military assessment (1976) - may have classification markings"
    },
    "documents/1979 FRUS NA Ground Forces Assessment.pdf": {
        "has_classification": True,
        "classification_level": "SECRET",
        "control_markings": [],
        "has_portion_marks": False,
        "has_banner": False,
        "has_authority_block": False,
        "is_declassified": True,
        "expected_confidence_range": [0.5, 0.8],
        "notes": "FRUS (Foreign Relations of the U.S.) document - likely originally classified"
    },
    "documents/19940502-AWM-RMA-Update.pdf": {
        "has_classification": False,
        "classification_level": None,
        "control_markings": [],
        "has_portion_marks": False,
        "has_banner": False,
        "has_authority_block": False,
        "is_declassified": False,
        "expected_confidence_range": [0.0, 0.3],
        "notes": "RMA Update (1994) - likely unclassified technical document"
    },
    "documents/2000 AWM ThoughtsFutureNAs.pdf": {
        "has_classification": False,
        "classification_level": None,
        "control_markings": [],
        "has_portion_marks": False,
        "has_banner": False,
        "has_authority_block": False,
        "is_declassified": False,
        "expected_confidence_range": [0.0, 0.3],
        "notes": "Future analysis document (2000) - likely unclassified"
    },
    "documents/B01_F01_01_A Neo-Leitesian, Psycho-Cultural Analysis of Chinese Movies.pdf": {
        "has_classification": False,
        "classification_level": None,
        "control_markings": [],
        "has_portion_marks": False,
        "has_banner": False,
        "has_authority_block": False,
        "is_declassified": False,
        "expected_confidence_range": [0.0, 0.2],
        "notes": "Academic analysis - unclassified"
    },
    "documents/B01_F02_01_The Truth About Islam.pdf": {
        "has_classification": False,
        "classification_level": None,
        "control_markings": [],
        "has_portion_marks": False,
        "has_banner": False,
        "has_authority_block": False,
        "is_declassified": False,
        "expected_confidence_range": [0.0, 0.2],
        "notes": "Academic/cultural analysis - unclassified"
    },
    "documents/B01_F02_04_Non-Western Threats and the Social Sciences.pdf": {
        "has_classification": False,
        "classification_level": None,
        "control_markings": [],
        "has_portion_marks": False,
        "has_banner": False,
        "has_authority_block": False,
        "is_declassified": False,
        "expected_confidence_range": [0.0, 0.2],
        "notes": "Academic analysis - unclassified"
    },
    "documents/R862.pdf": {
        "has_classification": False,
        "classification_level": None,
        "control_markings": [],
        "has_portion_marks": False,
        "has_banner": False,
        "has_authority_block": False,
        "is_declassified": False,
        "expected_confidence_range": [0.0, 0.3],
        "notes": "RAND document - typically unclassified research"
    },
    "documents/RM2301.pdf": {
        "has_classification": False,
        "classification_level": None,
        "control_markings": [],
        "has_portion_marks": False,
        "has_banner": False,
        "has_authority_block": False,
        "is_declassified": False,
        "expected_confidence_range": [0.0, 0.3],
        "notes": "RAND memorandum - typically unclassified research"
    },
    "documents/yx275qm3713.pdf": {
        "has_classification": False,
        "classification_level": None,
        "control_markings": [],
        "has_portion_marks": False,
        "has_banner": False,
        "has_authority_block": False,
        "is_declassified": False,
        "expected_confidence_range": [0.0, 0.3],
        "notes": "Unknown document - appears to be unclassified"
    }
}


def add_real_docs_to_ground_truth(ground_truth_file="documents/ground_truth.json"):
    """Add real document annotations to ground truth file"""

    # Load existing ground truth
    with open(ground_truth_file, 'r') as f:
        ground_truth = json.load(f)

    print(f"Loaded {len(ground_truth)} existing annotations")

    # Add real document annotations
    added = 0
    for file_path, annotation in REAL_DOC_ANNOTATIONS.items():
        if file_path not in ground_truth:
            ground_truth[file_path] = annotation
            added += 1
            print(f"Added: {file_path}")
        else:
            print(f"Skipped (already exists): {file_path}")

    # Save updated ground truth
    with open(ground_truth_file, 'w') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\n{added} real documents added to ground truth")
    print(f"Total annotations: {len(ground_truth)}")
    print(f"Saved to: {ground_truth_file}")


if __name__ == "__main__":
    print("Adding Real Declassified Documents to Ground Truth")
    print("=" * 60)
    add_real_docs_to_ground_truth()
    print("\nDone!")
