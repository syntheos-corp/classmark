#!/usr/bin/env python3
"""
Test script for Classification Scanner
Creates sample documents with various classification markings to verify detection
"""

import os
import tempfile
import shutil
from pathlib import Path

def create_test_documents():
    """Create test documents with various classification markings"""

    test_dir = tempfile.mkdtemp(prefix='classification_test_')
    print(f"Creating test documents in: {test_dir}")

    # Test 1: Plain text with TOP SECRET marking
    with open(os.path.join(test_dir, 'test_top_secret.txt'), 'w') as f:
        f.write("""
TOP SECRET//NOFORN

This is a test document with classification markings.

CLASSIFIED BY: John Doe
REASON: 1.4(c)
DECLASSIFY ON: 20351231

(TS) This paragraph contains top secret information.
(S) This paragraph contains secret information.
(U) This paragraph is unclassified.

TOP SECRET//NOFORN
""")

    # Test 2: Plain text with SECRET marking
    with open(os.path.join(test_dir, 'test_secret.txt'), 'w') as f:
        f.write("""
SECRET

CLASSIFIED BY: Jane Smith
DERIVED FROM: Multiple Sources
DECLASSIFY ON: 20301231

(S) This document contains secret information about operations.

SECRET
""")

    # Test 3: Plain text with CONFIDENTIAL marking
    with open(os.path.join(test_dir, 'test_confidential.txt'), 'w') as f:
        f.write("""
CONFIDENTIAL//NOFORN

This document is marked confidential.

CLASSIFIED BY: Security Officer
REASON: 1.4(a)
DECLASSIFY ON: 20281231

CONFIDENTIAL//NOFORN
""")

    # Test 4: CUI marking
    with open(os.path.join(test_dir, 'test_cui.txt'), 'w') as f:
        f.write("""
CONTROLLED UNCLASSIFIED INFORMATION

This document contains CUI material.

CUI//SP-PRVCY

CONTROLLED UNCLASSIFIED INFORMATION
""")

    # Test 5: False positive test (should NOT be flagged)
    with open(os.path.join(test_dir, 'test_false_positive.txt'), 'w') as f:
        f.write("""
The Secret to Success

This article discusses the secret to success in business.
Victoria's Secret is a well-known retailer.

Data can be classified as sensitive or non-sensitive.
This is a confidential conversation between friends.

The agent kept the information secret from the public.
""")

    # Test 6: Clean document (should NOT be flagged)
    with open(os.path.join(test_dir, 'test_clean.txt'), 'w') as f:
        f.write("""
Unclassified Document

This is a completely unclassified document with no security markings.
It contains general information that is publicly available.

Lorem ipsum dolor sit amet, consectetur adipiscing elit.
""")

    # Test 7: Mixed content (partial marking - edge case)
    with open(os.path.join(test_dir, 'test_mixed.txt'), 'w') as f:
        f.write("""
Some regular text here.

Then suddenly:

SECRET//REL TO USA, AUS

More classified content here.

CLASSIFIED BY: Test User
REASON: 1.4(d)

SECRET//REL TO USA, AUS
""")

    print(f"\nCreated {len(os.listdir(test_dir))} test documents:")
    for filename in sorted(os.listdir(test_dir)):
        print(f"  - {filename}")

    return test_dir


def run_tests():
    """Run scanner on test documents"""
    import subprocess
    import sys

    test_dir = None
    try:
        # Create test documents
        test_dir = create_test_documents()

        print("\n" + "="*80)
        print("TEST 1: Basic scan (pattern matching only)")
        print("="*80)

        result = subprocess.run([
            sys.executable,
            'classification_scanner.py',
            test_dir,
            '--no-recursive',
            '--show-all'
        ], capture_output=False)

        print("\n" + "="*80)
        print("TEST 2: High recall mode with fuzzy matching")
        print("="*80)

        result = subprocess.run([
            sys.executable,
            'classification_scanner.py',
            test_dir,
            '--sensitivity', 'high',
            '--fuzzy-matching',
            '--show-all'
        ], capture_output=False)

        print("\n" + "="*80)
        print("TEST 3: JSON report generation")
        print("="*80)

        json_output = os.path.join(test_dir, 'test_report.json')
        csv_output = os.path.join(test_dir, 'test_report.csv')

        result = subprocess.run([
            sys.executable,
            'classification_scanner.py',
            test_dir,
            '--output', json_output,
            '--csv', csv_output
        ], capture_output=False)

        if os.path.exists(json_output):
            print(f"\n✓ JSON report created: {json_output}")
            with open(json_output, 'r') as f:
                import json
                report = json.load(f)
                print(f"  Flagged files: {report['scan_summary']['flagged_files']}")

        if os.path.exists(csv_output):
            print(f"✓ CSV report created: {csv_output}")

        print("\n" + "="*80)
        print("EXPECTED RESULTS:")
        print("="*80)
        print("Should FLAG (with high confidence):")
        print("  ✓ test_top_secret.txt - TOP SECRET")
        print("  ✓ test_secret.txt - SECRET")
        print("  ✓ test_confidential.txt - CONFIDENTIAL")
        print("  ✓ test_cui.txt - CUI")
        print("  ✓ test_mixed.txt - SECRET")
        print("\nShould NOT flag:")
        print("  ✓ test_false_positive.txt - Casual usage of words")
        print("  ✓ test_clean.txt - No markings")
        print("="*80)

    finally:
        # Cleanup
        if test_dir and os.path.exists(test_dir):
            response = input(f"\nDelete test directory {test_dir}? [y/N]: ")
            if response.lower() == 'y':
                shutil.rmtree(test_dir)
                print(f"Deleted: {test_dir}")
            else:
                print(f"Test documents preserved at: {test_dir}")


if __name__ == '__main__':
    print("Classification Scanner Test Suite")
    print("="*80)

    # Check if scanner exists
    if not os.path.exists('classification_scanner.py'):
        print("Error: classification_scanner.py not found in current directory")
        print("Please run this test script from the same directory as classification_scanner.py")
        exit(1)

    run_tests()
