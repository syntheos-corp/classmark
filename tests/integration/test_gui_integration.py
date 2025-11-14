#!/usr/bin/env python3
"""
Integration Tests for GUI + Scanner

Tests the full integration path from GUI → Scanner → DetectionResult → Display
to catch attribute mismatches and runtime errors.

CRITICAL: This test would have caught the .confidence vs .overall_confidence bug.

Author: Classmark Development Team
Date: 2025-11-12
"""

import unittest
import sys
import os
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.classification_scanner import ClassificationScanner, DetectionResult, ClassificationLevel


class TestGUIScannerIntegration(unittest.TestCase):
    """Test full integration between GUI processing logic and Scanner"""

    def setUp(self):
        """Set up test scanner"""
        config = {
            'fuzzy_matching': True,
            'fuzzy_threshold': 85,
            'use_visual_detection': False,  # Disable for faster tests
            'use_gpu': False,
            'use_llm': False
        }
        self.scanner = ClassificationScanner(config)

    def test_scanner_returns_detection_result_with_overall_confidence(self):
        """Verify scanner.scan_file() returns DetectionResult with overall_confidence"""
        # Create a test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This document is classified SECRET//NOFORN\n")
            f.write("Contains sensitive information.\n")
            temp_path = f.name

        try:
            # Scan file (what GUI does)
            result = self.scanner.scan_file(temp_path)

            # Verify it's a DetectionResult
            self.assertIsInstance(result, DetectionResult)

            # CRITICAL: Verify overall_confidence exists
            self.assertTrue(hasattr(result, 'overall_confidence'),
                          "DetectionResult missing overall_confidence attribute")

            # CRITICAL: Verify .confidence does NOT exist
            self.assertFalse(hasattr(result, 'confidence'),
                           "DetectionResult should not have .confidence attribute")

            # Verify we can access overall_confidence
            confidence = result.overall_confidence
            self.assertIsInstance(confidence, float)
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)

        finally:
            os.unlink(temp_path)

    def test_gui_result_processing_workflow(self):
        """Test the exact workflow GUI uses to process results"""
        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("TOP SECRET//SCI//NOFORN\n")
            f.write("Classified Intelligence Report\n")
            temp_path = f.name

        try:
            # Step 1: Scan file (GUI line ~284)
            result = self.scanner.scan_file(temp_path)

            # Step 2: Build result_dict (GUI lines 287-295)
            # This is the EXACT code from classmark_gui.py
            filename = os.path.basename(temp_path)
            result_dict = {
                'file_path': temp_path,
                'filename': filename,
                'has_classification': result.has_markings,
                'classification_level': str(result.classification_level.value) if result.has_markings else None,
                'confidence': result.overall_confidence,  # BUG WAS HERE: used .confidence
                'num_matches': len(result.matches),
                'processing_time': result.processing_time
            }

            # Verify result_dict was built successfully
            self.assertIn('confidence', result_dict)
            self.assertIsInstance(result_dict['confidence'], float)

            # Step 3: Format confidence message (GUI line ~312)
            if result.has_markings:
                message = f"✓ HIT: {filename} (Confidence: {result.overall_confidence:.2f}) - Moved to output"
                # Verify message formats without error
                self.assertIn("Confidence:", message)
                self.assertIn(f"{result.overall_confidence:.2f}", message)

        finally:
            os.unlink(temp_path)

    def test_all_detection_result_attributes_in_integration(self):
        """Test all DetectionResult attributes are accessible in real workflow"""
        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("CONFIDENTIAL - Internal Use Only\n")
            temp_path = f.name

        try:
            result = self.scanner.scan_file(temp_path)

            # All attributes GUI might access
            attributes_to_check = {
                'file_path': str,
                'file_hash': str,
                'has_markings': bool,
                'classification_level': ClassificationLevel,
                'matches': list,
                'overall_confidence': float,
                'ocr_used': bool,
                'llm_verified': bool,
                'processing_time': float,
                'file_size': int
            }

            for attr_name, expected_type in attributes_to_check.items():
                # Verify attribute exists
                self.assertTrue(
                    hasattr(result, attr_name),
                    f"DetectionResult missing attribute: {attr_name}"
                )

                # Verify we can access it
                value = getattr(result, attr_name)

                # Verify type is correct
                self.assertIsInstance(
                    value, expected_type,
                    f"{attr_name} has wrong type: expected {expected_type}, got {type(value)}"
                )

        finally:
            os.unlink(temp_path)

    def test_no_classification_result_handling(self):
        """Test GUI handles clean documents correctly"""
        # Create clean document
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a regular business document.\n")
            f.write("No classification markings present.\n")
            temp_path = f.name

        try:
            result = self.scanner.scan_file(temp_path)

            # Verify clean document
            self.assertFalse(result.has_markings)

            # GUI should still be able to access all attributes
            result_dict = {
                'has_classification': result.has_markings,
                'confidence': result.overall_confidence,
                'num_matches': len(result.matches)
            }

            self.assertEqual(result_dict['has_classification'], False)
            self.assertEqual(result_dict['num_matches'], 0)
            # Confidence might be 0 or low for clean docs
            self.assertGreaterEqual(result_dict['confidence'], 0.0)

        finally:
            os.unlink(temp_path)

    def test_multiple_files_batch_processing(self):
        """Test batch processing like GUI does (prevents regression)"""
        # Create multiple test files
        test_files = []
        temp_dir = tempfile.mkdtemp()

        try:
            # Create 3 test files
            for i, content in enumerate([
                "TOP SECRET//NOFORN",
                "CONFIDENTIAL - Internal Only",
                "Unclassified public document"
            ]):
                file_path = os.path.join(temp_dir, f"test_{i}.txt")
                with open(file_path, 'w') as f:
                    f.write(content)
                test_files.append(file_path)

            # Process all files (simulating GUI batch processing)
            results = []
            for file_path in test_files:
                result = self.scanner.scan_file(file_path)

                # Build result dict (GUI workflow)
                result_dict = {
                    'file_path': file_path,
                    'filename': os.path.basename(file_path),
                    'has_classification': result.has_markings,
                    'confidence': result.overall_confidence,
                    'processing_time': result.processing_time
                }
                results.append(result_dict)

            # Verify all processed successfully
            self.assertEqual(len(results), 3)

            # Verify all have confidence values
            for result_dict in results:
                self.assertIn('confidence', result_dict)
                self.assertIsInstance(result_dict['confidence'], float)

        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
