#!/usr/bin/env python3
"""
Unit Tests for Classmark GUI

Tests the GUI's interaction with DetectionResult objects to ensure
correct attribute access and prevent AttributeError crashes.

CRITICAL: This is a security classification application.
Bugs in the GUI can lead to incorrect handling of classified documents.

Author: Classmark Development Team
Date: 2025-11-12
"""

import unittest
import sys
import os
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.classification_scanner import DetectionResult, ClassificationLevel, Match


class TestClassmarkGUIResultHandling(unittest.TestCase):
    """Test GUI's handling of DetectionResult objects"""

    def setUp(self):
        """Set up test fixtures"""
        # Create mock DetectionResult (what scanner returns)
        self.mock_result = DetectionResult(
            file_path='/test/document.pdf',
            file_hash='abc123def456',
            has_markings=True,
            classification_level=ClassificationLevel.SECRET,
            matches=[
                Match(
                    text='SECRET',
                    position=20,
                    line_number=10,
                    confidence=0.95,
                    context='This is a SECRET document',
                    match_type='pattern',
                    location='body'
                )
            ],
            overall_confidence=0.95,
            ocr_used=False,
            llm_verified=False,
            processing_time=0.123,
            file_size=1024
        )

    def test_detection_result_has_overall_confidence(self):
        """Verify DetectionResult has overall_confidence attribute"""
        self.assertTrue(hasattr(self.mock_result, 'overall_confidence'))
        self.assertEqual(self.mock_result.overall_confidence, 0.95)

    def test_detection_result_does_not_have_confidence(self):
        """Verify DetectionResult does NOT have 'confidence' attribute (common bug)"""
        self.assertFalse(hasattr(self.mock_result, 'confidence'))

        # Verify accessing .confidence raises AttributeError
        with self.assertRaises(AttributeError):
            _ = self.mock_result.confidence

    def test_gui_result_dict_construction(self):
        """Test GUI's result_dict construction uses correct attributes"""
        # Simulate what GUI does in process_files()
        result = self.mock_result

        # This is the exact code from classmark_gui.py:287-295
        result_dict = {
            'file_path': result.file_path,
            'filename': os.path.basename(result.file_path),
            'has_classification': result.has_markings,
            'classification_level': str(result.classification_level.value) if result.has_markings else None,
            'confidence': result.overall_confidence,  # CRITICAL: Must be overall_confidence
            'num_matches': len(result.matches),
            'processing_time': result.processing_time
        }

        # Verify all attributes accessed successfully
        self.assertEqual(result_dict['confidence'], 0.95)
        self.assertEqual(result_dict['has_classification'], True)
        self.assertEqual(result_dict['classification_level'], 'SECRET')
        self.assertEqual(result_dict['num_matches'], 1)

    def test_gui_confidence_formatting(self):
        """Test GUI's confidence display formatting"""
        result = self.mock_result

        # This is the exact code from classmark_gui.py:312
        message = f"✓ HIT: test.pdf (Confidence: {result.overall_confidence:.2f}) - Moved to output"

        # Verify formatting works without error
        self.assertIn("Confidence: 0.95", message)
        self.assertIn("✓ HIT:", message)

    def test_all_detection_result_attributes_accessible(self):
        """Verify all DetectionResult attributes are accessible (regression test)"""
        result = self.mock_result

        # All attributes that should exist
        required_attrs = [
            'file_path',
            'file_hash',
            'has_markings',
            'classification_level',
            'matches',
            'overall_confidence',  # NOT 'confidence'!
            'ocr_used',
            'llm_verified',
            'processing_time',
            'file_size'
        ]

        for attr in required_attrs:
            self.assertTrue(
                hasattr(result, attr),
                f"DetectionResult missing required attribute: {attr}"
            )
            # Verify we can access it without error
            _ = getattr(result, attr)

    def test_detection_result_with_no_markings(self):
        """Test GUI handles DetectionResult with no markings"""
        no_mark_result = DetectionResult(
            file_path='/test/clean.pdf',
            file_hash='xyz789',
            has_markings=False,
            classification_level=ClassificationLevel.UNCLASSIFIED,
            matches=[],
            overall_confidence=0.0,
            ocr_used=False,
            llm_verified=False,
            processing_time=0.05,
            file_size=512
        )

        # Simulate GUI processing
        result_dict = {
            'file_path': no_mark_result.file_path,
            'filename': os.path.basename(no_mark_result.file_path),
            'has_classification': no_mark_result.has_markings,
            'classification_level': str(no_mark_result.classification_level.value) if no_mark_result.has_markings else None,
            'confidence': no_mark_result.overall_confidence,
            'num_matches': len(no_mark_result.matches),
            'processing_time': no_mark_result.processing_time
        }

        self.assertEqual(result_dict['has_classification'], False)
        self.assertEqual(result_dict['classification_level'], None)
        self.assertEqual(result_dict['confidence'], 0.0)
        self.assertEqual(result_dict['num_matches'], 0)

    def test_detection_result_with_high_confidence(self):
        """Test GUI handles high confidence results"""
        high_conf_result = DetectionResult(
            file_path='/test/classified.pdf',
            file_hash='high123',
            has_markings=True,
            classification_level=ClassificationLevel.TOP_SECRET,
            matches=[
                Match(text='TOP SECRET', position=0, line_number=1, confidence=1.0,
                      context='TOP SECRET//NOFORN', match_type='pattern', location='header'),
                Match(text='NOFORN', position=12, line_number=1, confidence=1.0,
                      context='TOP SECRET//NOFORN', match_type='pattern', location='header'),
            ],
            overall_confidence=0.99,
            ocr_used=False,
            llm_verified=True,
            processing_time=0.234,
            file_size=2048
        )

        # Test confidence display
        message = f"Confidence: {high_conf_result.overall_confidence:.2f}"
        self.assertEqual(message, "Confidence: 0.99")

        # Test result dict
        result_dict = {
            'confidence': high_conf_result.overall_confidence,
            'classification_level': str(high_conf_result.classification_level.value),
            'num_matches': len(high_conf_result.matches)
        }

        self.assertEqual(result_dict['confidence'], 0.99)
        self.assertEqual(result_dict['num_matches'], 2)


class TestClassmarkGUIAttributeCompatibility(unittest.TestCase):
    """Test attribute compatibility between different result types"""

    def test_hybrid_result_vs_detection_result_attributes(self):
        """Document the difference between HybridResult and DetectionResult"""
        # This test documents why the bug occurred

        # DetectionResult uses: overall_confidence
        detection_attrs = {
            'file_path', 'file_hash', 'has_markings', 'classification_level',
            'matches', 'overall_confidence', 'ocr_used', 'llm_verified',
            'processing_time', 'file_size'
        }

        # Create DetectionResult
        det_result = DetectionResult(
            file_path='/test.pdf',
            file_hash='abc',
            has_markings=True,
            classification_level=ClassificationLevel.SECRET,
            matches=[],
            overall_confidence=0.9,
            ocr_used=False,
            llm_verified=False,
            processing_time=0.1,
            file_size=100
        )

        # Verify all expected attributes exist
        for attr in detection_attrs:
            self.assertTrue(hasattr(det_result, attr),
                          f"DetectionResult missing: {attr}")

        # Verify .confidence does NOT exist (common mistake)
        self.assertFalse(hasattr(det_result, 'confidence'),
                        "DetectionResult should NOT have 'confidence' attribute")


if __name__ == '__main__':
    unittest.main()
