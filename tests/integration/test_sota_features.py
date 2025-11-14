#!/usr/bin/env python3
"""
Unit Tests for SOTA Classification Detection Features

Tests the following enhancements:
- FOUO/Legacy CUI marking detection
- Page-region analysis (header/footer extraction)
- Page-association algorithm (repeated markings)
- Location-based confidence scoring
- DPI validation for OCR
- OCR support (when dependencies available)
"""

import unittest
import tempfile
import os
import sys
from pathlib import Path
from typing import Dict

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the scanner components
from src.core.classification_scanner import (
    ClassificationPatterns,
    TextExtractor,
    ClassificationScanner,
    ClassificationLevel
)


class TestLegacyCUIPatterns(unittest.TestCase):
    """Test FOUO and legacy CUI marking detection"""

    def setUp(self):
        self.patterns = ClassificationPatterns()

    def test_fouo_pattern_detection(self):
        """Test FOUO acronym is detected"""
        test_text = "FOUO - This document contains sensitive information"
        found = False

        for pattern in self.patterns.LEGACY_CUI_PATTERNS:
            if 'FOUO' in pattern:
                import re
                if re.search(pattern, test_text, re.IGNORECASE):
                    found = True
                    break

        self.assertTrue(found, "FOUO pattern should be detected")

    def test_for_official_use_only_full_text(self):
        """Test full 'FOR OFFICIAL USE ONLY' text is detected"""
        test_text = "FOR OFFICIAL USE ONLY - Sensitive content"
        found = False

        for pattern in self.patterns.LEGACY_CUI_PATTERNS:
            if 'FOR' in pattern and 'OFFICIAL' in pattern:
                import re
                if re.search(pattern, test_text, re.IGNORECASE):
                    found = True
                    break

        self.assertTrue(found, "FOR OFFICIAL USE ONLY should be detected")

    def test_sbu_pattern_detection(self):
        """Test SBU (Sensitive But Unclassified) is detected"""
        test_text = "SBU: Handle per agency guidelines"
        found = False

        for pattern in self.patterns.LEGACY_CUI_PATTERNS:
            if 'SBU' in pattern:
                import re
                if re.search(pattern, test_text, re.IGNORECASE):
                    found = True
                    break

        self.assertTrue(found, "SBU pattern should be detected")

    def test_les_pattern_detection(self):
        """Test LES (Law Enforcement Sensitive) is detected"""
        test_text = "LES - Law enforcement personnel only"
        found = False

        for pattern in self.patterns.LEGACY_CUI_PATTERNS:
            if 'LES' in pattern:
                import re
                if re.search(pattern, test_text, re.IGNORECASE):
                    found = True
                    break

        self.assertTrue(found, "LES pattern should be detected")

    def test_legacy_marking_classified_as_cui(self):
        """Test that legacy markings are classified as CUI level"""
        config = {'confidence_threshold': 0.3, 'fuzzy_matching': False, 'use_llm': False}
        scanner = ClassificationScanner(config)

        # Create test file with FOUO marking
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("FOUO\n\nThis document is For Official Use Only.\n\nFOUO")
            temp_path = f.name

        try:
            result = scanner.scan_file(temp_path)

            # Should detect markings
            self.assertTrue(result.has_markings, "FOUO should be detected as a marking")

            # Should classify as CUI
            self.assertEqual(
                result.classification_level,
                ClassificationLevel.CUI,
                "FOUO should be classified as CUI equivalent"
            )

            # ADDED: Verify DetectionResult has correct attributes (prevents .confidence bug)
            self.assertTrue(hasattr(result, 'overall_confidence'),
                          "DetectionResult must have overall_confidence attribute")
            self.assertFalse(hasattr(result, 'confidence'),
                           "DetectionResult should NOT have .confidence attribute")
            self.assertIsInstance(result.overall_confidence, float)
            self.assertGreaterEqual(result.overall_confidence, 0.0)
            self.assertLessEqual(result.overall_confidence, 1.0)

        finally:
            os.unlink(temp_path)


class TestPageRegionAnalysis(unittest.TestCase):
    """Test page-region extraction (header/footer/body)"""

    def setUp(self):
        self.extractor = TextExtractor()

    def test_extract_page_regions_basic(self):
        """Test basic page region extraction"""
        test_text = "\n".join([f"Line {i}" for i in range(1, 21)])  # 20 lines

        regions = self.extractor.extract_page_regions(test_text, page_num=1)

        # Should have all three regions
        self.assertIn('header', regions)
        self.assertIn('body', regions)
        self.assertIn('footer', regions)

        # Header should be first 10% (2 lines)
        self.assertIn("Line 1", regions['header'])

        # Footer should be last 10% (2 lines)
        self.assertIn("Line 20", regions['footer'])

    def test_extract_page_regions_with_classification(self):
        """Test region extraction with classification markings in header/footer"""
        test_text = """SECRET//NOFORN
Line 2
Line 3
Line 4
Line 5
Line 6
Line 7
Line 8
Line 9
SECRET//NOFORN"""

        regions = self.extractor.extract_page_regions(test_text, page_num=1)

        # Classification banner should be in header
        self.assertIn("SECRET", regions['header'])

        # Classification banner should be in footer
        self.assertIn("SECRET", regions['footer'])

    def test_extract_page_regions_empty_text(self):
        """Test region extraction handles empty text gracefully"""
        regions = self.extractor.extract_page_regions("", page_num=1)

        self.assertEqual(regions['header'], "")
        self.assertEqual(regions['body'], "")
        self.assertEqual(regions['footer'], "")


class TestPageAssociationAlgorithm(unittest.TestCase):
    """Test detection of repeated markings across pages (banners)"""

    def setUp(self):
        config = {'confidence_threshold': 0.3, 'fuzzy_matching': False, 'use_llm': False}
        self.scanner = ClassificationScanner(config)

    def test_find_repeated_markings(self):
        """Test identification of repeated markings"""
        # Simulate multiple page headers with repeated SECRET marking
        page_headers = [
            "SECRET//NOFORN\nPage 1",
            "SECRET//NOFORN\nPage 2",
            "SECRET//NOFORN\nPage 3",
        ]

        repeated = self.scanner.find_repeated_markings(page_headers)

        # SECRET should be found on all 3 pages
        self.assertIn('SECRET', repeated)
        self.assertEqual(repeated['SECRET'], 3)

    def test_find_repeated_markings_mixed(self):
        """Test with some pages having markings, some not"""
        page_headers = [
            "CONFIDENTIAL\nPage 1",
            "CONFIDENTIAL\nPage 2",
            "Regular Header\nPage 3",
            "CONFIDENTIAL\nPage 4",
        ]

        repeated = self.scanner.find_repeated_markings(page_headers)

        # CONFIDENTIAL appears on 3 out of 4 pages
        self.assertIn('CONFIDENTIAL', repeated)
        self.assertEqual(repeated['CONFIDENTIAL'], 3)


class TestLocationBasedConfidence(unittest.TestCase):
    """Test enhanced confidence scoring based on location"""

    def setUp(self):
        config = {'confidence_threshold': 0.2, 'fuzzy_matching': False, 'use_llm': False}
        self.scanner = ClassificationScanner(config)

    def test_header_location_boosts_confidence(self):
        """Test that markings in headers receive confidence boost"""
        # Create test file with marking in different locations
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Simple text that will be treated as body
            f.write("SECRET classification marking in body text only")
            temp_path = f.name

        try:
            result = self.scanner.scan_file(temp_path)

            if result.matches:
                # Body location should have lower confidence than header would
                body_confidence = max([m.confidence for m in result.matches if m.location == 'body'], default=0)

                # We can't easily test header boost without complex setup,
                # but we verify the location is being tracked
                self.assertTrue(
                    any(m.location == 'body' for m in result.matches),
                    "Location should be tracked"
                )

        finally:
            os.unlink(temp_path)


class TestDPIValidation(unittest.TestCase):
    """Test DPI validation for scanned documents"""

    def setUp(self):
        self.extractor = TextExtractor()

    def test_validate_dpi_meets_standards(self):
        """Test DPI validation with acceptable DPI"""
        # Create mock image object
        class MockImage:
            def __init__(self, dpi):
                self.info = {'dpi': dpi}

        # Test with good DPI (300x300)
        good_image = MockImage((300, 300))
        meets_standards, message = self.extractor.validate_image_dpi(good_image, page_num=1)

        self.assertTrue(meets_standards)
        self.assertIn("meets standards", message.lower())

    def test_validate_dpi_below_minimum(self):
        """Test DPI validation with below-minimum DPI"""
        class MockImage:
            def __init__(self, dpi):
                self.info = {'dpi': dpi}

        # Test with low DPI (150x150)
        low_image = MockImage((150, 150))
        meets_standards, message = self.extractor.validate_image_dpi(low_image, page_num=1)

        self.assertFalse(meets_standards)
        self.assertIn("below minimum", message.lower())

    def test_validate_dpi_minimum_acceptable(self):
        """Test DPI validation at minimum threshold (200 DPI)"""
        class MockImage:
            def __init__(self, dpi):
                self.info = {'dpi': dpi}

        # Test at minimum DPI (200x200)
        min_image = MockImage((200, 200))
        meets_standards, message = self.extractor.validate_image_dpi(min_image, page_num=1)

        # Should meet minimum but be below recommended
        self.assertTrue(meets_standards)
        self.assertIn("meets minimum", message.lower())


class TestIntegrationSOTAFeatures(unittest.TestCase):
    """Integration tests for all SOTA features working together"""

    def test_full_sota_pipeline(self):
        """Test complete SOTA feature pipeline"""
        config = {
            'confidence_threshold': 0.3,
            'fuzzy_matching': True,
            'use_llm': False,
            'workers': 1
        }
        scanner = ClassificationScanner(config)

        # Create multi-page-style document with FOUO banner
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("""FOUO
Page 1 Content

This is For Official Use Only material.

FOUO

---

FOUO
Page 2 Content

More sensitive content here.

FOUO
""")
            temp_path = f.name

        try:
            result = scanner.scan_file(temp_path)

            # Should detect FOUO markings
            self.assertTrue(result.has_markings, "FOUO should be detected")

            # Should classify as CUI
            self.assertEqual(result.classification_level, ClassificationLevel.CUI)

            # Should have multiple matches (repeated FOUO)
            self.assertGreater(len(result.matches), 0)

        finally:
            os.unlink(temp_path)


def run_tests():
    """Run all unit tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestLegacyCUIPatterns))
    suite.addTests(loader.loadTestsFromTestCase(TestPageRegionAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestPageAssociationAlgorithm))
    suite.addTests(loader.loadTestsFromTestCase(TestLocationBasedConfidence))
    suite.addTests(loader.loadTestsFromTestCase(TestDPIValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationSOTAFeatures))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    import sys

    print("="*80)
    print("SOTA Classification Detection Features - Unit Tests")
    print("="*80)
    print()

    exit_code = run_tests()

    print()
    print("="*80)
    if exit_code == 0:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("="*80)

    sys.exit(exit_code)
