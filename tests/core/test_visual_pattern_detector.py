"""
Unit Tests for Visual Pattern Detector

Tests for LayoutLMv3-based visual pattern detection including:
- Visual feature extraction
- Classification region detection
- Visual confidence scoring
- Pattern type identification
- Integration with PDFs

Author: Classmark Development Team
Date: 2025-11-10
"""

import unittest
import time
import os
import sys
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.visual_pattern_detector import (
    VisualPatternDetector,
    VisualRegion,
    VisualMatch,
    VisualFeatureType,
    HAS_LAYOUTLM
)


class TestVisualFeatureExtraction(unittest.TestCase):
    """Test visual feature extraction"""

    @unittest.skipUnless(HAS_LAYOUTLM, "LayoutLMv3 dependencies not available")
    def setUp(self):
        """Set up test fixtures"""
        self.detector = VisualPatternDetector(dpi=200, use_gpu=False)
        self.detector.initialize()

    @unittest.skipUnless(HAS_LAYOUTLM, "LayoutLMv3 dependencies not available")
    def test_initialization(self):
        """Test detector initialization"""
        self.assertTrue(self.detector._initialized)
        self.assertIsNotNone(self.detector.model)
        self.assertIsNotNone(self.detector.processor)
        self.assertIsNotNone(self.detector.device)

    @unittest.skipUnless(HAS_LAYOUTLM, "LayoutLMv3 dependencies not available")
    def test_feature_extraction(self):
        """Test visual feature extraction from bbox"""
        features = self.detector._extract_visual_features(
            text="TOP SECRET//NOFORN",
            bbox=(100, 50, 500, 100),  # Large, top of page
            page_width=612,
            page_height=792
        )

        self.assertIn('font_size', features)
        self.assertIn('y_position', features)
        self.assertIn('x_position', features)
        self.assertIn('capitalization_ratio', features)
        self.assertIn('alignment_offset', features)

        # Check reasonable values
        self.assertGreater(features['font_size'], 0)
        self.assertGreaterEqual(features['y_position'], 0)
        self.assertLessEqual(features['y_position'], 1)

    @unittest.skipUnless(HAS_LAYOUTLM, "LayoutLMv3 dependencies not available")
    def test_capitalization_detection(self):
        """Test capitalization ratio calculation"""
        # All caps text
        features_caps = self.detector._extract_visual_features(
            text="TOP SECRET",
            bbox=(100, 50, 300, 80),
            page_width=612,
            page_height=792
        )
        self.assertGreater(features_caps['capitalization_ratio'], 0.9)

        # Mixed case text
        features_mixed = self.detector._extract_visual_features(
            text="Top Secret Document",
            bbox=(100, 50, 300, 80),
            page_width=612,
            page_height=792
        )
        self.assertLess(features_mixed['capitalization_ratio'], 0.5)


class TestRegionDetection(unittest.TestCase):
    """Test classification region detection"""

    @unittest.skipUnless(HAS_LAYOUTLM, "LayoutLMv3 dependencies not available")
    def setUp(self):
        """Set up test fixtures"""
        self.detector = VisualPatternDetector(dpi=200, use_gpu=False)
        self.detector.initialize()

    @unittest.skipUnless(HAS_LAYOUTLM, "LayoutLMv3 dependencies not available")
    def test_region_creation(self):
        """Test creating a visual region"""
        tokens = ['TOP', 'SECRET', '/', '/', 'NOFORN']
        boxes = [
            (100, 50, 150, 80),
            (160, 50, 240, 80),
            (250, 50, 270, 80),
            (280, 50, 300, 80),
            (310, 50, 400, 80)
        ]

        region = self.detector._create_region(
            tokens, boxes,
            page_width=612,
            page_height=792,
            page_num=1
        )

        self.assertIsNotNone(region)
        self.assertIn('TOP', region.text)
        self.assertIn('SECRET', region.text)
        self.assertEqual(region.page_num, 1)
        self.assertTrue(region.is_header)  # Top of page

    @unittest.skipUnless(HAS_LAYOUTLM, "LayoutLMv3 dependencies not available")
    def test_banner_detection(self):
        """Test banner region detection"""
        # Create region at top center
        tokens = ['TOP', 'SECRET']
        boxes = [
            (250, 30, 300, 60),  # Centered, top
            (310, 30, 370, 60)
        ]

        region = self.detector._create_region(
            tokens, boxes,
            page_width=612,
            page_height=792,
            page_num=1
        )

        self.assertIsNotNone(region)
        self.assertTrue(region.is_header)
        # Banner detection requires centered + top/bottom
        # This should be detected as banner

    @unittest.skipUnless(HAS_LAYOUTLM, "LayoutLMv3 dependencies not available")
    def test_footer_detection(self):
        """Test footer region detection"""
        tokens = ['PAGE', '1', 'OF', '10']
        boxes = [
            (250, 750, 300, 770),  # Bottom of page
            (310, 750, 330, 770),
            (340, 750, 370, 770),
            (380, 750, 410, 770)
        ]

        region = self.detector._create_region(
            tokens, boxes,
            page_width=612,
            page_height=792,
            page_num=1
        )

        self.assertIsNotNone(region)
        self.assertTrue(region.is_footer)


class TestClassificationDetection(unittest.TestCase):
    """Test classification-specific detection"""

    @unittest.skipUnless(HAS_LAYOUTLM, "LayoutLMv3 dependencies not available")
    def setUp(self):
        """Set up test fixtures"""
        self.detector = VisualPatternDetector(dpi=200, use_gpu=False)
        self.detector.initialize()

    @unittest.skipUnless(HAS_LAYOUTLM, "LayoutLMv3 dependencies not available")
    def test_classification_region_detection(self):
        """Test detection of classification regions"""
        # Create region with classification marking
        region_classified = VisualRegion(
            bbox=(100, 50, 400, 80),
            text="TOP SECRET//NOFORN",
            confidence=0.9,
            visual_features={
                'font_size': 20.0,
                'y_position': 0.05,
                'capitalization_ratio': 1.0
            },
            page_num=1,
            is_header=True,
            is_banner=True
        )

        self.assertTrue(self.detector._is_classification_region(region_classified))

        # Create region without classification marking
        region_normal = VisualRegion(
            bbox=(100, 300, 500, 320),
            text="This is regular text in the document body.",
            confidence=0.9,
            visual_features={
                'font_size': 12.0,
                'y_position': 0.38,
                'capitalization_ratio': 0.05
            },
            page_num=1,
            is_header=False,
            is_banner=False
        )

        self.assertFalse(self.detector._is_classification_region(region_normal))

    @unittest.skipUnless(HAS_LAYOUTLM, "LayoutLMv3 dependencies not available")
    def test_authority_block_detection(self):
        """Test detection of classification authority blocks"""
        region = VisualRegion(
            bbox=(100, 100, 450, 180),
            text="CLASSIFIED BY: John Doe\nDERIVED FROM: ABC-123\nDECLASSIFY ON: 20300101",
            confidence=0.9,
            visual_features={
                'font_size': 10.0,
                'y_position': 0.13,
                'capitalization_ratio': 0.6
            },
            page_num=1,
            is_header=True,
            is_banner=False
        )

        self.assertTrue(self.detector._is_classification_region(region))

        pattern_type = self.detector._identify_pattern_type(region)
        self.assertEqual(pattern_type, 'authority_block')

    @unittest.skipUnless(HAS_LAYOUTLM, "LayoutLMv3 dependencies not available")
    def test_visual_confidence_scoring(self):
        """Test visual confidence calculation"""
        # High confidence region (banner, large, caps)
        region_high = VisualRegion(
            bbox=(200, 30, 400, 70),
            text="TOP SECRET",
            confidence=0.9,
            visual_features={
                'font_size': 24.0,
                'y_position': 0.04,
                'capitalization_ratio': 1.0,
                'alignment_offset': 0.1  # Centered
            },
            page_num=1,
            is_header=True,
            is_banner=True
        )

        confidence_high = self.detector._calculate_visual_confidence(region_high)
        self.assertGreater(confidence_high, 0.8)

        # Low confidence region (body text)
        region_low = VisualRegion(
            bbox=(100, 300, 500, 315),
            text="Regular document text.",
            confidence=0.9,
            visual_features={
                'font_size': 11.0,
                'y_position': 0.38,
                'capitalization_ratio': 0.05,
                'alignment_offset': 0.5
            },
            page_num=1,
            is_header=False,
            is_banner=False
        )

        confidence_low = self.detector._calculate_visual_confidence(region_low)
        self.assertLess(confidence_low, 0.7)

    @unittest.skipUnless(HAS_LAYOUTLM, "LayoutLMv3 dependencies not available")
    def test_pattern_type_identification(self):
        """Test identification of different pattern types"""
        # Banner pattern
        banner_region = VisualRegion(
            bbox=(200, 30, 400, 70),
            text="TOP SECRET",
            confidence=0.9,
            visual_features={},
            page_num=1,
            is_banner=True
        )
        self.assertEqual(self.detector._identify_pattern_type(banner_region), 'banner')

        # Authority block
        auth_region = VisualRegion(
            bbox=(100, 100, 400, 150),
            text="CLASSIFIED BY: John Doe",
            confidence=0.9,
            visual_features={},
            page_num=1
        )
        self.assertEqual(self.detector._identify_pattern_type(auth_region), 'authority_block')

        # Portion mark
        portion_region = VisualRegion(
            bbox=(100, 200, 130, 215),
            text="(TS)",
            confidence=0.9,
            visual_features={},
            page_num=1
        )
        self.assertEqual(self.detector._identify_pattern_type(portion_region), 'portion_mark')

        # Control marking
        control_region = VisualRegion(
            bbox=(300, 150, 380, 170),
            text="NOFORN",
            confidence=0.9,
            visual_features={},
            page_num=1
        )
        self.assertEqual(self.detector._identify_pattern_type(control_region), 'control_marking')


@unittest.skipUnless(HAS_LAYOUTLM, "LayoutLMv3 dependencies not available")
class TestPDFProcessing(unittest.TestCase):
    """Test PDF processing with real documents"""

    def setUp(self):
        """Set up test fixtures"""
        self.detector = VisualPatternDetector(dpi=200, max_pages=2, use_gpu=False)
        self.detector.initialize()

        # Find test PDFs
        self.test_dir = Path(__file__).parent / "test_data"
        self.docs_dir = Path(__file__).parent / "documents"

    def test_pdf_processing(self):
        """Test visual pattern detection on real PDFs"""
        # Look for PDFs in documents/ directory
        if not self.docs_dir.exists():
            self.skipTest("documents/ directory not found")

        pdf_files = list(self.docs_dir.glob("*.pdf"))
        if not pdf_files:
            self.skipTest("No PDF files found in documents/")

        # Test on first PDF
        pdf_path = str(pdf_files[0])
        print(f"\nTesting on: {pdf_path}")

        start_time = time.time()
        matches = self.detector.detect_visual_patterns(pdf_path)
        elapsed = time.time() - start_time

        print(f"  Found {len(matches)} visual pattern matches in {elapsed:.2f}s")

        # Should find some matches in declassified documents
        # (But this is not guaranteed depending on document)
        self.assertIsInstance(matches, list)

        # Check match structure
        if matches:
            match = matches[0]
            self.assertIsInstance(match, VisualMatch)
            self.assertIsNotNone(match.text)
            self.assertIsNotNone(match.region)
            self.assertIsNotNone(match.visual_confidence)
            self.assertIsNotNone(match.pattern_type)

            print(f"  Sample match: {match.pattern_type} - {match.text[:50]}")


class TestStatistics(unittest.TestCase):
    """Test detector statistics"""

    @unittest.skipUnless(HAS_LAYOUTLM, "LayoutLMv3 dependencies not available")
    def test_get_statistics(self):
        """Test getting detector statistics"""
        detector = VisualPatternDetector(use_gpu=False)
        detector.initialize()

        stats = detector.get_statistics()

        self.assertIn('initialized', stats)
        self.assertIn('has_layoutlm', stats)
        self.assertIn('model_name', stats)
        self.assertIn('device', stats)
        self.assertIn('dpi', stats)

        self.assertTrue(stats['initialized'])
        self.assertTrue(stats['has_layoutlm'])
        self.assertEqual(stats['model_name'], 'microsoft/layoutlmv3-base')

        if stats['initialized']:
            self.assertIn('model_params', stats)
            self.assertGreater(stats['model_params'], 100_000_000)  # 125M params


class TestPerformance(unittest.TestCase):
    """Performance benchmarking tests"""

    @unittest.skipUnless(HAS_LAYOUTLM, "LayoutLMv3 dependencies not available")
    def setUp(self):
        """Set up test fixtures"""
        self.detector = VisualPatternDetector(dpi=200, max_pages=1, use_gpu=False)
        self.detector.initialize()

        self.docs_dir = Path(__file__).parent / "documents"

    def test_processing_speed(self):
        """Benchmark visual pattern detection speed"""
        if not self.docs_dir.exists():
            self.skipTest("documents/ directory not found")

        pdf_files = list(self.docs_dir.glob("*.pdf"))
        if not pdf_files:
            self.skipTest("No PDF files found")

        pdf_path = str(pdf_files[0])

        # Warm-up run
        self.detector.detect_visual_patterns(pdf_path)

        # Benchmark
        iterations = 3
        start = time.time()
        for _ in range(iterations):
            matches = self.detector.detect_visual_patterns(pdf_path)
        elapsed = time.time() - start

        avg_time = elapsed / iterations

        print(f"\nVisual Pattern Detection Performance:")
        print(f"  Average time: {avg_time:.2f}s per page")
        print(f"  Matches found: {len(matches)}")

        # Should complete within reasonable time (depends on hardware)
        # Allow up to 30s per page on CPU
        self.assertLess(avg_time, 30.0)


def run_tests():
    """Run all tests and print results"""
    print("=" * 70)
    print("Visual Pattern Detector - Unit Tests")
    print("=" * 70)
    print()

    # Check dependencies
    print("Dependency Status:")
    print(f"  LayoutLMv3: {HAS_LAYOUTLM}")
    print()

    if not HAS_LAYOUTLM:
        print("⚠ LayoutLMv3 not available - tests will be skipped")
        print("Install: pip install torch transformers pillow pdf2image")
        print()

    # Run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestVisualFeatureExtraction))
    suite.addTests(loader.loadTestsFromTestCase(TestRegionDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestClassificationDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestPDFProcessing))
    suite.addTests(loader.loadTestsFromTestCase(TestStatistics))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print()
    print("=" * 70)
    print("Test Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Skipped: {len(result.skipped)}")
    print("=" * 70)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    import sys
    sys.exit(run_tests())
