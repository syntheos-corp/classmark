"""
Unit Tests for YOLO OCR Module

Tests for YOLOv8 + Tesseract OCR functionality including:
- Text region detection
- OCR extraction
- DPI validation
- Selective processing
- Fallback pipeline

Author: Classmark Development Team
Date: 2025-11-10
"""

import unittest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Check if dependencies are available
try:
    from src.core.yolo_ocr import (
        YOLOTextDetector,
        TextRegion,
        OCRResult,
        extract_text_with_yolo_ocr,
        HAS_OCR_BASE,
        HAS_YOLO
    )
    YOLO_OCR_AVAILABLE = True
except ImportError:
    YOLO_OCR_AVAILABLE = False
    print("Warning: yolo_ocr module not available for testing")


@unittest.skipUnless(YOLO_OCR_AVAILABLE, "YOLO OCR module not available")
class TestTextRegion(unittest.TestCase):
    """Test TextRegion dataclass"""

    def test_text_region_creation(self):
        """Test creating a TextRegion"""
        region = TextRegion(
            bbox=(100, 100, 200, 200),
            confidence=0.95,
            text="TEST TEXT"
        )

        self.assertEqual(region.bbox, (100, 100, 200, 200))
        self.assertEqual(region.confidence, 0.95)
        self.assertEqual(region.text, "TEST TEXT")

    def test_text_region_without_text(self):
        """Test TextRegion without text (before OCR)"""
        region = TextRegion(
            bbox=(50, 50, 150, 150),
            confidence=0.85
        )

        self.assertIsNone(region.text)


@unittest.skipUnless(YOLO_OCR_AVAILABLE, "YOLO OCR module not available")
class TestOCRResult(unittest.TestCase):
    """Test OCRResult dataclass"""

    def test_ocr_result_creation(self):
        """Test creating an OCRResult"""
        regions = [
            TextRegion(bbox=(0, 0, 100, 100), confidence=0.9, text="Line 1"),
            TextRegion(bbox=(0, 100, 100, 200), confidence=0.85, text="Line 2")
        ]

        result = OCRResult(
            page_num=1,
            text="Line 1\nLine 2",
            regions=regions,
            processing_time=0.5,
            dpi=(300, 300),
            meets_dpi_standard=True
        )

        self.assertEqual(result.page_num, 1)
        self.assertEqual(result.text, "Line 1\nLine 2")
        self.assertEqual(len(result.regions), 2)
        self.assertEqual(result.processing_time, 0.5)
        self.assertEqual(result.dpi, (300, 300))
        self.assertTrue(result.meets_dpi_standard)


@unittest.skipUnless(HAS_OCR_BASE and HAS_YOLO, "YOLO and OCR dependencies not available")
class TestYOLOTextDetector(unittest.TestCase):
    """Test YOLOTextDetector class"""

    def setUp(self):
        """Set up test fixtures"""
        # Mock YOLO model to avoid downloading
        with patch('yolo_ocr.YOLO') as mock_yolo:
            mock_model = MagicMock()
            mock_yolo.return_value = mock_model
            self.detector = YOLOTextDetector(use_gpu=False)

    def test_initialization(self):
        """Test YOLOTextDetector initialization"""
        self.assertEqual(self.detector.dpi_for_conversion, 300)
        self.assertEqual(self.detector.min_dpi_standard, 200)
        self.assertEqual(self.detector.tesseract_config, '--psm 6')

    def test_custom_initialization(self):
        """Test YOLOTextDetector with custom parameters"""
        with patch('yolo_ocr.YOLO'):
            detector = YOLOTextDetector(
                dpi_for_conversion=200,
                min_dpi_standard=150,
                tesseract_config='--psm 3',
                use_gpu=False
            )

            self.assertEqual(detector.dpi_for_conversion, 200)
            self.assertEqual(detector.min_dpi_standard, 150)
            self.assertEqual(detector.tesseract_config, '--psm 3')

    @patch('yolo_ocr.Image')
    def test_validate_dpi_meets_standard(self, mock_image):
        """Test DPI validation when DPI meets standard"""
        # Create mock image with good DPI
        mock_img = MagicMock()
        mock_img.info = {'dpi': (300, 300)}

        meets_standard, dpi, message = self.detector.validate_dpi(mock_img)

        self.assertTrue(meets_standard)
        self.assertEqual(dpi, (300, 300))
        self.assertIn("excellent quality", message)

    @patch('yolo_ocr.Image')
    def test_validate_dpi_below_standard(self, mock_image):
        """Test DPI validation when DPI is below standard"""
        # Create mock image with low DPI
        mock_img = MagicMock()
        mock_img.info = {'dpi': (150, 150)}

        meets_standard, dpi, message = self.detector.validate_dpi(mock_img)

        self.assertFalse(meets_standard)
        self.assertEqual(dpi, (150, 150))
        self.assertIn("below minimum", message)

    @patch('yolo_ocr.Image')
    def test_validate_dpi_no_dpi_info(self, mock_image):
        """Test DPI validation when DPI info is missing"""
        # Create mock image without DPI info
        mock_img = MagicMock()
        mock_img.info = {}
        mock_img.width = 2550  # ~300 DPI for 8.5" width
        mock_img.height = 3300

        meets_standard, dpi, message = self.detector.validate_dpi(mock_img)

        # Should estimate DPI
        self.assertIn("estimated", message)

    def test_should_use_ocr_below_threshold(self):
        """Test should_use_ocr with text below threshold"""
        text = "Only a few words here"  # ~5 words
        result = self.detector.should_use_ocr(text, word_threshold=50)

        self.assertTrue(result)

    def test_should_use_ocr_above_threshold(self):
        """Test should_use_ocr with text above threshold"""
        text = " ".join(["word"] * 100)  # 100 words
        result = self.detector.should_use_ocr(text, word_threshold=50)

        self.assertFalse(result)

    @patch('yolo_ocr.Image')
    def test_detect_text_regions_no_model(self, mock_image):
        """Test text region detection when YOLO model is unavailable"""
        # Set model to None to simulate fallback
        self.detector.model = None

        mock_img = MagicMock()
        mock_img.width = 1000
        mock_img.height = 1000

        regions = self.detector.detect_text_regions(mock_img)

        # Should return full image as single region
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].bbox, (0, 0, 1000, 1000))
        self.assertEqual(regions[0].confidence, 1.0)

    @patch('yolo_ocr.pytesseract')
    @patch('yolo_ocr.Image')
    def test_extract_text_from_region(self, mock_image, mock_tesseract):
        """Test text extraction from a region"""
        # Mock image and Tesseract
        mock_img = MagicMock()
        mock_tesseract.image_to_string.return_value = "  Extracted Text  "

        region = TextRegion(bbox=(100, 100, 200, 200), confidence=0.9)

        text = self.detector.extract_text_from_region(mock_img, region)

        self.assertEqual(text, "Extracted Text")
        mock_tesseract.image_to_string.assert_called_once()


@unittest.skipUnless(HAS_OCR_BASE, "OCR base dependencies not available")
class TestExtractTextWithYOLOOCR(unittest.TestCase):
    """Test extract_text_with_yolo_ocr convenience function"""

    @patch('yolo_ocr.YOLOTextDetector')
    def test_extract_text_sufficient_fallback(self, mock_detector_class):
        """Test extraction when fallback text is sufficient"""
        fallback_text = " ".join(["word"] * 100)  # 100 words

        text, ocr_used = extract_text_with_yolo_ocr(
            "test.pdf",
            fallback_text=fallback_text,
            word_threshold=50
        )

        self.assertEqual(text, fallback_text)
        self.assertFalse(ocr_used)
        # YOLOTextDetector should not be instantiated
        mock_detector_class.assert_not_called()

    @patch('yolo_ocr.YOLOTextDetector')
    def test_extract_text_insufficient_fallback(self, mock_detector_class):
        """Test extraction when fallback text is insufficient"""
        fallback_text = "Only few words"  # ~3 words

        # Mock detector and its methods
        mock_detector = MagicMock()
        mock_result = OCRResult(
            page_num=1,
            text="OCR extracted text with many words " * 20,  # >50 words
            regions=[],
            processing_time=1.0
        )
        mock_detector.process_pdf.return_value = [mock_result]
        mock_detector_class.return_value = mock_detector

        text, ocr_used = extract_text_with_yolo_ocr(
            "test.pdf",
            fallback_text=fallback_text,
            word_threshold=50
        )

        self.assertIn("[Page 1]", text)
        self.assertTrue(ocr_used)
        mock_detector.process_pdf.assert_called_once()

    @patch('yolo_ocr.YOLOTextDetector')
    def test_extract_text_ocr_failure(self, mock_detector_class):
        """Test extraction when OCR fails"""
        fallback_text = "Few words"

        # Mock detector to raise exception
        mock_detector_class.side_effect = Exception("OCR failed")

        text, ocr_used = extract_text_with_yolo_ocr(
            "test.pdf",
            fallback_text=fallback_text,
            word_threshold=50
        )

        # Should fall back to original text
        self.assertEqual(text, fallback_text)
        self.assertFalse(ocr_used)


class TestYOLOOCRIntegration(unittest.TestCase):
    """Integration tests for YOLO OCR (requires dependencies)"""

    @unittest.skipUnless(HAS_YOLO and HAS_OCR_BASE, "YOLO and OCR dependencies required")
    def test_full_pipeline_availability(self):
        """Test that full YOLO + OCR pipeline can be initialized"""
        # This test verifies dependencies are installed correctly
        try:
            with patch('yolo_ocr.YOLO'):  # Mock to avoid model download
                detector = YOLOTextDetector(use_gpu=False)
                self.assertIsNotNone(detector)
        except Exception as e:
            self.fail(f"Failed to initialize YOLO OCR pipeline: {e}")


def run_tests():
    """Run all tests and print results"""
    print("=" * 70)
    print("YOLO OCR Module - Unit Tests")
    print("=" * 70)
    print()

    # Check dependencies
    print("Dependency Status:")
    print(f"  OCR Base (PIL, pytesseract, pdf2image): {HAS_OCR_BASE}")
    print(f"  YOLO (ultralytics, torch): {HAS_YOLO}")
    print()

    if not YOLO_OCR_AVAILABLE:
        print("⚠️  YOLO OCR module not available - skipping tests")
        print("Install dependencies: pip install torch torchvision ultralytics")
        return 1

    # Run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestTextRegion))
    suite.addTests(loader.loadTestsFromTestCase(TestOCRResult))
    suite.addTests(loader.loadTestsFromTestCase(TestYOLOTextDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestExtractTextWithYOLOOCR))
    suite.addTests(loader.loadTestsFromTestCase(TestYOLOOCRIntegration))

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
    sys.exit(run_tests())
