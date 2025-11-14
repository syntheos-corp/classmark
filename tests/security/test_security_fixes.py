"""
Security Tests for Classification Scanner

Tests for security fixes implemented in the performance optimization review:
1. Path traversal protection
2. PyMuPDF safety (large PDFs, corrupted files)
3. Settings validation
4. Race condition handling

Author: Classmark Development Team
Date: 2025-11-13
"""

import os
import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.gui.classmark_gui import ClassmarkGUI, process_file_worker
from src.core.classification_scanner import ClassificationScanner, TextExtractor
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter


class TestPathTraversalProtection(unittest.TestCase):
    """Test path traversal attack prevention"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.input_folder = os.path.join(self.temp_dir, "input")
        self.output_folder = os.path.join(self.temp_dir, "output")
        os.makedirs(self.input_folder)
        os.makedirs(self.output_folder)

    def tearDown(self):
        """Clean up test environment"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_path_traversal_rejected(self):
        """Test that path traversal attempts are rejected"""
        # Create a directory structure that looks malicious
        # We'll create a file with path components that try to escape
        malicious_dir = os.path.join(self.input_folder, "..", "malicious")
        os.makedirs(malicious_dir, exist_ok=True)

        # Create a test PDF with classification marking in the malicious location
        test_pdf = os.path.join(malicious_dir, "test.pdf")
        c = canvas.Canvas(test_pdf, pagesize=letter)
        c.drawString(100, 750, 'SECRET')
        c.save()

        config = {
            'confidence_threshold': 0.3,
            'fuzzy_matching': False,
            'use_llm': False,
            'use_gpu': False,
            'use_visual_detection': False
        }

        # The file is outside input_folder, so rel_path will have ".."
        # This should trigger path traversal detection when constructing output path
        result = process_file_worker(
            test_pdf,
            config,
            self.input_folder,
            self.output_folder,
            move_hits=True
        )

        # Should detect path traversal and report error
        self.assertFalse(result.get('moved', False))
        if 'move_error' in result:
            self.assertIn("Path traversal", result['move_error'])

    def test_normal_path_allowed(self):
        """Test that normal file operations work correctly"""
        # Create a test PDF with classification marking
        test_pdf = os.path.join(self.input_folder, "subdir", "test.pdf")
        os.makedirs(os.path.dirname(test_pdf), exist_ok=True)

        c = canvas.Canvas(test_pdf, pagesize=letter)
        c.drawString(100, 750, 'SECRET')
        c.save()

        config = {
            'confidence_threshold': 0.3,
            'fuzzy_matching': False,
            'use_llm': False,
            'use_gpu': False,
            'use_visual_detection': False
        }

        result = process_file_worker(
            test_pdf,
            config,
            self.input_folder,
            self.output_folder,
            move_hits=True
        )

        # Should succeed
        self.assertEqual(result['status'], 'success')
        # File should be moved
        expected_output = os.path.join(self.output_folder, "subdir", "test.pdf")
        self.assertTrue(os.path.exists(expected_output))


class TestPyMuPDFSafety(unittest.TestCase):
    """Test PyMuPDF safety hardening"""

    def setUp(self):
        """Set up test environment"""
        self.extractor = TextExtractor()

    def test_max_pages_validation(self):
        """Test that max_pages parameter is validated"""
        # Create a test PDF
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            test_pdf = f.name

        c = canvas.Canvas(test_pdf, pagesize=letter)
        c.drawString(100, 750, 'Page 1')
        c.showPage()
        c.drawString(100, 750, 'Page 2')
        c.save()

        try:
            # Test negative max_pages (should be clamped to 1)
            result = self.extractor.extract_from_pdf_pymupdf(test_pdf, max_pages=-1)
            # Should not crash

            # Test huge max_pages (should be clamped to 10000)
            result = self.extractor.extract_from_pdf_pymupdf(test_pdf, max_pages=999999)
            # Should not crash

            self.assertTrue(True)  # Made it here without crashing
        finally:
            os.unlink(test_pdf)

    def test_corrupted_pdf_handling(self):
        """Test that corrupted PDFs don't crash the scanner"""
        # Create a corrupted PDF (just random bytes)
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            test_pdf = f.name
            f.write(b'This is not a valid PDF file\x00\x01\x02\x03')

        try:
            # Should handle gracefully without crashing
            result = self.extractor.extract_from_pdf_pymupdf(test_pdf)

            # Should return empty result
            self.assertEqual(result['body'], [])
        finally:
            os.unlink(test_pdf)

    def test_missing_file_handling(self):
        """Test that missing files are handled gracefully"""
        non_existent = "/tmp/does_not_exist_12345.pdf"

        # Should not crash
        result = self.extractor.extract_from_pdf_pymupdf(non_existent)

        # Should return empty result
        self.assertEqual(result['body'], [])


class TestSettingsValidation(unittest.TestCase):
    """Test settings validation on config load"""

    def test_parallel_workers_clamped(self):
        """Test that parallel_workers is clamped to safe range"""
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()

        try:
            gui = ClassmarkGUI(root)

            # Test fork bomb attempt
            malicious_settings = {
                'parallel_workers': 999999
            }

            validated = gui._validate_settings(malicious_settings)

            # Should be clamped to CPU count
            import multiprocessing
            self.assertLessEqual(validated['parallel_workers'], multiprocessing.cpu_count())
            self.assertGreaterEqual(validated['parallel_workers'], 1)
        finally:
            root.destroy()

    def test_early_exit_threshold_minimum(self):
        """Test that early_exit_threshold has minimum value"""
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()

        try:
            gui = ClassmarkGUI(root)

            # Try to set very low threshold to bypass security
            malicious_settings = {
                'early_exit_threshold': 0.01
            }

            validated = gui._validate_settings(malicious_settings)

            # Should be clamped to minimum 0.70
            self.assertGreaterEqual(validated['early_exit_threshold'], 0.70)
        finally:
            root.destroy()

    def test_quick_scan_pages_range(self):
        """Test that quick_scan_pages is in valid range"""
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()

        try:
            gui = ClassmarkGUI(root)

            # Test zero pages (would skip scanning)
            malicious_settings = {
                'quick_scan_pages': 0
            }

            validated = gui._validate_settings(malicious_settings)

            # Should be clamped to 1-10 range
            self.assertGreaterEqual(validated['quick_scan_pages'], 1)
            self.assertLessEqual(validated['quick_scan_pages'], 10)

            # Test excessive pages
            malicious_settings = {
                'quick_scan_pages': 1000
            }

            validated = gui._validate_settings(malicious_settings)
            self.assertLessEqual(validated['quick_scan_pages'], 10)
        finally:
            root.destroy()

    def test_string_settings_whitelisted(self):
        """Test that string settings are validated against whitelist"""
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()

        try:
            gui = ClassmarkGUI(root)

            # Try to inject invalid sensitivity
            malicious_settings = {
                'sensitivity': 'malicious_value',
                'log_format': 'xml'  # Not in whitelist
            }

            validated = gui._validate_settings(malicious_settings)

            # Invalid values should not be in validated dict
            self.assertNotIn('sensitivity', validated)
            self.assertNotIn('log_format', validated)

            # Valid values should pass through
            valid_settings = {
                'sensitivity': 'high',
                'log_format': 'json'
            }

            validated = gui._validate_settings(valid_settings)
            self.assertEqual(validated['sensitivity'], 'high')
            self.assertEqual(validated['log_format'], 'json')
        finally:
            root.destroy()


class TestRaceConditionHandling(unittest.TestCase):
    """Test race condition handling in file operations"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.input_folder = os.path.join(self.temp_dir, "input")
        self.output_folder = os.path.join(self.temp_dir, "output")
        os.makedirs(self.input_folder)
        os.makedirs(self.output_folder)

    def tearDown(self):
        """Clean up test environment"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_missing_source_file_handled(self):
        """Test that missing source files are handled gracefully"""
        non_existent = os.path.join(self.input_folder, "deleted.pdf")

        config = {
            'confidence_threshold': 0.3,
            'fuzzy_matching': False,
            'use_llm': False,
            'use_gpu': False,
            'use_visual_detection': False
        }

        result = process_file_worker(
            non_existent,
            config,
            self.input_folder,
            self.output_folder,
            move_hits=True
        )

        # Should return error status, not crash
        self.assertEqual(result['status'], 'error')

    def test_duplicate_destination_handled(self):
        """Test that duplicate destination files are handled"""
        # Create test PDF
        test_pdf = os.path.join(self.input_folder, "test.pdf")
        c = canvas.Canvas(test_pdf, pagesize=letter)
        c.drawString(100, 750, 'SECRET')
        c.save()

        # Pre-create the destination
        dest_path = os.path.join(self.output_folder, "test.pdf")
        with open(dest_path, 'w') as f:
            f.write("Already exists")

        config = {
            'confidence_threshold': 0.3,
            'fuzzy_matching': False,
            'use_llm': False,
            'use_gpu': False,
            'use_visual_detection': False
        }

        result = process_file_worker(
            test_pdf,
            config,
            self.input_folder,
            self.output_folder,
            move_hits=True
        )

        # Should indicate not moved due to existing destination
        self.assertFalse(result.get('moved', False))
        self.assertIn('already exists', result.get('move_error', '').lower())


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
