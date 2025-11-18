"""
Test for CSV logging bug fix

Verifies that save_log() correctly handles heterogeneous result dictionaries
where different results have different fields (e.g., some with 'output_path',
some without).

Bug: csv.DictWriter was initialized with fieldnames=results[0].keys(), which
failed when later results had additional fields like 'output_path'.

Fix: Collect all unique fields from all results before creating DictWriter.
"""

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path


def save_log_csv(results, output_path):
    """
    Standalone implementation of the fixed save_log() CSV logic for testing.

    This is extracted from classmark_gui.py to test without GUI dependencies.
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        if results:
            # Define preferred field order
            preferred_order = [
                'file_path', 'filename', 'status',
                'has_classification', 'classification_level', 'confidence',
                'num_matches', 'moved', 'output_path', 'move_error',
                'processing_time', 'error'
            ]

            # Collect all actual fields present in results
            actual_fields = set()
            for result in results:
                actual_fields.update(result.keys())

            # Use preferred order for known fields, append any extras
            fieldnames = [f for f in preferred_order if f in actual_fields]
            extra_fields = sorted(actual_fields - set(preferred_order))
            fieldnames.extend(extra_fields)

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        else:
            # Write empty CSV with standard headers
            fieldnames = ['file_path', 'filename', 'status']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()


class TestSaveLogFix(unittest.TestCase):
    """Test cases for CSV logging bug fix"""

    def setUp(self):
        """Create temporary directory for test outputs"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_scenario_1_mixed_success_and_error(self):
        """Test Case 1: Mixed success and error results"""
        results = [
            {
                'file_path': '/path/to/error.pdf',
                'filename': 'error.pdf',
                'status': 'error',
                'error': 'Failed to read file'
            },
            {
                'file_path': '/path/to/success.pdf',
                'filename': 'success.pdf',
                'status': 'success',
                'has_classification': True,
                'classification_level': 'SECRET',
                'confidence': 0.95,
                'num_matches': 2,
                'processing_time': 0.5,
                'moved': False
            }
        ]

        output_path = os.path.join(self.temp_dir, 'test1.csv')
        save_log_csv(results, output_path)

        # Verify CSV was created and is valid
        self.assertTrue(os.path.exists(output_path))

        # Read back and verify
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]['status'], 'error')
            self.assertEqual(rows[1]['status'], 'success')
            self.assertEqual(rows[1]['confidence'], '0.95')

    def test_scenario_2_some_moved_some_not(self):
        """Test Case 2: Some files moved, some not (THE BUG SCENARIO)"""
        results = [
            {
                'file_path': '/path/to/clean.pdf',
                'filename': 'clean.pdf',
                'status': 'success',
                'has_classification': False,
                'classification_level': None,
                'confidence': 0.0,
                'num_matches': 0,
                'processing_time': 0.3,
                'moved': False  # No output_path field
            },
            {
                'file_path': '/path/to/classified.pdf',
                'filename': 'classified.pdf',
                'status': 'success',
                'has_classification': True,
                'classification_level': 'TOP SECRET',
                'confidence': 0.98,
                'num_matches': 5,
                'processing_time': 1.2,
                'moved': True,
                'output_path': '/output/classified.pdf'  # Has output_path field
            },
            {
                'file_path': '/path/to/failed_move.pdf',
                'filename': 'failed_move.pdf',
                'status': 'success',
                'has_classification': True,
                'classification_level': 'CONFIDENTIAL',
                'confidence': 0.87,
                'num_matches': 3,
                'processing_time': 0.8,
                'moved': False,
                'move_error': 'Permission denied'  # Has move_error field
            }
        ]

        output_path = os.path.join(self.temp_dir, 'test2.csv')

        # This should NOT raise ValueError about 'output_path' field
        save_log_csv(results, output_path)

        # Verify CSV was created
        self.assertTrue(os.path.exists(output_path))

        # Read back and verify all fields are present
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)

            # Verify fieldnames include all conditional fields
            self.assertIn('output_path', fieldnames)
            self.assertIn('move_error', fieldnames)

            # Verify data
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]['moved'], 'False')
            self.assertEqual(rows[0]['output_path'], '')  # Empty for non-moved file

            self.assertEqual(rows[1]['moved'], 'True')
            self.assertEqual(rows[1]['output_path'], '/output/classified.pdf')

            self.assertEqual(rows[2]['moved'], 'False')
            self.assertEqual(rows[2]['move_error'], 'Permission denied')

    def test_scenario_3_all_moved(self):
        """Test Case 3: All files successfully moved"""
        results = [
            {
                'file_path': '/a.pdf',
                'filename': 'a.pdf',
                'status': 'success',
                'has_classification': True,
                'classification_level': 'SECRET',
                'confidence': 0.92,
                'num_matches': 3,
                'processing_time': 0.7,
                'moved': True,
                'output_path': '/out/a.pdf'
            },
            {
                'file_path': '/b.pdf',
                'filename': 'b.pdf',
                'status': 'success',
                'has_classification': True,
                'classification_level': 'TOP SECRET',
                'confidence': 0.99,
                'num_matches': 7,
                'processing_time': 1.5,
                'moved': True,
                'output_path': '/out/b.pdf'
            }
        ]

        output_path = os.path.join(self.temp_dir, 'test3.csv')
        save_log_csv(results, output_path)

        self.assertTrue(os.path.exists(output_path))

        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]['output_path'], '/out/a.pdf')
            self.assertEqual(rows[1]['output_path'], '/out/b.pdf')

    def test_scenario_4_empty_results(self):
        """Test Case 4: Empty results list"""
        results = []

        output_path = os.path.join(self.temp_dir, 'test4.csv')
        save_log_csv(results, output_path)

        # Should create empty CSV with headers
        self.assertTrue(os.path.exists(output_path))

        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            self.assertEqual(len(rows), 0)
            # Should have standard headers
            self.assertIn('file_path', reader.fieldnames)
            self.assertIn('filename', reader.fieldnames)
            self.assertIn('status', reader.fieldnames)

    def test_scenario_5_field_order(self):
        """Test Case 5: Verify field ordering is consistent"""
        results = [
            {
                'file_path': '/test.pdf',
                'filename': 'test.pdf',
                'status': 'success',
                'has_classification': True,
                'classification_level': 'SECRET',
                'confidence': 0.95,
                'num_matches': 2,
                'processing_time': 0.5,
                'moved': True,
                'output_path': '/out/test.pdf'
            }
        ]

        output_path = os.path.join(self.temp_dir, 'test5.csv')
        save_log_csv(results, output_path)

        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames

            # Verify preferred ordering
            self.assertEqual(fieldnames[0], 'file_path')
            self.assertEqual(fieldnames[1], 'filename')
            self.assertEqual(fieldnames[2], 'status')

            # Output_path should come before processing_time per preferred order
            output_path_idx = fieldnames.index('output_path')
            processing_time_idx = fieldnames.index('processing_time')
            self.assertLess(output_path_idx, processing_time_idx)

    def test_scenario_6_special_characters(self):
        """Test Case 6: Special characters in file paths"""
        results = [
            {
                'file_path': '/path/with spaces/file.pdf',
                'filename': 'file.pdf',
                'status': 'success',
                'has_classification': False,
                'confidence': 0.0,
                'num_matches': 0,
                'processing_time': 0.2,
                'moved': False
            },
            {
                'file_path': '/path/with,comma/file.pdf',
                'filename': 'file.pdf',
                'status': 'success',
                'has_classification': False,
                'confidence': 0.0,
                'num_matches': 0,
                'processing_time': 0.2,
                'moved': False
            }
        ]

        output_path = os.path.join(self.temp_dir, 'test6.csv')
        save_log_csv(results, output_path)

        # Should handle special characters correctly
        self.assertTrue(os.path.exists(output_path))

        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            self.assertEqual(rows[0]['file_path'], '/path/with spaces/file.pdf')
            self.assertEqual(rows[1]['file_path'], '/path/with,comma/file.pdf')


if __name__ == '__main__':
    unittest.main()
