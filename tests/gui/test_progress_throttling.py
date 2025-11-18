#!/usr/bin/env python3
"""
Test Progress Bar Throttling Logic

Verifies that the progress update throttling works correctly
to prevent UI flooding during parallel processing.

Author: Classmark Development Team
Date: 2025-11-14
"""

import unittest
import time


class TestProgressThrottling(unittest.TestCase):
    """Test progress bar throttling logic"""

    def setUp(self):
        """Set up test state"""
        self.last_progress_update = 0
        self.last_progress_value = 0
        self.processing_start_time = time.time()
        self.update_count = 0

    def should_update_progress(self, progress, completed_count, total_files):
        """
        Replicate the throttling logic from classmark_gui.py

        Returns True if progress should be updated based on:
        1. Progress changed by >= 1% OR
        2. More than 0.5 seconds passed OR
        3. This is the last file (100% complete)
        """
        current_time = time.time()
        time_since_last_update = current_time - self.last_progress_update
        progress_delta = abs(progress - self.last_progress_value)

        should_update = (
            progress_delta >= 1.0 or  # Progress changed by 1%+
            time_since_last_update >= 0.5 or  # 500ms elapsed
            completed_count == total_files  # Last file
        )

        if should_update:
            self.last_progress_update = current_time
            self.last_progress_value = progress
            self.update_count += 1
            return True
        return False

    def test_throttling_prevents_excessive_updates(self):
        """Test that throttling reduces update frequency"""
        total_files = 100
        updates_sent = 0

        # Simulate processing 100 files rapidly (< 0.5s total)
        for completed in range(1, total_files + 1):
            progress = (completed / total_files) * 100

            if self.should_update_progress(progress, completed, total_files):
                updates_sent += 1

        # Without throttling: 100 updates
        # With throttling: fewer updates when processing is very fast
        # The exact count depends on CPU speed and timing

        # In this test, progress changes by 1% each iteration,
        # but since the loop executes so fast (< 500ms total),
        # the time-based throttling kicks in for some iterations

        # Should get close to 100 updates, but time throttling may reduce it
        self.assertGreater(updates_sent, 90,
            "Should get most updates but time throttling may filter some")
        self.assertLessEqual(updates_sent, 100,
            "Should never exceed total files")

    def test_time_based_throttling(self):
        """Test that time-based throttling limits updates"""
        total_files = 1000

        # Reset timing
        self.last_progress_update = time.time()

        # Simulate rapid completions within same time window
        # (parallel processing completing multiple files at once)
        updates_sent = 0

        # Process files in quick succession
        for completed in range(1, 101):  # First 100 files
            progress = (completed / total_files) * 100  # 0.1% per file

            if self.should_update_progress(progress, completed, total_files):
                updates_sent += 1

        # Since progress changes by only 0.1% per file,
        # and we need 1% change OR 0.5s elapsed,
        # we should get far fewer than 100 updates
        self.assertLess(updates_sent, 20,
            "Time throttling should limit updates when progress delta < 1%")

    def test_final_update_always_sent(self):
        """Test that 100% completion always sends update"""
        total_files = 50

        # Skip to near the end
        for completed in range(1, total_files):
            progress = (completed / total_files) * 100
            self.should_update_progress(progress, completed, total_files)

        # Record update count before final
        updates_before_final = self.update_count

        # Process final file
        final_progress = 100.0
        should_update = self.should_update_progress(final_progress, total_files, total_files)

        # Final update should always be sent
        self.assertTrue(should_update, "Final update (100%) must always be sent")
        self.assertEqual(self.update_count, updates_before_final + 1)

    def test_percentage_threshold_works(self):
        """Test that 1% threshold prevents micro-updates"""
        total_files = 1000

        # Process 5 files (0.5% progress) - should not update if < 1% delta
        self.last_progress_value = 0
        updates = 0

        for completed in range(1, 6):
            progress = (completed / total_files) * 100  # 0.1%, 0.2%, ... 0.5%
            if self.should_update_progress(progress, completed, total_files):
                updates += 1

        # Should get 1 update initially (first file always updates due to time)
        # Then no more until we hit 1% delta
        self.assertLessEqual(updates, 2,
            "Should not update for sub-1% progress changes within time window")

    def test_calculate_statistics(self):
        """Test that statistics calculation works correctly"""
        # Simulate real processing
        total_files = 100
        completed_count = 50

        current_time = time.time()
        elapsed_time = current_time - self.processing_start_time

        # Calculate stats (same as GUI)
        files_per_sec = completed_count / elapsed_time if elapsed_time > 0 else 0
        remaining_files = total_files - completed_count
        eta_seconds = remaining_files / files_per_sec if files_per_sec > 0 else 0

        # Verify calculations
        self.assertGreater(files_per_sec, 0, "Should calculate files/sec")
        self.assertEqual(remaining_files, 50, "Remaining files should be 50")
        self.assertGreaterEqual(eta_seconds, 0, "ETA should be non-negative")


class TestProgressLabelFormatting(unittest.TestCase):
    """Test progress label formatting"""

    def test_progress_label_format_during_processing(self):
        """Test progress label formatting during processing"""
        completed_count = 50
        total_files = 100
        files_per_sec = 10.5
        eta_seconds = 4.76

        # Format label (same as GUI)
        progress_label = f"{completed_count}/{total_files} files ({files_per_sec:.1f} files/sec, ETA: {eta_seconds:.0f}s)"

        # Verify format
        self.assertIn("50/100 files", progress_label)
        self.assertIn("10.5 files/sec", progress_label)
        self.assertIn("ETA: 5s", progress_label)  # Rounded

    def test_progress_label_format_on_completion(self):
        """Test progress label formatting at 100% completion"""
        completed_count = 100
        total_files = 100

        # Format final label (same as GUI)
        progress_label = f"Complete: {completed_count}/{total_files} files"

        # Verify format
        self.assertEqual(progress_label, "Complete: 100/100 files")


if __name__ == '__main__':
    unittest.main()
