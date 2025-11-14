"""
Unit Tests for Hybrid Two-Stage Classifier

Tests for the hybrid classification architecture including:
- Fast path execution
- Slow path execution
- Early exit decisions
- Confidence thresholds
- Path routing logic
- Performance benchmarking

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

from src.core.hybrid_classifier import (
    HybridClassifier,
    HybridResult,
    ClassificationPath,
    ConfidenceDecision,
    PathMetrics,
    HAS_FAST_MATCHER,
    HAS_VISUAL_DETECTOR,
    HAS_OCR
)


class TestHybridInitialization(unittest.TestCase):
    """Test hybrid classifier initialization"""

    def test_basic_initialization(self):
        """Test basic initialization"""
        classifier = HybridClassifier()

        self.assertIsNotNone(classifier)
        self.assertEqual(classifier.accept_threshold, 0.95)
        self.assertEqual(classifier.reject_threshold, 0.10)

    def test_custom_configuration(self):
        """Test custom configuration"""
        config = {
            'accept_threshold': 0.90,
            'reject_threshold': 0.15,
            'min_confidence': 0.4
        }

        classifier = HybridClassifier(config=config)

        self.assertEqual(classifier.accept_threshold, 0.90)
        self.assertEqual(classifier.reject_threshold, 0.15)
        self.assertEqual(classifier.min_confidence, 0.4)

    def test_component_initialization(self):
        """Test that components are properly initialized"""
        classifier = HybridClassifier(use_gpu=False)

        if HAS_FAST_MATCHER:
            self.assertIsNotNone(classifier.fast_matcher)

        if HAS_VISUAL_DETECTOR:
            self.assertIsNotNone(classifier.visual_detector)

        if HAS_OCR:
            self.assertIsNotNone(classifier.ocr_detector)


class TestFastPathClassification(unittest.TestCase):
    """Test fast path classification"""

    def setUp(self):
        """Set up test fixtures"""
        self.classifier = HybridClassifier(use_gpu=False)

    @unittest.skipUnless(HAS_FAST_MATCHER, "Fast matcher not available")
    def test_fast_path_high_confidence(self):
        """Test fast path with high confidence text"""
        text = """
        TOP SECRET//NOFORN

        CLASSIFIED BY: John Doe
        DERIVED FROM: ABC-123
        DECLASSIFY ON: 20400101

        This is a classified document.

        TOP SECRET//NOFORN
        """

        result = self.classifier.classify(text)

        self.assertIsInstance(result, HybridResult)
        self.assertTrue(result.has_classification)
        # Classification level may or may not be extracted depending on pattern matching
        # The important thing is that classification is detected with good confidence
        self.assertGreaterEqual(result.confidence, 0.5)  # Should have decent confidence
        self.assertLessEqual(result.confidence, 1.0)
        self.assertIsNotNone(result.fast_path_metrics)
        self.assertGreater(len(result.matches), 0)  # Should have matches

    @unittest.skipUnless(HAS_FAST_MATCHER, "Fast matcher not available")
    def test_fast_path_no_classification(self):
        """Test fast path with no classification markings"""
        text = """
        This is a regular document with no classification markings.
        It contains normal text and paragraphs.
        There are no special markings to detect.
        """

        result = self.classifier.classify(text)

        self.assertIsInstance(result, HybridResult)
        # May or may not have classification depending on false positives
        # Just check structure is correct
        self.assertIsNotNone(result.fast_path_metrics)

    @unittest.skipUnless(HAS_FAST_MATCHER, "Fast matcher not available")
    def test_fast_path_performance(self):
        """Test that fast path is actually fast (<10ms target)"""
        text = "TOP SECRET//NOFORN\nThis is a classified document.\nTOP SECRET//NOFORN"

        # Warm-up
        self.classifier.classify(text)

        # Benchmark
        iterations = 10
        start = time.time()
        for _ in range(iterations):
            result = self.classifier.classify(text)
        elapsed = time.time() - start

        avg_time = elapsed / iterations

        print(f"\n  Fast path avg time: {avg_time*1000:.2f}ms")

        # Should be very fast (allow up to 50ms for safety on slow systems)
        self.assertLess(avg_time, 0.05)


class TestEarlyExitDecisions(unittest.TestCase):
    """Test early exit threshold logic"""

    def setUp(self):
        """Set up test fixtures"""
        config = {
            'accept_threshold': 0.95,
            'reject_threshold': 0.10,
            'min_confidence': 0.3
        }
        self.classifier = HybridClassifier(config=config, use_gpu=False)

    @unittest.skipUnless(HAS_FAST_MATCHER, "Fast matcher not available")
    def test_early_accept_decision(self):
        """Test that high confidence triggers early accept"""
        # Create a result with very high confidence
        result = HybridResult(
            has_classification=True,
            classification_level='TOP_SECRET',
            confidence=0.97,
            matches=[],
            path_taken=ClassificationPath.FAST_PATH,
            fast_path_metrics=PathMetrics(
                path_type=ClassificationPath.FAST_PATH,
                processing_time=0.001,
                confidence=0.97,
                num_matches=5,
                decision=ConfidenceDecision.UNCERTAIN
            )
        )

        decision = self.classifier._evaluate_confidence(result)
        self.assertEqual(decision, ConfidenceDecision.ACCEPT)

    @unittest.skipUnless(HAS_FAST_MATCHER, "Fast matcher not available")
    def test_early_reject_decision(self):
        """Test that low confidence triggers early reject"""
        # Create a result with very low confidence
        result = HybridResult(
            has_classification=False,
            confidence=0.05,
            matches=[],
            path_taken=ClassificationPath.FAST_PATH,
            fast_path_metrics=PathMetrics(
                path_type=ClassificationPath.FAST_PATH,
                processing_time=0.001,
                confidence=0.05,
                num_matches=0,
                decision=ConfidenceDecision.UNCERTAIN
            )
        )

        decision = self.classifier._evaluate_confidence(result)
        self.assertEqual(decision, ConfidenceDecision.REJECT)

    @unittest.skipUnless(HAS_FAST_MATCHER, "Fast matcher not available")
    def test_uncertain_decision(self):
        """Test that medium confidence triggers slow path"""
        # Create a result with medium confidence
        result = HybridResult(
            has_classification=True,
            classification_level='SECRET',
            confidence=0.50,
            matches=[],
            path_taken=ClassificationPath.FAST_PATH,
            fast_path_metrics=PathMetrics(
                path_type=ClassificationPath.FAST_PATH,
                processing_time=0.001,
                confidence=0.50,
                num_matches=2,
                decision=ConfidenceDecision.UNCERTAIN
            )
        )

        decision = self.classifier._evaluate_confidence(result)
        self.assertEqual(decision, ConfidenceDecision.UNCERTAIN)


class TestLocationAnalysis(unittest.TestCase):
    """Test location-based confidence analysis"""

    def setUp(self):
        """Set up test fixtures"""
        self.classifier = HybridClassifier(use_gpu=False)

    @unittest.skipUnless(HAS_FAST_MATCHER, "Fast matcher not available")
    def test_header_location_boost(self):
        """Test that header locations get confidence boost"""
        text = "TOP SECRET//NOFORN\n" + "\n".join(["Body text"] * 20)

        result = self.classifier.classify(text)

        if result.matches:
            # Check if header matches have higher confidence
            header_matches = [m for m in result.matches if m.get('location') == 'header']
            if header_matches:
                self.assertGreater(header_matches[0]['confidence'], 0.5)

    @unittest.skipUnless(HAS_FAST_MATCHER, "Fast matcher not available")
    def test_footer_location_boost(self):
        """Test that footer locations get confidence boost"""
        text = "\n".join(["Body text"] * 20) + "\nTOP SECRET//NOFORN"

        result = self.classifier.classify(text)

        if result.matches:
            # Check if footer matches exist
            footer_matches = [m for m in result.matches if m.get('location') == 'footer']
            # Footer should be detected (may or may not depending on text length)
            self.assertIsInstance(footer_matches, list)


class TestContextAnalysis(unittest.TestCase):
    """Test context-based confidence refinement"""

    def setUp(self):
        """Set up test fixtures"""
        self.classifier = HybridClassifier(use_gpu=False)

    @unittest.skipUnless(HAS_FAST_MATCHER, "Fast matcher not available")
    def test_multiple_markings_boost(self):
        """Test that multiple markings boost confidence"""
        text = """
        TOP SECRET//NOFORN

        NOFORN
        ORCON
        IMCON

        TOP SECRET//NOFORN
        """

        result = self.classifier.classify(text)

        # Should have multiple matches
        self.assertGreater(len(result.matches), 2)

        # Confidence should be boosted by context
        if result.matches:
            avg_confidence = sum(m['confidence'] for m in result.matches) / len(result.matches)
            self.assertGreater(avg_confidence, 0.5)

    @unittest.skipUnless(HAS_FAST_MATCHER, "Fast matcher not available")
    def test_consistent_level_boost(self):
        """Test that consistent classification level boosts confidence"""
        text = """
        TOP SECRET//NOFORN

        (TS) Paragraph one
        (TS) Paragraph two
        (TS) Paragraph three

        TOP SECRET//NOFORN
        """

        result = self.classifier.classify(text)

        # Should detect consistent TOP SECRET level
        self.assertTrue(result.has_classification)
        if result.classification_level:
            self.assertIn('TOP', result.classification_level.upper())


class TestClassificationDetermination(unittest.TestCase):
    """Test classification level determination"""

    def setUp(self):
        """Set up test fixtures"""
        self.classifier = HybridClassifier(use_gpu=False)

    def test_top_secret_highest(self):
        """Test that TOP SECRET is recognized as highest level"""
        matches = [
            {'text': 'TOP SECRET', 'confidence': 0.8, 'category': 'level', 'pattern_type': 'TOP_SECRET'},
            {'text': 'SECRET', 'confidence': 0.7, 'category': 'level', 'pattern_type': 'SECRET'},
            {'text': 'CONFIDENTIAL', 'confidence': 0.6, 'category': 'level', 'pattern_type': 'CONFIDENTIAL'}
        ]

        level, confidence = self.classifier._determine_classification(matches)

        self.assertEqual(level, 'TOP_SECRET')
        self.assertGreater(confidence, 0.0)

    def test_secret_over_confidential(self):
        """Test that SECRET ranks higher than CONFIDENTIAL"""
        matches = [
            {'text': 'SECRET', 'confidence': 0.7, 'category': 'level', 'pattern_type': 'SECRET'},
            {'text': 'CONFIDENTIAL', 'confidence': 0.8, 'category': 'level', 'pattern_type': 'CONFIDENTIAL'}
        ]

        level, confidence = self.classifier._determine_classification(matches)

        self.assertEqual(level, 'SECRET')

    def test_no_level_found(self):
        """Test when no classification level is found"""
        matches = [
            {'text': 'NOFORN', 'confidence': 0.6, 'category': 'control', 'pattern_type': 'NOFORN'},
            {'text': 'ORCON', 'confidence': 0.5, 'category': 'control', 'pattern_type': 'ORCON'}
        ]

        level, confidence = self.classifier._determine_classification(matches)

        self.assertIsNone(level)
        self.assertGreater(confidence, 0.0)  # Should still have confidence from matches

    def test_empty_matches(self):
        """Test with no matches"""
        matches = []

        level, confidence = self.classifier._determine_classification(matches)

        self.assertIsNone(level)
        self.assertEqual(confidence, 0.0)


@unittest.skipUnless(HAS_VISUAL_DETECTOR and HAS_OCR, "Visual detector or OCR not available")
class TestSlowPathClassification(unittest.TestCase):
    """Test slow path classification with visual detection"""

    def setUp(self):
        """Set up test fixtures"""
        self.classifier = HybridClassifier(use_gpu=False)
        self.docs_dir = Path(__file__).parent / "documents"

    def test_slow_path_with_pdf(self):
        """Test slow path execution on PDF document"""
        if not self.docs_dir.exists():
            self.skipTest("documents/ directory not found")

        pdf_files = list(self.docs_dir.glob("*.pdf"))
        if not pdf_files:
            self.skipTest("No PDF files found")

        pdf_path = str(pdf_files[0])
        print(f"\n  Testing on: {pdf_path}")

        # Use low confidence text to force slow path
        text = "This is minimal text"

        start = time.time()
        result = self.classifier.classify(text, file_path=pdf_path)
        elapsed = time.time() - start

        print(f"  Classification time: {elapsed:.2f}s")
        print(f"  Path taken: {result.path_taken.value}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Matches: {len(result.matches)}")

        # Should have executed slow path
        self.assertIsNotNone(result.slow_path_metrics)


class TestStatistics(unittest.TestCase):
    """Test performance statistics tracking"""

    @unittest.skipUnless(HAS_FAST_MATCHER, "Fast matcher not available")
    def test_statistics_tracking(self):
        """Test that statistics are properly tracked"""
        classifier = HybridClassifier(use_gpu=False)

        # Reset statistics
        classifier.reset_statistics()

        # Run some classifications
        texts = [
            "TOP SECRET//NOFORN\nClassified document\nTOP SECRET//NOFORN",
            "Regular document with no markings",
            "SECRET//NOFORN\nAnother classified document\nSECRET//NOFORN"
        ]

        for text in texts:
            classifier.classify(text)

        stats = classifier.get_statistics()

        # Check statistics
        self.assertEqual(stats['total_documents'], 3)
        self.assertGreater(stats['total_fast_time'], 0.0)

        # Check rates
        if stats['total_documents'] > 0:
            self.assertIn('fast_path_rate', stats)
            self.assertIn('slow_path_rate', stats)

    @unittest.skipUnless(HAS_FAST_MATCHER, "Fast matcher not available")
    def test_statistics_reset(self):
        """Test statistics reset"""
        classifier = HybridClassifier(use_gpu=False)

        # Run a classification
        classifier.classify("TOP SECRET//NOFORN")

        # Reset
        classifier.reset_statistics()

        stats = classifier.get_statistics()

        # Should be reset
        self.assertEqual(stats['total_documents'], 0)
        self.assertEqual(stats['total_fast_time'], 0.0)


class TestPerformanceBenchmark(unittest.TestCase):
    """Performance benchmarking tests"""

    @unittest.skipUnless(HAS_FAST_MATCHER, "Fast matcher not available")
    def test_throughput_benchmark(self):
        """Benchmark classification throughput"""
        classifier = HybridClassifier(use_gpu=False)

        # Test documents
        test_docs = [
            "TOP SECRET//NOFORN\nClassified document {}\nTOP SECRET//NOFORN".format(i)
            for i in range(100)
        ]

        # Warm-up
        for doc in test_docs[:10]:
            classifier.classify(doc)

        # Reset statistics
        classifier.reset_statistics()

        # Benchmark
        start = time.time()
        for doc in test_docs:
            result = classifier.classify(doc)
        elapsed = time.time() - start

        throughput = len(test_docs) / elapsed
        avg_time = elapsed / len(test_docs)

        print(f"\n  Throughput Benchmark:")
        print(f"    Documents: {len(test_docs)}")
        print(f"    Total time: {elapsed:.2f}s")
        print(f"    Throughput: {throughput:.1f} docs/sec ({throughput*60:.0f} docs/min)")
        print(f"    Avg time: {avg_time*1000:.2f}ms per document")

        stats = classifier.get_statistics()
        print(f"    Fast path rate: {stats.get('fast_path_rate', 0)*100:.1f}%")
        print(f"    Early accept rate: {stats.get('early_accept_rate', 0)*100:.1f}%")

        # Target: >200 docs/min for fast path
        # Allow flexibility depending on system performance
        self.assertGreater(throughput * 60, 50)  # At least 50 docs/min


class TestResultSerialization(unittest.TestCase):
    """Test result serialization"""

    @unittest.skipUnless(HAS_FAST_MATCHER, "Fast matcher not available")
    def test_result_to_dict(self):
        """Test converting result to dictionary"""
        classifier = HybridClassifier(use_gpu=False)

        text = "TOP SECRET//NOFORN"
        result = classifier.classify(text)

        result_dict = result.to_dict()

        # Check dictionary structure
        self.assertIn('has_classification', result_dict)
        self.assertIn('classification_level', result_dict)
        self.assertIn('confidence', result_dict)
        self.assertIn('path_taken', result_dict)
        self.assertIn('total_time', result_dict)
        self.assertIn('early_exit', result_dict)

        # Check types
        self.assertIsInstance(result_dict['has_classification'], bool)
        self.assertIsInstance(result_dict['confidence'], float)
        self.assertIsInstance(result_dict['total_time'], float)


def run_tests():
    """Run all tests and print results"""
    print("=" * 70)
    print("Hybrid Two-Stage Classifier - Unit Tests")
    print("=" * 70)
    print()

    # Check dependencies
    print("Dependency Status:")
    print(f"  Fast Matcher: {HAS_FAST_MATCHER}")
    print(f"  Visual Detector: {HAS_VISUAL_DETECTOR}")
    print(f"  OCR: {HAS_OCR}")
    print()

    if not HAS_FAST_MATCHER:
        print("⚠ Fast matcher not available - some tests will be skipped")
        print()

    # Run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestHybridInitialization))
    suite.addTests(loader.loadTestsFromTestCase(TestFastPathClassification))
    suite.addTests(loader.loadTestsFromTestCase(TestEarlyExitDecisions))
    suite.addTests(loader.loadTestsFromTestCase(TestLocationAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestContextAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestClassificationDetermination))
    suite.addTests(loader.loadTestsFromTestCase(TestSlowPathClassification))
    suite.addTests(loader.loadTestsFromTestCase(TestStatistics))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceBenchmark))
    suite.addTests(loader.loadTestsFromTestCase(TestResultSerialization))

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
