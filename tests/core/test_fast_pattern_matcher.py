"""
Unit Tests for Fast Pattern Matcher

Tests for Aho-Corasick + Regex hybrid pattern matching including:
- Pattern matching accuracy
- Performance comparison (Aho-Corasick vs Regex)
- Deduplication
- Word boundary handling
- Case-insensitive matching

Author: Classmark Development Team
Date: 2025-11-10
"""

import unittest
import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.fast_pattern_matcher import (
    FastPatternMatcher,
    PatternMatch,
    PatternType,
    HAS_AHOCORASICK
)


class TestPatternMatch(unittest.TestCase):
    """Test PatternMatch dataclass"""

    def test_pattern_match_creation(self):
        """Test creating a PatternMatch"""
        match = PatternMatch(
            pattern="TOP_SECRET",
            category="level",
            start=0,
            end=10,
            matched_text="TOP SECRET",
            confidence=1.0
        )

        self.assertEqual(match.pattern, "TOP_SECRET")
        self.assertEqual(match.category, "level")
        self.assertEqual(match.start, 0)
        self.assertEqual(match.end, 10)
        self.assertEqual(match.matched_text, "TOP SECRET")
        self.assertEqual(match.confidence, 1.0)


@unittest.skipUnless(HAS_AHOCORASICK, "pyahocorasick not available")
class TestFastPatternMatcher(unittest.TestCase):
    """Test FastPatternMatcher class"""

    def setUp(self):
        """Set up test fixtures"""
        self.matcher = FastPatternMatcher()
        self.matcher.initialize()

    def test_initialization(self):
        """Test matcher initialization"""
        stats = self.matcher.get_statistics()

        self.assertGreater(stats['literal_patterns'], 0)
        self.assertGreater(stats['regex_patterns'], 0)
        self.assertTrue(stats['has_ahocorasick'])
        self.assertTrue(self.matcher._initialized)

    def test_literal_pattern_matching(self):
        """Test Aho-Corasick literal pattern matching"""
        text = "This document contains NOFORN information."

        matches = self.matcher.match_literal_patterns(text)

        # Should find NOFORN
        self.assertGreater(len(matches), 0)
        noforn_match = next((m for m in matches if m.pattern == 'NOFORN'), None)
        self.assertIsNotNone(noforn_match)
        self.assertEqual(noforn_match.matched_text, 'NOFORN')

    def test_case_insensitive_matching(self):
        """Test case-insensitive literal matching"""
        text_upper = "CLASSIFIED BY: John Doe"
        text_lower = "classified by: John Doe"
        text_mixed = "Classified By: John Doe"

        matches_upper = self.matcher.match_literal_patterns(text_upper)
        matches_lower = self.matcher.match_literal_patterns(text_lower)
        matches_mixed = self.matcher.match_literal_patterns(text_mixed)

        # All should find CLASSIFIED BY
        self.assertGreater(len(matches_upper), 0)
        self.assertGreater(len(matches_lower), 0)
        self.assertGreater(len(matches_mixed), 0)

    def test_word_boundary_detection(self):
        """Test word boundary handling"""
        # Should match: NOFORN with word boundaries
        text_valid = "TOP SECRET//NOFORN"
        matches_valid = self.matcher.match_literal_patterns(text_valid)
        noforn_valid = next((m for m in matches_valid if m.pattern == 'NOFORN'), None)
        self.assertIsNotNone(noforn_valid)

        # Should also match: NOFORN preceded by slash (not a letter)
        text_slash = "//NOFORN"
        matches_slash = self.matcher.match_literal_patterns(text_slash)
        noforn_slash = next((m for m in matches_slash if m.pattern == 'NOFORN'), None)
        self.assertIsNotNone(noforn_slash)

    def test_regex_pattern_matching(self):
        """Test regex pattern matching"""
        text = "TOP SECRET//NOFORN\n\n(TS) Classified content."

        matches = self.matcher.match_regex_patterns(text)

        # Should find TOP SECRET and portion marking
        self.assertGreater(len(matches), 0)

        # Check for specific patterns
        categories = [m.category for m in matches]
        self.assertIn('level', categories)

    def test_combined_matching(self):
        """Test combined Aho-Corasick + Regex matching"""
        text = """
        TOP SECRET//NOFORN

        CLASSIFIED BY: John Doe
        DERIVED FROM: ABC-123
        DECLASSIFY ON: 20300101

        (TS) This is classified information with FISA controls.
        """

        matches = self.matcher.match_all(text)

        # Should find multiple matches
        self.assertGreater(len(matches), 5)

        # Check for specific patterns
        patterns_found = {m.pattern for m in matches}
        self.assertIn('NOFORN', patterns_found)
        self.assertIn('CLASSIFIED_BY', patterns_found)
        self.assertIn('FISA', patterns_found)

    def test_deduplication(self):
        """Test match deduplication"""
        # Create overlapping matches manually
        matches = [
            PatternMatch("A", "cat1", 0, 10, "test text", 1.0),
            PatternMatch("B", "cat2", 5, 15, "text more", 0.9),  # Overlaps
            PatternMatch("C", "cat3", 20, 30, "other", 1.0),
        ]

        deduped = self.matcher._deduplicate_matches(matches)

        # Should keep first match (higher confidence) and third match
        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped[0].pattern, "A")
        self.assertEqual(deduped[1].pattern, "C")

    def test_control_markings(self):
        """Test detection of control markings"""
        text = "Document contains ORCON, IMCON, and RELIDO controls."

        matches = self.matcher.match_literal_patterns(text)

        control_matches = [m for m in matches if m.category == 'control']
        self.assertGreater(len(control_matches), 0)

        patterns = {m.pattern for m in control_matches}
        self.assertIn('ORCON', patterns)

    def test_authority_block_detection(self):
        """Test detection of classification authority blocks"""
        text = """
        CLASSIFIED BY: Jane Smith, OCA
        DERIVED FROM: Multiple sources
        DECLASSIFY ON: 20401231
        """

        matches = self.matcher.match_literal_patterns(text)

        authority_matches = [m for m in matches if m.category == 'authority']
        self.assertGreaterEqual(len(authority_matches), 3)

        patterns = {m.pattern for m in authority_matches}
        self.assertIn('CLASSIFIED_BY', patterns)
        self.assertIn('DERIVED_FROM', patterns)
        self.assertIn('DECLASSIFY_ON', patterns)

    def test_declassification_indicators(self):
        """Test detection of declassification indicators"""
        text = "This document was DECLASSIFIED on 2025-01-01. REDACTED portions remain."

        matches = self.matcher.match_literal_patterns(text)

        declassification_matches = [m for m in matches if m.category == 'declassification']
        self.assertGreater(len(declassification_matches), 0)

        patterns = {m.pattern for m in declassification_matches}
        self.assertIn('DECLASSIFIED', patterns)

    def test_legacy_cui_markings(self):
        """Test detection of legacy CUI markings"""
        text = "This document is marked FOUO and contains SBU information."

        matches = self.matcher.match_literal_patterns(text)

        legacy_matches = [m for m in matches if m.category == 'legacy_cui']
        self.assertGreater(len(legacy_matches), 0)

        patterns = {m.pattern for m in legacy_matches}
        self.assertIn('FOUO', patterns)


class TestPerformanceComparison(unittest.TestCase):
    """Performance comparison tests"""

    def setUp(self):
        """Set up test fixtures"""
        self.matcher = FastPatternMatcher()
        self.matcher.initialize()

        # Create test document
        self.test_doc = """
        TOP SECRET//SCI//NOFORN

        CLASSIFIED BY: John Doe, Senior Official
        DERIVED FROM: Intelligence Report IR-2025-001
        DECLASSIFY ON: 20450101
        REASON: 1.4(a)

        (TS//SI//NF) This compartmented intelligence report contains information
        about foreign intelligence activities. The material is classified under
        the authority of Executive Order 13526.

        SECRET//ORCON//IMCON

        (S) Additional information includes FISA-derived intelligence and RELIDO
        controlled data. This information is PROPIN and subject to originator
        control restrictions.

        CONFIDENTIAL//NOFORN

        (C) Supporting documentation includes references to historical operations
        that were DECLASSIFIED in 2020 and subsequently REDACTED for public release.

        FOR OFFICIAL USE ONLY (FOUO)
        SENSITIVE BUT UNCLASSIFIED (SBU)
        LAW ENFORCEMENT SENSITIVE (LES)

        END OF DOCUMENT
        """ * 10  # Repeat 10 times for performance testing

    @unittest.skipUnless(HAS_AHOCORASICK, "pyahocorasick not available")
    def test_literal_pattern_performance(self):
        """Benchmark Aho-Corasick literal pattern matching"""
        iterations = 100

        start = time.time()
        for _ in range(iterations):
            matches = self.matcher.match_literal_patterns(self.test_doc)
        elapsed = time.time() - start

        avg_time_ms = (elapsed / iterations) * 1000

        # Should be very fast (< 1ms per document on average)
        self.assertLess(avg_time_ms, 5.0)  # Allow 5ms margin

        print(f"\nAho-Corasick Performance: {avg_time_ms:.3f}ms avg ({iterations} iterations)")
        print(f"  Matches found: {len(matches)}")

    @unittest.skipUnless(HAS_AHOCORASICK, "pyahocorasick not available")
    def test_regex_pattern_performance(self):
        """Benchmark regex pattern matching"""
        iterations = 100

        start = time.time()
        for _ in range(iterations):
            matches = self.matcher.match_regex_patterns(self.test_doc)
        elapsed = time.time() - start

        avg_time_ms = (elapsed / iterations) * 1000

        print(f"\nRegex Performance: {avg_time_ms:.3f}ms avg ({iterations} iterations)")
        print(f"  Matches found: {len(matches)}")

    @unittest.skipUnless(HAS_AHOCORASICK, "pyahocorasick not available")
    def test_combined_pattern_performance(self):
        """Benchmark combined matching with deduplication"""
        iterations = 100

        start = time.time()
        for _ in range(iterations):
            matches = self.matcher.match_all(self.test_doc)
        elapsed = time.time() - start

        avg_time_ms = (elapsed / iterations) * 1000

        # Combined should still be fast
        self.assertLess(avg_time_ms, 10.0)  # Allow 10ms margin

        print(f"\nCombined Performance: {avg_time_ms:.3f}ms avg ({iterations} iterations)")
        print(f"  Deduplicated matches: {len(matches)}")

    @unittest.skipUnless(HAS_AHOCORASICK, "pyahocorasick not available")
    def test_performance_comparison(self):
        """Compare Aho-Corasick vs Regex performance"""
        iterations = 50

        # Benchmark literal patterns
        start = time.time()
        for _ in range(iterations):
            self.matcher.match_literal_patterns(self.test_doc)
        literal_time = time.time() - start

        # Benchmark regex patterns
        start = time.time()
        for _ in range(iterations):
            self.matcher.match_regex_patterns(self.test_doc)
        regex_time = time.time() - start

        speedup = regex_time / literal_time if literal_time > 0 else 0

        print(f"\nPerformance Comparison ({iterations} iterations):")
        print(f"  Aho-Corasick: {literal_time*1000:.2f}ms total ({literal_time/iterations*1000:.3f}ms avg)")
        print(f"  Regex:        {regex_time*1000:.2f}ms total ({regex_time/iterations*1000:.3f}ms avg)")
        print(f"  Speedup:      {speedup:.1f}x")

        # Aho-Corasick should be faster
        self.assertGreater(speedup, 1.0)


def run_tests():
    """Run all tests and print results"""
    print("=" * 70)
    print("Fast Pattern Matcher - Unit Tests")
    print("=" * 70)
    print()

    # Check dependencies
    print("Dependency Status:")
    print(f"  pyahocorasick: {HAS_AHOCORASICK}")
    print()

    if not HAS_AHOCORASICK:
        print("⚠️  pyahocorasick not available - some tests will be skipped")
        print("Install: pip install pyahocorasick")
        print()

    # Run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestPatternMatch))
    suite.addTests(loader.loadTestsFromTestCase(TestFastPatternMatcher))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceComparison))

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
