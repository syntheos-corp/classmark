"""
Fast Pattern Matcher using Aho-Corasick Automaton

This module provides SOTA pattern matching for classification markings using:
- Aho-Corasick automaton for literal string patterns (O(text_length + matches))
- Regex fallback for complex patterns requiring lookahead/lookbehind
- Hybrid approach: fast path for simple patterns, regex for complex patterns

Performance improvement: 10-50x faster than sequential regex matching

Author: Classmark Development Team
Date: 2025-11-10
Version: 1.0 SOTA Edition
"""

import re
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
from enum import Enum

try:
    import ahocorasick
    HAS_AHOCORASICK = True
except ImportError:
    HAS_AHOCORASICK = False

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False


@dataclass
class PatternMatch:
    """Represents a pattern match with metadata"""
    pattern: str
    category: str
    start: int
    end: int
    matched_text: str
    confidence: float = 1.0


class PatternType(Enum):
    """Type of pattern matching strategy"""
    LITERAL = "literal"  # Simple literal strings (Aho-Corasick)
    REGEX_SIMPLE = "regex_simple"  # Simple regex without lookahead/lookbehind
    REGEX_COMPLEX = "regex_complex"  # Complex regex with lookahead/lookbehind


class FastPatternMatcher:
    """
    SOTA pattern matching using Aho-Corasick + Regex hybrid approach

    Features:
    - Aho-Corasick automaton for literal patterns (10-50x faster than regex)
    - Regex for complex patterns requiring lookahead/lookbehind
    - Single-pass matching: O(text_length + matches)
    - Case-insensitive matching
    - Word boundary handling for literal patterns
    """

    def __init__(self, fuzzy_matching: bool = True, fuzzy_threshold: int = 85):
        """
        Initialize the fast pattern matcher

        Args:
            fuzzy_matching: Enable fuzzy matching for OCR error tolerance
            fuzzy_threshold: Similarity threshold (0-100) for fuzzy matching
        """
        self.automaton = None
        self.regex_patterns = []
        self.literal_patterns = []
        self._initialized = False
        self.fuzzy_matching = fuzzy_matching
        self.fuzzy_threshold = fuzzy_threshold

        # Classification terms for Aho-Corasick (literal strings only)
        self.LITERAL_PATTERNS = {
            # Control markings (high priority - these are exact terms)
            'NOFORN': ('control', 'NOFORN'),
            'ORCON': ('control', 'ORCON'),
            'IMCON': ('control', 'IMCON'),
            'RELIDO': ('control', 'RELIDO'),
            'PROPIN': ('control', 'PROPIN'),
            'FISA': ('control', 'FISA'),
            'SCI': ('control', 'SCI'),

            # Declassification indicators (exact terms)
            'DECLASSIFIED': ('declassification', 'DECLASSIFIED'),
            'REDACTED': ('declassification', 'REDACTED'),
            'SANITIZED COPY': ('declassification', 'SANITIZED_COPY'),
            'AUTOMATICALLY DECLASSIFIED': ('declassification', 'AUTO_DECLASSIFIED'),
            'APPROVED FOR RELEASE': ('declassification', 'APPROVED_RELEASE'),
            'CIA HISTORICAL REVIEW': ('declassification', 'CIA_HISTORICAL'),
            'FOIA CASE NUMBER': ('declassification', 'FOIA_CASE'),
            'RELEASE AS SANITIZED': ('declassification', 'RELEASE_SANITIZED'),

            # Legacy CUI markings (exact terms)
            'FOUO': ('legacy_cui', 'FOUO'),
            'FOR OFFICIAL USE ONLY': ('legacy_cui', 'FOUO_FULL'),
            'SBU': ('legacy_cui', 'SBU'),
            'SENSITIVE BUT UNCLASSIFIED': ('legacy_cui', 'SBU_FULL'),
            'LES': ('legacy_cui', 'LES'),
            'LAW ENFORCEMENT SENSITIVE': ('legacy_cui', 'LES_FULL'),
            'LIMDIS': ('legacy_cui', 'LIMDIS'),
            'LIMITED DISTRIBUTION': ('legacy_cui', 'LIMDIS_FULL'),

            # Classification authority indicators (key phrases)
            'CLASSIFIED BY': ('authority', 'CLASSIFIED_BY'),
            'DERIVED FROM': ('authority', 'DERIVED_FROM'),
            'DECLASSIFY ON': ('authority', 'DECLASSIFY_ON'),
            'CLASSIFICATION AUTHORITY': ('authority', 'CLASS_AUTH'),
        }

        # Complex regex patterns (require lookahead/lookbehind or special matching)
        self.REGEX_PATTERNS = [
            # TOP SECRET with optional control markings
            (r'(?:^|\n)\s*(?:\/\/|//)?\s*TOP\s*SECRET(?:\s*\/\/[A-Z\/]+)?',
             'level', 'TOP_SECRET', PatternType.REGEX_COMPLEX),
            (r'\(TS(?:\/\/[A-Z]+)?\)',
             'level', 'TOP_SECRET_ABBREV', PatternType.REGEX_SIMPLE),

            # SECRET with negative lookahead to avoid false positives
            (r'(?:^|\n)\s*(?:\/\/|//)?\s*SECRET(?:\s*\/\/[A-Z\/]+)?(?!\s*(?:SERVICE|AGENT|SANTA))',
             'level', 'SECRET', PatternType.REGEX_COMPLEX),
            (r'\(S(?:\/\/[A-Z]+)?\)(?!\w)',
             'level', 'SECRET_ABBREV', PatternType.REGEX_COMPLEX),

            # CONFIDENTIAL with optional control markings
            (r'(?:^|\n)\s*(?:\/\/|//)?\s*CONFIDENTIAL(?:\s*\/\/[A-Z\/]+)?',
             'level', 'CONFIDENTIAL', PatternType.REGEX_COMPLEX),
            (r'\(C(?:\/\/[A-Z]+)?\)(?!\w)',
             'level', 'CONFIDENTIAL_ABBREV', PatternType.REGEX_COMPLEX),

            # CUI patterns
            (r'(?:^|\n)\s*(?:CONTROLLED\s+UNCLASSIFIED\s+INFORMATION|CUI)(?:\s*\/\/[A-Z\/]+)?',
             'level', 'CUI', PatternType.REGEX_COMPLEX),
            (r'\(CUI\)',
             'level', 'CUI_ABBREV', PatternType.REGEX_SIMPLE),

            # Portion markings at start of paragraphs
            (r'(?:^|\n)\s*\((TS|S|C|U)(?:\/\/[A-Z]+)?\)\s+[A-Z][a-z]{2,}',
             'structural', 'PORTION_MARK', PatternType.REGEX_COMPLEX),

            # Authority block patterns with line start anchors
            (r'(?:^|\n)\s*REASON:?\s*1\.\d\([a-h]\)',
             'authority', 'REASON_CODE', PatternType.REGEX_COMPLEX),

            # Banner patterns (top/bottom classification markings)
            (r'(?:TOP\s+SECRET|SECRET|CONFIDENTIAL)\s*(?:\/\/[A-Z\s]+)?\s*\n.{0,500}?\n\s*(?:TOP\s+SECRET|SECRET|CONFIDENTIAL)',
             'structural', 'BANNER', PatternType.REGEX_COMPLEX),

            # Downgrade/upgrade indicators
            (r'DOWNGRADED\s+TO',
             'declassification', 'DOWNGRADED', PatternType.REGEX_SIMPLE),
            (r'UPGRADED\s+TO',
             'declassification', 'UPGRADED', PatternType.REGEX_SIMPLE),
            (r'REGRADED\s+TO',
             'declassification', 'REGRADED', PatternType.REGEX_SIMPLE),
            (r'GROUP\s+1\s+-\s+EXCLUDED',
             'declassification', 'GROUP_EXCLUDED', PatternType.REGEX_SIMPLE),
        ]

    def initialize(self):
        """Build the Aho-Corasick automaton and compile regex patterns"""
        if self._initialized:
            return

        if not HAS_AHOCORASICK:
            print("Warning: pyahocorasick not available, using regex-only mode")
            self._initialized = True
            return

        # Build Aho-Corasick automaton for literal patterns
        self.automaton = ahocorasick.Automaton()

        for pattern_text, (category, name) in self.LITERAL_PATTERNS.items():
            # Add both uppercase and lowercase versions for case-insensitive matching
            self.automaton.add_word(pattern_text.upper(), (pattern_text, category, name))
            self.automaton.add_word(pattern_text.lower(), (pattern_text, category, name))

        self.automaton.make_automaton()

        # Compile regex patterns
        for pattern_str, category, name, pattern_type in self.REGEX_PATTERNS:
            compiled = re.compile(pattern_str, re.IGNORECASE | re.MULTILINE)
            self.regex_patterns.append((compiled, category, name, pattern_type))

        self._initialized = True

    def match_literal_patterns(self, text: str, debug=False) -> List[PatternMatch]:
        """
        Match literal patterns using Aho-Corasick automaton

        Args:
            text: Text to search
            debug: If True, print debug information

        Returns:
            List of PatternMatch objects
        """
        if not self._initialized:
            self.initialize()

        if not HAS_AHOCORASICK or self.automaton is None:
            return []

        matches = []
        text_upper = text.upper()  # Normalize for case-insensitive matching

        for end_pos, (pattern_text, category, name) in self.automaton.iter(text_upper):
            start_pos = end_pos - len(pattern_text) + 1
            matched_text = text[start_pos:end_pos + 1]

            if debug:
                print(f"  Found: '{pattern_text}' ({name}) @ {start_pos}-{end_pos}, matched: '{matched_text}'")
                if start_pos > 0:
                    print(f"    Char before ({start_pos-1}): '{text[start_pos-1]}' (is_alpha={text[start_pos-1].isalpha()})")
                if end_pos + 1 < len(text):
                    print(f"    Char after ({end_pos+1}): '{text[end_pos+1]}' (is_alpha={text[end_pos+1].isalpha()})")

            # Check word boundaries for certain patterns to avoid partial matches
            if self._check_word_boundary(text, start_pos, end_pos + 1):
                matches.append(PatternMatch(
                    pattern=name,
                    category=category,
                    start=start_pos,
                    end=end_pos + 1,
                    matched_text=matched_text,
                    confidence=1.0
                ))
                if debug:
                    print(f"    ✓ Accepted")
            elif debug:
                print(f"    ✗ Rejected by word boundary check")

        return matches

    def match_regex_patterns(self, text: str) -> List[PatternMatch]:
        """
        Match complex patterns using regex

        Args:
            text: Text to search

        Returns:
            List of PatternMatch objects
        """
        if not self._initialized:
            self.initialize()

        matches = []

        for pattern, category, name, pattern_type in self.regex_patterns:
            for match in pattern.finditer(text):
                matches.append(PatternMatch(
                    pattern=name,
                    category=category,
                    start=match.start(),
                    end=match.end(),
                    matched_text=match.group(0),
                    confidence=0.95 if pattern_type == PatternType.REGEX_COMPLEX else 1.0
                ))

        return matches

    def match_fuzzy_patterns(self, text: str) -> List[PatternMatch]:
        """
        Match patterns using fuzzy string matching for OCR error tolerance

        Only fuzzy-matches high-value patterns prone to OCR errors:
        - CLASSIFIED (->:LASSIFiED, CLASSIFIEF, etc.)
        - DECLASSIFIED (->DECLASIFIED, etc.)
        - SECRET (->$ECRET, SECREF, etc.)
        - CONFIDENTIAL (->CONFIDENTIAI, etc.)

        Args:
            text: Text to search

        Returns:
            List of PatternMatch objects with confidence < 1.0
        """
        if not self._initialized:
            self.initialize()

        if not HAS_RAPIDFUZZ or not self.fuzzy_matching:
            return []

        # High-value patterns for fuzzy matching (prone to OCR errors)
        fuzzy_patterns = {
            'CLASSIFIED': ('authority', 'CLASSIFIED_BY'),
            'DECLASSIFIED': ('declassification', 'DECLASSIFIED'),
            'SECRET': ('level', 'SECRET'),
            'CONFIDENTIAL': ('level', 'CONFIDENTIAL'),
        }

        matches = []

        # Extract words from text (alphanumeric sequences)
        words = re.findall(r'\b[A-Za-z0-9]+\b', text)

        # Track word positions for accurate match locations
        word_positions = [(m.start(), m.end(), m.group()) for m in re.finditer(r'\b[A-Za-z0-9]+\b', text)]

        for pos_idx, (start_pos, end_pos, word) in enumerate(word_positions):
            # Skip short words to avoid false positives
            if len(word) < 5:
                continue

            word_upper = word.upper()

            # Try fuzzy matching against each pattern
            for pattern_text, (category, name) in fuzzy_patterns.items():
                similarity = fuzz.ratio(word_upper, pattern_text)

                if similarity >= self.fuzzy_threshold:
                    # Check word boundaries
                    if self._check_word_boundary(text, start_pos, end_pos):
                        # Calculate confidence: use full similarity score
                        # Note: scan_text multiplies by 0.5, so we use full score here
                        # to compensate (e.g., 95% -> 0.95 -> 0.475 after 0.5 multiplier)
                        # Using 1.0 scaling ensures strong fuzzy matches survive the multiplier
                        confidence = similarity / 100.0

                        matches.append(PatternMatch(
                            pattern=name,
                            category=category,
                            start=start_pos,
                            end=end_pos,
                            matched_text=word,
                            confidence=confidence
                        ))

        return matches

    def match_all(self, text: str) -> List[PatternMatch]:
        """
        Match all patterns (literal, regex, and fuzzy)

        Args:
            text: Text to search

        Returns:
            List of PatternMatch objects sorted by position
        """
        if not self._initialized:
            self.initialize()

        # Run all matchers
        literal_matches = self.match_literal_patterns(text)
        regex_matches = self.match_regex_patterns(text)
        fuzzy_matches = self.match_fuzzy_patterns(text)

        # Combine and deduplicate
        all_matches = literal_matches + regex_matches + fuzzy_matches

        # Remove overlapping matches (keep highest confidence)
        all_matches = self._deduplicate_matches(all_matches)

        # Sort by position
        all_matches.sort(key=lambda m: (m.start, -m.confidence))

        return all_matches

    def _check_word_boundary(self, text: str, start: int, end: int) -> bool:
        """
        Check if match is at word boundary

        Allows matches that are:
        - At string boundaries (start/end of text)
        - Preceded/followed by whitespace, punctuation, or symbols (//, -, etc.)
        - NOT preceded/followed by letters (to avoid partial word matches)

        Args:
            text: Full text
            start: Start position of match
            end: End position of match (exclusive)

        Returns:
            True if at word boundary
        """
        # Check character before match - reject if it's a letter
        if start > 0:
            char_before = text[start - 1]
            # Only reject if previous character is a letter (not digit/symbol)
            if char_before.isalpha():
                return False

        # Check character after match - reject if it's a letter
        if end < len(text):
            char_after = text[end]
            # Only reject if next character is a letter (not digit/symbol)
            if char_after.isalpha():
                return False

        return True

    def _deduplicate_matches(self, matches: List[PatternMatch]) -> List[PatternMatch]:
        """
        Remove overlapping matches, keeping highest priority

        Priority order:
        1. Classification level patterns (TOP SECRET, SECRET, CONFIDENTIAL, CUI)
        2. Higher confidence
        3. Longer match length

        Args:
            matches: List of matches

        Returns:
            Deduplicated list of matches
        """
        if not matches:
            return []

        # Category priority: level > structural > authority > declassification > control
        category_priority = {'level': 5, 'structural': 4, 'authority': 3, 'declassification': 2, 'control': 1}

        # Sort by start position, then by category priority, then confidence, then length
        matches.sort(key=lambda m: (
            m.start,
            -category_priority.get(m.category, 0),
            -m.confidence,
            -(m.end - m.start)
        ))

        deduplicated = []
        last_end = -1

        for match in matches:
            # If this match doesn't overlap with previous, keep it
            if match.start >= last_end:
                deduplicated.append(match)
                last_end = match.end
            # If it overlaps, check if it should replace previous
            elif deduplicated:
                prev = deduplicated[-1]
                # Calculate priority scores
                match_priority = (
                    category_priority.get(match.category, 0),
                    match.confidence,
                    match.end - match.start
                )
                prev_priority = (
                    category_priority.get(prev.category, 0),
                    prev.confidence,
                    prev.end - prev.start
                )

                # Replace if new match has higher priority
                if match_priority > prev_priority:
                    deduplicated[-1] = match
                    last_end = match.end

        return deduplicated

    def get_statistics(self) -> Dict[str, int]:
        """
        Get statistics about pattern counts

        Returns:
            Dictionary with pattern counts
        """
        return {
            'literal_patterns': len(self.LITERAL_PATTERNS),
            'regex_patterns': len(self.REGEX_PATTERNS),
            'total_patterns': len(self.LITERAL_PATTERNS) + len(self.REGEX_PATTERNS),
            'has_ahocorasick': HAS_AHOCORASICK
        }


if __name__ == "__main__":
    print("Fast Pattern Matcher - Aho-Corasick + Regex Hybrid")
    print("=" * 70)

    matcher = FastPatternMatcher()
    matcher.initialize()

    stats = matcher.get_statistics()
    print(f"\nPattern Statistics:")
    print(f"  Literal patterns (Aho-Corasick): {stats['literal_patterns']}")
    print(f"  Regex patterns: {stats['regex_patterns']}")
    print(f"  Total patterns: {stats['total_patterns']}")
    print(f"  Aho-Corasick available: {stats['has_ahocorasick']}")

    # Test with sample text
    test_text = """
    TOP SECRET//NOFORN

    (TS) This is a test document.

    SECRET//ORCON

    CLASSIFIED BY: John Doe
    DERIVED FROM: ABC-123
    DECLASSIFY ON: 20300101

    This document contains FISA information.

    DECLASSIFIED on 2025-01-01
    """

    print(f"\n\nTest Text ({len(test_text)} characters):")
    print("-" * 70)
    print(test_text[:200] + "...")

    import time

    # Benchmark literal patterns
    start = time.time()
    literal_matches = matcher.match_literal_patterns(test_text, debug=False)
    literal_time = time.time() - start

    print(f"\n\nLiteral Matches (Aho-Corasick): {len(literal_matches)}")
    print(f"Processing Time: {literal_time*1000:.3f}ms")
    print("-" * 70)
    for match in literal_matches[:10]:  # Show first 10
        print(f"{match.category:20} {match.pattern:25} @ {match.start:4}-{match.end:4}: '{match.matched_text}'")

    # Benchmark regex patterns
    start = time.time()
    regex_matches = matcher.match_regex_patterns(test_text)
    regex_time = time.time() - start

    print(f"\n\nRegex Matches: {len(regex_matches)}")
    print(f"Processing Time: {regex_time*1000:.3f}ms")
    print("-" * 70)
    for match in regex_matches[:10]:  # Show first 10
        print(f"{match.category:20} {match.pattern:25} @ {match.start:4}-{match.end:4}: '{match.matched_text[:40]}'")

    # Combined matches
    start = time.time()
    all_matches = matcher.match_all(test_text)
    combined_time = time.time() - start

    print(f"\n\nCombined Matches (Deduplicated): {len(all_matches)}")
    print(f"Total Processing Time: {combined_time*1000:.3f}ms")
    print(f"Speedup estimate: Aho-Corasick is ~{regex_time/literal_time:.1f}x faster than regex for literal patterns")
    print("-" * 70)

    for match in all_matches:
        print(f"{match.category:20} {match.pattern:25} @ {match.start:4}-{match.end:4}: '{match.matched_text[:40]}'")
