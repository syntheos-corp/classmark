"""
Hybrid Two-Stage Classification Architecture

This module implements a dual-path classification system for optimal
performance and accuracy:

Fast Path: Aho-Corasick + location analysis for instant detection
Slow Path: YOLO + LayoutLMv3 + context + LLM for thorough analysis

The system routes documents to the appropriate path based on early
confidence thresholds:
- >95% confidence → ACCEPT (skip slow path)
- <10% confidence → REJECT (skip slow path)
- 10-95% confidence → Route to slow path

Target performance: 200-500 docs/min throughput for fast path

Author: Classmark Development Team
Date: 2025-11-10
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json

# Fast pattern matching
try:
    from fast_pattern_matcher import FastPatternMatcher, HAS_AHOCORASICK
    HAS_FAST_MATCHER = True
except ImportError:
    HAS_FAST_MATCHER = False
    HAS_AHOCORASICK = False

# Visual pattern detection
try:
    from visual_pattern_detector import VisualPatternDetector, HAS_LAYOUTLM
    HAS_VISUAL_DETECTOR = True
except ImportError:
    HAS_VISUAL_DETECTOR = False
    HAS_LAYOUTLM = False

# OCR capabilities
try:
    from yolo_ocr import YOLOTextDetector, HAS_YOLO, HAS_TESSERACT
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    HAS_YOLO = False
    HAS_TESSERACT = False

# LLM integration (optional)
try:
    from llm_integration import LLMClassifier
    HAS_LLM = True
except ImportError:
    HAS_LLM = False


class ClassificationPath(Enum):
    """Classification path taken"""
    FAST_PATH = "fast_path"
    SLOW_PATH = "slow_path"
    HYBRID = "hybrid"  # Both paths used


class ConfidenceDecision(Enum):
    """Early exit decision"""
    ACCEPT = "accept"  # High confidence, skip slow path
    REJECT = "reject"  # Low confidence, skip slow path
    UNCERTAIN = "uncertain"  # Route to slow path


@dataclass
class PathMetrics:
    """Metrics for a single path execution"""
    path_type: ClassificationPath
    processing_time: float  # seconds
    confidence: float
    num_matches: int
    decision: ConfidenceDecision
    early_exit: bool = False


@dataclass
class HybridResult:
    """Result from hybrid classification"""
    has_classification: bool
    classification_level: Optional[str] = None
    confidence: float = 0.0
    matches: List[Any] = field(default_factory=list)
    path_taken: ClassificationPath = ClassificationPath.FAST_PATH
    fast_path_metrics: Optional[PathMetrics] = None
    slow_path_metrics: Optional[PathMetrics] = None
    total_time: float = 0.0
    early_exit: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'has_classification': self.has_classification,
            'classification_level': self.classification_level,
            'confidence': self.confidence,
            'num_matches': len(self.matches),
            'path_taken': self.path_taken.value,
            'total_time': self.total_time,
            'early_exit': self.early_exit,
            'fast_path_time': self.fast_path_metrics.processing_time if self.fast_path_metrics else None,
            'slow_path_time': self.slow_path_metrics.processing_time if self.slow_path_metrics else None
        }


class HybridClassifier:
    """
    Hybrid two-stage classification system

    Fast Path:
    - Aho-Corasick pattern matching (O(n) complexity)
    - Location-based confidence boosting
    - Early exit thresholds

    Slow Path:
    - YOLO OCR for scanned documents
    - LayoutLMv3 visual pattern analysis
    - Context-based confidence refinement
    - Optional LLM verification
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        use_gpu: bool = True,
        enable_llm: bool = False
    ):
        """
        Initialize hybrid classifier

        Args:
            config: Configuration dictionary with thresholds and settings
            use_gpu: Whether to use GPU for visual detection
            enable_llm: Whether to enable LLM verification (slow but thorough)
        """
        self.config = config or {}
        self.use_gpu = use_gpu
        self.enable_llm = enable_llm

        # Early exit thresholds
        self.accept_threshold = self.config.get('accept_threshold', 0.95)
        self.reject_threshold = self.config.get('reject_threshold', 0.10)

        # Confidence thresholds for classification detection
        self.min_confidence = self.config.get('min_confidence', 0.3)

        # Initialize components
        self._initialize_components()

        # Performance tracking
        self.stats = {
            'total_documents': 0,
            'fast_path_only': 0,
            'slow_path_invoked': 0,
            'early_accept': 0,
            'early_reject': 0,
            'total_fast_time': 0.0,
            'total_slow_time': 0.0
        }

    def _initialize_components(self):
        """Initialize classification components"""

        # Fast pattern matcher (Aho-Corasick)
        if HAS_FAST_MATCHER:
            self.fast_matcher = FastPatternMatcher()
            self.fast_matcher.initialize()
            print(f"✓ Fast pattern matcher initialized (Aho-Corasick: {HAS_AHOCORASICK})", file=sys.stderr)
        else:
            self.fast_matcher = None
            print("⚠ Fast pattern matcher not available", file=sys.stderr)

        # Visual pattern detector (LayoutLMv3)
        if HAS_VISUAL_DETECTOR:
            self.visual_detector = VisualPatternDetector(
                use_gpu=self.use_gpu,
                dpi=200,
                max_pages=self.config.get('max_pages_visual', None)
            )
            print(f"✓ Visual pattern detector available (LayoutLMv3: {HAS_LAYOUTLM})", file=sys.stderr)
        else:
            self.visual_detector = None
            print("⚠ Visual pattern detector not available", file=sys.stderr)

        # OCR detector (YOLO + Tesseract)
        if HAS_OCR:
            self.ocr_detector = YOLOTextDetector(
                use_gpu=self.use_gpu,
                dpi_for_conversion=300,
                min_dpi_standard=200
            )
            print(f"✓ OCR detector available (YOLO: {HAS_YOLO}, Tesseract: {HAS_TESSERACT})", file=sys.stderr)
        else:
            self.ocr_detector = None
            print("⚠ OCR detector not available", file=sys.stderr)

        # LLM classifier (optional)
        if self.enable_llm and HAS_LLM:
            self.llm_classifier = LLMClassifier()
            print("✓ LLM classifier available", file=sys.stderr)
        else:
            self.llm_classifier = None

    def classify(
        self,
        text: str,
        file_path: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> HybridResult:
        """
        Classify document using hybrid two-stage architecture

        Args:
            text: Document text content
            file_path: Optional path to document file (for PDFs)
            metadata: Optional document metadata

        Returns:
            HybridResult with classification details and path metrics
        """
        start_time = time.time()
        self.stats['total_documents'] += 1

        # Stage 1: Fast Path
        fast_result = self._fast_path_classify(text, file_path, metadata)

        # Check early exit decision
        decision = self._evaluate_confidence(fast_result)

        if decision == ConfidenceDecision.ACCEPT:
            # High confidence - accept and exit
            self.stats['fast_path_only'] += 1
            self.stats['early_accept'] += 1
            self.stats['total_fast_time'] += fast_result.fast_path_metrics.processing_time

            fast_result.early_exit = True
            fast_result.total_time = time.time() - start_time
            return fast_result

        elif decision == ConfidenceDecision.REJECT:
            # Low confidence - reject and exit
            self.stats['fast_path_only'] += 1
            self.stats['early_reject'] += 1
            self.stats['total_fast_time'] += fast_result.fast_path_metrics.processing_time

            fast_result.early_exit = True
            fast_result.total_time = time.time() - start_time
            return fast_result

        else:
            # Uncertain - route to slow path
            self.stats['slow_path_invoked'] += 1

            slow_result = self._slow_path_classify(text, file_path, metadata, fast_result)

            self.stats['total_fast_time'] += fast_result.fast_path_metrics.processing_time
            self.stats['total_slow_time'] += slow_result.slow_path_metrics.processing_time

            slow_result.total_time = time.time() - start_time
            return slow_result

    def _fast_path_classify(
        self,
        text: str,
        file_path: Optional[str],
        metadata: Optional[Dict]
    ) -> HybridResult:
        """
        Fast path classification using Aho-Corasick + location analysis

        Target: <10ms per document
        """
        start_time = time.time()

        if not self.fast_matcher:
            # Fallback if fast matcher not available
            return HybridResult(
                has_classification=False,
                confidence=0.0,
                path_taken=ClassificationPath.FAST_PATH,
                fast_path_metrics=PathMetrics(
                    path_type=ClassificationPath.FAST_PATH,
                    processing_time=time.time() - start_time,
                    confidence=0.0,
                    num_matches=0,
                    decision=ConfidenceDecision.REJECT
                )
            )

        # Run fast pattern matching
        matches = self.fast_matcher.match_all(text)

        # Location-based confidence analysis
        enhanced_matches = self._analyze_match_locations(matches, text)

        # Determine classification level and confidence
        classification_level, confidence = self._determine_classification(enhanced_matches)

        processing_time = time.time() - start_time

        # Create result
        result = HybridResult(
            has_classification=len(enhanced_matches) > 0,
            classification_level=classification_level,
            confidence=confidence,
            matches=enhanced_matches,
            path_taken=ClassificationPath.FAST_PATH,
            fast_path_metrics=PathMetrics(
                path_type=ClassificationPath.FAST_PATH,
                processing_time=processing_time,
                confidence=confidence,
                num_matches=len(enhanced_matches),
                decision=ConfidenceDecision.UNCERTAIN  # Will be evaluated by caller
            )
        )

        return result

    def _slow_path_classify(
        self,
        text: str,
        file_path: Optional[str],
        metadata: Optional[Dict],
        fast_result: HybridResult
    ) -> HybridResult:
        """
        Slow path classification using YOLO + LayoutLMv3 + context + LLM

        Target: Thorough analysis for uncertain cases
        """
        start_time = time.time()

        # Start with fast path results
        all_matches = fast_result.matches.copy()
        combined_confidence = fast_result.confidence

        # Check if we need OCR (scanned document or low word count)
        needs_ocr = False
        if file_path and file_path.lower().endswith('.pdf'):
            word_count = len(text.split())
            if word_count < 50:  # Likely scanned
                needs_ocr = True

        # Layer 1: OCR enhancement (if needed)
        if needs_ocr and self.ocr_detector and file_path:
            try:
                ocr_results = self.ocr_detector.process_pdf(file_path)

                # Extract additional text
                additional_text = "\n".join([r.text for r in ocr_results])

                # Re-run fast matching on OCR text
                if self.fast_matcher:
                    ocr_matches = self.fast_matcher.match_all(additional_text)
                    all_matches.extend(self._analyze_match_locations(ocr_matches, additional_text))

            except Exception as e:
                print(f"  ⚠ OCR processing error: {e}", file=sys.stderr)

        # Layer 2: Visual pattern detection (LayoutLMv3)
        if self.visual_detector and file_path and file_path.lower().endswith('.pdf'):
            try:
                # Initialize if not already done
                if not self.visual_detector._initialized:
                    self.visual_detector.initialize()

                visual_matches = self.visual_detector.detect_visual_patterns(file_path, text)

                # Boost confidence for matches with visual confirmation
                for visual_match in visual_matches:
                    # Check if this visual match confirms existing match
                    confirmed = False
                    for match in all_matches:
                        if self._matches_overlap(match, visual_match):
                            # Boost confidence
                            match['confidence'] = min(1.0, match['confidence'] + 0.25)
                            confirmed = True
                            break

                    # Add new visual-only matches
                    if not confirmed:
                        all_matches.append({
                            'text': visual_match.text,
                            'confidence': visual_match.visual_confidence * 0.8,
                            'pattern_type': visual_match.pattern_type,
                            'category': 'visual',
                            'location': 'header' if visual_match.region.is_header else 'body'
                        })

            except Exception as e:
                print(f"  ⚠ Visual pattern detection error: {e}", file=sys.stderr)

        # Layer 3: Context analysis
        all_matches = self._context_analysis(all_matches, text)

        # Layer 4: LLM verification (optional, very slow)
        if self.enable_llm and self.llm_classifier:
            try:
                llm_result = self.llm_classifier.verify_classification(
                    text,
                    all_matches,
                    metadata
                )

                # Use LLM confidence to adjust overall confidence
                if llm_result.get('confidence'):
                    combined_confidence = (combined_confidence + llm_result['confidence']) / 2

            except Exception as e:
                print(f"  ⚠ LLM verification error: {e}", file=sys.stderr)

        # Final classification determination
        classification_level, final_confidence = self._determine_classification(all_matches)

        processing_time = time.time() - start_time

        # Create result
        result = HybridResult(
            has_classification=len(all_matches) > 0,
            classification_level=classification_level,
            confidence=final_confidence,
            matches=all_matches,
            path_taken=ClassificationPath.HYBRID,
            fast_path_metrics=fast_result.fast_path_metrics,
            slow_path_metrics=PathMetrics(
                path_type=ClassificationPath.SLOW_PATH,
                processing_time=processing_time,
                confidence=final_confidence,
                num_matches=len(all_matches),
                decision=ConfidenceDecision.ACCEPT if final_confidence > self.min_confidence else ConfidenceDecision.REJECT
            )
        )

        return result

    def _evaluate_confidence(self, result: HybridResult) -> ConfidenceDecision:
        """
        Evaluate confidence and make early exit decision

        Args:
            result: Fast path result

        Returns:
            ConfidenceDecision (ACCEPT, REJECT, or UNCERTAIN)
        """
        confidence = result.confidence

        if confidence >= self.accept_threshold:
            return ConfidenceDecision.ACCEPT
        elif confidence <= self.reject_threshold:
            return ConfidenceDecision.REJECT
        else:
            return ConfidenceDecision.UNCERTAIN

    def _analyze_match_locations(
        self,
        matches: List[Any],
        text: str
    ) -> List[Dict]:
        """
        Analyze match locations to boost confidence

        Location heuristics:
        - Header matches: +0.15 confidence
        - Footer matches: +0.10 confidence
        - Body matches: +0.05 confidence
        - Banner patterns: +0.20 confidence
        """
        enhanced_matches = []
        lines = text.split('\n')
        total_lines = len(lines)

        for match in matches:
            match_dict = {
                'text': match.matched_text if hasattr(match, 'matched_text') else str(match),
                'confidence': 0.5,  # Base confidence
                'pattern_type': match.pattern if hasattr(match, 'pattern') else 'unknown',  # Pattern name is in 'pattern' attribute
                'category': match.category if hasattr(match, 'category') else 'unknown',
                'location': 'body'
            }

            # Find match position in text
            if hasattr(match, 'start'):
                position = match.start

                # Determine line number
                char_count = 0
                line_num = 0
                for i, line in enumerate(lines):
                    if char_count + len(line) >= position:
                        line_num = i
                        break
                    char_count += len(line) + 1  # +1 for newline

                # Location-based confidence boost
                if line_num < total_lines * 0.1:
                    # Header (top 10%)
                    match_dict['location'] = 'header'
                    match_dict['confidence'] += 0.15
                elif line_num > total_lines * 0.9:
                    # Footer (bottom 10%)
                    match_dict['location'] = 'footer'
                    match_dict['confidence'] += 0.10
                else:
                    # Body
                    match_dict['confidence'] += 0.05

            # Pattern-type based confidence boost
            if match_dict['category'] == 'level':
                match_dict['confidence'] += 0.20
            elif match_dict['category'] == 'control':
                match_dict['confidence'] += 0.15
            elif match_dict['category'] == 'authority':
                match_dict['confidence'] += 0.10

            # Ensure confidence is in [0, 1]
            match_dict['confidence'] = max(0.0, min(1.0, match_dict['confidence']))

            enhanced_matches.append(match_dict)

        return enhanced_matches

    def _context_analysis(
        self,
        matches: List[Dict],
        text: str
    ) -> List[Dict]:
        """
        Analyze context around matches to refine confidence

        Context heuristics:
        - Multiple classification markings: +0.10 confidence
        - Authority block nearby: +0.15 confidence
        - Consistent classification level: +0.10 confidence
        """
        # Count classification levels
        level_counts = {}
        for match in matches:
            if match['category'] == 'level':
                level = match.get('pattern_type', 'unknown')
                level_counts[level] = level_counts.get(level, 0) + 1

        # Find dominant level
        dominant_level = None
        if level_counts:
            dominant_level = max(level_counts, key=level_counts.get)

        # Enhance matches based on context
        enhanced = []
        for match in matches:
            # Boost if multiple markings present
            if len(matches) > 3:
                match['confidence'] = min(1.0, match['confidence'] + 0.10)

            # Boost if consistent with dominant level
            if dominant_level and match.get('pattern_type') == dominant_level:
                match['confidence'] = min(1.0, match['confidence'] + 0.10)

            # Boost control markings if classification level present
            if match['category'] == 'control' and level_counts:
                match['confidence'] = min(1.0, match['confidence'] + 0.15)

            enhanced.append(match)

        return enhanced

    def _determine_classification(
        self,
        matches: List[Dict]
    ) -> Tuple[Optional[str], float]:
        """
        Determine classification level and overall confidence

        Returns:
            (classification_level, confidence)
        """
        if not matches:
            return None, 0.0

        # Find highest classification level
        level_matches = [m for m in matches if m['category'] == 'level']

        if not level_matches:
            # No classification level found, return highest confidence match
            max_match = max(matches, key=lambda m: m['confidence'])
            return None, max_match['confidence']

        # Classification level hierarchy
        level_hierarchy = {
            'TOP_SECRET': 3,
            'SECRET': 2,
            'CONFIDENTIAL': 1,
            'CUI': 0
        }

        # Find highest level
        highest_level = None
        highest_rank = -1
        total_confidence = 0.0

        for match in level_matches:
            pattern_type = match.get('pattern_type', '').upper()

            # Normalize pattern type
            if 'TOP' in pattern_type and 'SECRET' in pattern_type:
                pattern_type = 'TOP_SECRET'
            elif 'SECRET' in pattern_type:
                pattern_type = 'SECRET'
            elif 'CONFIDENTIAL' in pattern_type:
                pattern_type = 'CONFIDENTIAL'
            elif 'CUI' in pattern_type:
                pattern_type = 'CUI'

            rank = level_hierarchy.get(pattern_type, -1)
            if rank > highest_rank:
                highest_rank = rank
                highest_level = pattern_type

            total_confidence += match['confidence']

        # Average confidence across level matches
        avg_confidence = total_confidence / len(level_matches) if level_matches else 0.0

        return highest_level, avg_confidence

    def _matches_overlap(self, match1: Dict, match2: Any) -> bool:
        """Check if two matches overlap in text"""
        text1 = match1.get('text', '').upper()
        text2 = match2.text.upper() if hasattr(match2, 'text') else str(match2).upper()

        # Simple overlap check
        return text1 in text2 or text2 in text1

    def get_statistics(self) -> Dict:
        """
        Get performance statistics

        Returns:
            Dictionary with performance metrics
        """
        stats = self.stats.copy()

        # Calculate derived metrics
        if stats['total_documents'] > 0:
            stats['fast_path_rate'] = stats['fast_path_only'] / stats['total_documents']
            stats['slow_path_rate'] = stats['slow_path_invoked'] / stats['total_documents']
            stats['early_accept_rate'] = stats['early_accept'] / stats['total_documents']
            stats['early_reject_rate'] = stats['early_reject'] / stats['total_documents']

        if stats['fast_path_only'] > 0:
            stats['avg_fast_time'] = stats['total_fast_time'] / stats['total_documents']

        if stats['slow_path_invoked'] > 0:
            stats['avg_slow_time'] = stats['total_slow_time'] / stats['slow_path_invoked']

        # Component availability
        stats['has_fast_matcher'] = HAS_FAST_MATCHER
        stats['has_visual_detector'] = HAS_VISUAL_DETECTOR
        stats['has_ocr'] = HAS_OCR
        stats['has_llm'] = self.enable_llm and HAS_LLM

        return stats

    def reset_statistics(self):
        """Reset performance statistics"""
        self.stats = {
            'total_documents': 0,
            'fast_path_only': 0,
            'slow_path_invoked': 0,
            'early_accept': 0,
            'early_reject': 0,
            'total_fast_time': 0.0,
            'total_slow_time': 0.0
        }

    def save_statistics(self, output_path: str):
        """Save statistics to JSON file"""
        stats = self.get_statistics()

        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"✓ Statistics saved to {output_path}")


def main():
    """Example usage"""
    print("=" * 70)
    print("Hybrid Two-Stage Classification System")
    print("=" * 70)
    print()

    # Initialize classifier
    config = {
        'accept_threshold': 0.95,
        'reject_threshold': 0.10,
        'min_confidence': 0.3
    }

    classifier = HybridClassifier(
        config=config,
        use_gpu=True,
        enable_llm=False
    )

    print("\nComponent Status:")
    print(f"  Fast Matcher: {HAS_FAST_MATCHER}")
    print(f"  Visual Detector: {HAS_VISUAL_DETECTOR}")
    print(f"  OCR: {HAS_OCR}")
    print()

    # Example classification
    test_text = """
    TOP SECRET//NOFORN

    CLASSIFIED BY: John Doe, DoD
    DERIVED FROM: Multiple Sources
    DECLASSIFY ON: 20401231

    This is a test document with classification markings.

    (S) This paragraph contains secret information.
    (TS) This paragraph contains top secret information.

    TOP SECRET//NOFORN
    """

    print("Testing classification...")
    result = classifier.classify(test_text)

    print(f"\nResult:")
    print(f"  Has classification: {result.has_classification}")
    print(f"  Level: {result.classification_level}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Matches: {len(result.matches)}")
    print(f"  Path: {result.path_taken.value}")
    print(f"  Early exit: {result.early_exit}")
    print(f"  Total time: {result.total_time*1000:.2f}ms")

    # Show statistics
    print("\nPerformance Statistics:")
    stats = classifier.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
