#!/usr/bin/env python3
"""Debug script to understand classification level detection"""

from hybrid_classifier import HybridClassifier

# Initialize classifier
classifier = HybridClassifier(use_gpu=False)

# Test text from failing test
text = """
TOP SECRET//NOFORN

CLASSIFIED BY: John Doe
DERIVED FROM: ABC-123
DECLASSIFY ON: 20400101

This is a classified document.

TOP SECRET//NOFORN
"""

print("Testing classification...")
result = classifier.classify(text)

print(f"\nResult:")
print(f"  has_classification: {result.has_classification}")
print(f"  classification_level: {result.classification_level}")
print(f"  confidence: {result.confidence}")
print(f"  num matches: {len(result.matches)}")

if result.matches:
    print(f"\nMatches:")
    for i, match in enumerate(result.matches[:5], 1):
        print(f"  {i}. text='{match['text'][:30]}' category='{match['category']}' pattern_type='{match['pattern_type']}'")

if result.fast_path_metrics:
    print(f"\nFast path metrics:")
    print(f"  processing_time: {result.fast_path_metrics.processing_time*1000:.2f}ms")
    print(f"  confidence: {result.fast_path_metrics.confidence}")
    print(f"  num_matches: {result.fast_path_metrics.num_matches}")
