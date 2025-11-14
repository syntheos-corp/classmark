#!/usr/bin/env python3
"""Debug script to check what patterns are loaded"""

from fast_pattern_matcher import FastPatternMatcher

# Initialize matcher
matcher = FastPatternMatcher()
matcher.initialize()

print("Initialized!")
print(f"Number of regex patterns: {len(matcher.regex_patterns)}")
print(f"Number of literal patterns: {len(matcher.literal_patterns)}")

print("\nFirst few regex patterns:")
for i, (pattern, category, name, pattern_type) in enumerate(matcher.regex_patterns[:5], 1):
    print(f"{i}. name='{name}' category='{category}'")

print("\nLooking for TOP_SECRET pattern...")
for pattern, category, name, pattern_type in matcher.regex_patterns:
    if 'TOP' in name:
        print(f"  Found: name='{name}' category='{category}'")
        print(f"    Pattern: {pattern.pattern[:100]}")
