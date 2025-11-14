#!/usr/bin/env python3
"""Debug script to see what FastPatternMatcher finds"""

from fast_pattern_matcher import FastPatternMatcher

# Initialize matcher
matcher = FastPatternMatcher()
matcher.initialize()

# Test text
text = """
TOP SECRET//NOFORN

CLASSIFIED BY: John Doe
DERIVED FROM: ABC-123
DECLASSIFY ON: 20400101

This is a classified document.

TOP SECRET//NOFORN
"""

print("Matching patterns...")
matches = matcher.match_all(text)

print(f"\nFound {len(matches)} matches:\n")
for i, match in enumerate(matches, 1):
    print(f"{i}. pattern='{match.pattern}' category='{match.category}' text='{match.matched_text}' pos={match.start}-{match.end}")
