#!/usr/bin/env python3
"""Debug script with exact test text"""

import re

# Exact text from test
text = """
        TOP SECRET//NOFORN

        CLASSIFIED BY: John Doe
        DERIVED FROM: ABC-123
        DECLASSIFY ON: 20400101

        This is a classified document.

        TOP SECRET//NOFORN
        """

# TOP SECRET pattern
pattern = re.compile(r'(?:^|\n)\s*(?:\/\/|//)?\s*TOP\s*SECRET(?:\s*\/\/[A-Z\s\/]+)?')

print("Text:")
print(repr(text[:100]))
print("\nSearching for TOP SECRET...")

matches = list(pattern.finditer(text))
print(f"Found {len(matches)} matches:")
for match in matches:
    print(f"  '{match.group()}' at {match.start()}-{match.end()}")

# Now test with FastPatternMatcher
from fast_pattern_matcher import FastPatternMatcher

matcher = FastPatternMatcher()
matcher.initialize()

print("\nFastPatternMatcher results:")
matches = matcher.match_all(text)
print(f"Found {len(matches)} matches:")
for match in matches:
    if 'TOP' in match.pattern or 'SECRET' in match.pattern:
        print(f"  pattern='{match.pattern}' text='{match.matched_text}' pos={match.start}-{match.end}")
