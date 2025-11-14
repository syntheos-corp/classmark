#!/usr/bin/env python3
"""Test TOP SECRET regex pattern"""

import re

# Pattern from fast_pattern_matcher.py
pattern = r'(?:^|\n)\s*(?:\/\/|//)?\s*TOP\s*SECRET(?:\s*\/\/[A-Z\s\/]+)?'

# Test text
text1 = "\nTOP SECRET//NOFORN\n"
text2 = "TOP SECRET//NOFORN"
text3 = "\n\nTOP SECRET//NOFORN\n"

regex = re.compile(pattern)

print("Testing TOP SECRET regex pattern...")
print(f"Pattern: {pattern}\n")

for i, text in enumerate([text1, text2, text3], 1):
    print(f"Test {i}: repr={repr(text)}")
    matches = list(regex.finditer(text))
    if matches:
        for match in matches:
            print(f"  MATCH: '{match.group()}' at {match.start()}-{match.end()}")
    else:
        print("  NO MATCH")
    print()
