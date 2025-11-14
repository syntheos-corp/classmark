# Developer Tools

This directory contains debugging and development utilities for Classmark.

## Debug Scripts

### debug_fast_matcher.py
Test and debug fast pattern matching functionality.

**Usage:**
```bash
python debug_fast_matcher.py
```

### debug_full_text.py
Debug full text extraction from documents.

**Usage:**
```bash
python debug_full_text.py
```

### debug_hybrid.py
Debug hybrid classifier (fast + slow path routing).

**Usage:**
```bash
python debug_hybrid.py
```

### debug_patterns.py
Debug pattern detection and matching.

**Usage:**
```bash
python debug_patterns.py
```

## Running Debug Scripts

From the project root:
```bash
cd dev_tools
python debug_fast_matcher.py
```

Or with Python path:
```bash
PYTHONPATH=. python dev_tools/debug_fast_matcher.py
```

## Notes

- These scripts are for development/debugging only
- Not included in production builds
- May require additional dependencies
- Output is verbose for debugging purposes
