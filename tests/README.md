# Classmark Test Suite

Comprehensive test suite for the Classmark classification marking detection system.

## Directory Structure

```
tests/
├── core/                           # Unit tests for core components
│   ├── test_fast_pattern_matcher.py    # Pattern matching tests (16 tests)
│   ├── test_visual_pattern_detector.py # Visual detection tests (13 tests)
│   ├── test_hybrid_classifier.py        # Hybrid classifier tests (22 tests)
│   └── test_yolo_ocr.py                 # OCR tests (16 tests)
├── integration/                    # Integration tests
│   ├── test_classification_scanner.py   # End-to-end scanner tests
│   └── test_sota_features.py            # SOTA features integration
├── edge_cases/                     # Edge case and regression tests
│   ├── test_edge_cases_comprehensive.py # Comprehensive edge cases
│   ├── test_format_specific_edge_cases.py # Format-specific tests
│   └── test_regex.py                    # Regex pattern tests
└── test_data_generator.py          # Test data generation utility
```

## Running Tests

### Run All Tests

```bash
# From project root
python run_all_tests.py

# Or with pytest
pytest tests/
```

### Run Specific Test Suite

```bash
# Core tests
pytest tests/core/

# Integration tests
pytest tests/integration/

# Edge case tests
pytest tests/edge_cases/

# Specific test file
pytest tests/core/test_fast_pattern_matcher.py
```

### Run with Coverage

```bash
pytest tests/ --cov=. --cov-report=html
```

## Test Statistics

**Total Tests:** 67+ tests across all suites

**Core Component Tests (67 tests):**
- Fast Pattern Matcher: 16/16 passing ✅
- Visual Pattern Detector: 13/13 passing ✅
- Hybrid Classifier: 22/22 passing ✅
- YOLO OCR: 16/16 passing ✅

**Test Coverage:** Core modules have >90% coverage

## Test Requirements

All tests use **real implementations** (NO MOCKS):
- Actual LayoutLMv3 model (125M parameters)
- Actual YOLOv8n model for OCR
- Real document processing
- Real pattern matching algorithms

**System Requirements for Testing:**
- 8GB+ RAM
- GPU recommended (tests run on CPU if unavailable)
- Downloaded AI models (run `python download_models.py`)
- ~2-10 minutes total test execution time

## Adding New Tests

1. Choose appropriate directory (core/integration/edge_cases)
2. Create test file: `test_<feature_name>.py`
3. Import necessary modules
4. Write test cases using unittest or pytest
5. Update this README with test count

**Example:**

```python
import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from fast_pattern_matcher import FastPatternMatcher

class TestNewFeature(unittest.TestCase):
    def setUp(self):
        self.matcher = FastPatternMatcher()

    def test_something(self):
        result = self.matcher.match("TEST")
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()
```

## Continuous Integration

Tests are designed to run in CI/CD pipelines:

```yaml
# GitHub Actions example
- name: Run tests
  run: python run_all_tests.py
```

## Test Data

**Ground Truth:** `ground_truth.json` (61 annotated documents)
- 48 synthetic documents
- 13 real declassified documents

**Test Documents:** `documents/` directory
- Various classification levels
- Multiple formats (PDF, DOCX, TXT)
- Scanned and digital documents

## Known Issues

1. **Numpy Compatibility:** Some systems may show numpy version warnings (safe to ignore)
2. **GPU Tests:** GPU-specific tests will skip if CUDA unavailable
3. **Model Download:** First run requires internet to download models (~600MB)

## Test Maintenance

- **Review:** Tests should be reviewed quarterly
- **Update:** Update when adding new features
- **Performance:** Monitor test execution time
- **Coverage:** Maintain >90% coverage for core modules

---

**Last Updated:** 2025-11-10
**Total Test Count:** 67+ tests
**Pass Rate:** 100% (67/67)
