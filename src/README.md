# Classmark Source Code

Core implementation of the Classmark classification marking detection system.

## Directory Structure

```
src/
├── core/                   # Core detection modules
│   ├── classification_scanner.py      # Main scanner orchestration
│   ├── fast_pattern_matcher.py        # Fast pattern matching (Aho-Corasick)
│   ├── visual_pattern_detector.py     # Visual detection (LayoutLMv3)
│   ├── yolo_ocr.py                    # YOLO + Tesseract OCR
│   ├── simple_ocr.py                  # Simple OCR fallback
│   ├── hybrid_classifier.py           # Two-stage classifier
│   ├── calibrated_scorer.py           # Confidence calibration
│   └── evaluation.py                  # Metrics and evaluation
├── gui/                    # GUI applications
│   └── classmark_gui.py              # Desktop GUI application
├── config/                 # Configuration
│   └── offline_config.py             # Offline mode configuration
└── utils/                  # Utilities
    ├── download_models.py            # Model downloader
    └── finetune_layoutlm.py          # Fine-tuning script
```

## Module Overview

### Core Modules

#### classification_scanner.py
Main scanner that orchestrates all detection layers.

**Key Classes:**
- `ClassificationScanner` - Main scanner interface
- `DetectionResult` - Detection result dataclass
- `ReportGenerator` - Generate detailed reports

**Layers:**
1. Fast keyword detection
2. Fast pattern matching (Aho-Corasick)
3. Regex pattern matching
4. Fuzzy matching
5. Context analysis
6. LLM verification (optional)
7. Visual pattern detection (LayoutLMv3)

#### fast_pattern_matcher.py
High-performance pattern matching using Aho-Corasick automaton.

**Performance:**
- 11.8x faster than sequential regex
- O(text_length + matches) complexity
- 27 literal patterns + 15 regex patterns

#### visual_pattern_detector.py
Visual pattern detection using LayoutLMv3 multimodal transformer.

**Features:**
- 125M parameter model
- Analyzes text + layout + visual features
- Detects banners, headers, portion marks
- 15-25% accuracy improvement

#### yolo_ocr.py
SOTA OCR using YOLOv8n for text detection + Tesseract.

**Features:**
- YOLOv8n nano model (~6MB)
- Fast region detection (5-10ms per page)
- DPI validation (200-300 DPI standards)
- Selective processing (<50 words trigger)

#### simple_ocr.py
Simple fallback OCR when YOLO unavailable.

**Usage:**
- Fallback mechanism only
- Direct Tesseract + pdf2image
- Lower accuracy than YOLO approach

#### hybrid_classifier.py
Two-stage architecture with fast/slow path routing.

**Architecture:**
- Stage 1: Fast pattern matching
- Stage 2: Visual detection (if confidence ambiguous)
- Early exit thresholds (>95% accept, <10% reject)

**Performance:**
- 2.2 million docs/min (fast path)
- 5,000-20,000 docs/min (hybrid)

#### calibrated_scorer.py
Confidence score calibration.

**Methods:**
- Platt scaling
- Temperature scaling
- Isotonic regression

**Metrics:**
- ECE (Expected Calibration Error)
- MCE (Maximum Calibration Error)
- Brier score
- Log loss

#### evaluation.py
Comprehensive evaluation framework.

**Features:**
- Ground truth management
- Precision, recall, F1, accuracy
- Per-level metrics
- Confidence range validation
- ROC curves, confusion matrices

### GUI Module

#### classmark_gui.py
Desktop GUI application (tkinter).

**Features:**
- Folder-based batch processing
- Automatic file organization
- CSV/JSON logging
- Settings dialog
- Real-time progress
- Cross-platform (Windows, macOS, Linux)

### Configuration Module

#### offline_config.py
Manages offline operation configuration.

**Features:**
- Auto-detect offline mode
- Model path resolution
- Configuration file management
- Model verification

### Utilities Module

#### download_models.py
Download AI models for offline operation.

**Downloads:**
- LayoutLMv3-base (~500MB)
- YOLOv8n (~6MB)
- Creates offline cache
- Verification

#### finetune_layoutlm.py
Fine-tune LayoutLMv3 on custom classification data.

**Features:**
- BIO token classification (15 labels)
- Training on ground truth
- Custom dataset support
- GPU acceleration

## Import Usage

From project root (with src/ in PYTHONPATH):

```python
# Core modules
from src.core.classification_scanner import ClassificationScanner
from src.core.fast_pattern_matcher import FastPatternMatcher
from src.core.visual_pattern_detector import VisualPatternDetector

# GUI
from src.gui.classmark_gui import ClassmarkGUI

# Config
from src.config.offline_config import OfflineConfigManager

# Utils
from src.utils.download_models import download_layoutlmv3, download_yolo
```

From build scripts (adjust paths or PYTHONPATH):

```python
# Option 1: Adjust sys.path
import sys
sys.path.insert(0, 'src')
from core.classification_scanner import ClassificationScanner

# Option 2: Use absolute imports with package
from classmark.core.classification_scanner import ClassificationScanner
```

## Development

### Running from src/

```bash
# Set PYTHONPATH
export PYTHONPATH=.:src

# Run module
python -m src.gui.classmark_gui

# Run tests
pytest tests/
```

### Code Style

- **PEP 8** compliance
- **Type hints** for public APIs
- **Docstrings** for all classes and functions
- **Comments** for complex logic

### Adding New Modules

1. Create module in appropriate directory
2. Add to `__init__.py` if needed
3. Write comprehensive tests
4. Update documentation
5. Update build scripts if needed

## Dependencies

See `requirements.txt` in project root.

**Key Dependencies:**
- PyTorch >= 2.7.1
- transformers >= 4.57.1
- ultralytics >= 8.3.227
- pyahocorasick >= 2.2.0
- scikit-learn >= 1.7.2

## Performance

**Detection Performance:**
- Fast path: 0.04ms per document
- Hybrid: 3-12ms per document
- Throughput: 2.2M docs/min (fast), 5K-20K docs/min (hybrid)

**Accuracy:**
- Precision: 93.02%
- Recall: 100%
- F1 Score: 96.36%

---

**Version:** 1.0.0
**Last Updated:** 2025-11-10
