# Classmark - SOTA Classification Marking Detection

State-of-the-art classification marking detection system using advanced AI and pattern matching.

**Version:** 1.0.0
**Status:** Production Ready
**Platforms:** Windows, macOS

---

## Overview

Classmark automatically detects US Government security classification markings in documents using:
- **LayoutLMv3** - 125M parameter multimodal transformer for visual detection
- **YOLOv8n** - Fast text region detection for OCR
- **Aho-Corasick** - High-performance pattern matching (11.8x faster than regex)
- **Hybrid Classifier** - Two-stage architecture (2.2M docs/min throughput)

**Performance:**
- **Precision:** 93.02%
- **Recall:** 100%
- **Speed:** 2.2 million documents/minute (fast path)
- **Fully Offline:** No internet required after installation

---

## Features

### Detection Capabilities
- ✅ TOP SECRET, SECRET, CONFIDENTIAL, CUI
- ✅ Control markings (NOFORN, ORCON, NOCONTRACT, etc.)
- ✅ Portion markings (TS), (S), (C), (U)
- ✅ Classification authority blocks
- ✅ Declassification instructions
- ✅ Visual pattern detection (banners, headers)
- ✅ Scanned document OCR

### User Features
- 🖥️ **Desktop GUI** - User-friendly interface
- 📁 **Batch Processing** - Scan entire folders
- 🔄 **Auto Organization** - Move classified files automatically
- 📊 **Detailed Logs** - CSV/JSON with confidence scores
- ⚙️ **Configurable** - Sensitivity, thresholds, GPU toggle
- 🔒 **100% Offline** - No internet required

---

## Quick Start

### Installation

**Windows:**
1. Download `ClassmarkSetup.exe`
2. Run installer
3. Launch Classmark from Start Menu

**macOS:**
1. Download `Classmark-1.0.0.dmg`
2. Drag to Applications folder
3. Launch from Applications

**First Run:**
If models weren't included, download them once:
```bash
# Windows
cd "C:\Program Files\Classmark"
python download_models.py

# macOS
cd /Applications/Classmark.app/Contents/Resources
python3 download_models.py
```

### Usage

1. **Select Input Folder** - Folder containing documents to scan
2. **Select Output Folder** - Where to move classified documents
3. **Configure Settings** (optional) - Sensitivity, thresholds, etc.
4. **Start Processing** - Scan begins automatically
5. **Review Results** - Check output folder and log file

**Supported Formats:** PDF, DOCX, TXT

---

## Project Structure

```
classmark/
├── src/                    # Source code
│   ├── core/              # Core detection modules
│   ├── gui/               # Desktop GUI application
│   ├── config/            # Configuration management
│   └── utils/             # Utilities (model download, fine-tuning)
├── tests/                  # Test suite (67 tests)
│   ├── core/              # Unit tests
│   ├── integration/       # Integration tests
│   └── edge_cases/        # Edge case tests
├── build/                  # Build configurations
│   ├── windows/           # Windows build (PyInstaller + Inno Setup)
│   └── macos/             # macOS build (py2app + DMG)
├── docs/                   # Documentation
├── tools/                  # Dataset annotation tools
├── scripts/                # Test and evaluation scripts
├── dev_tools/              # Developer debugging tools
├── models/                 # AI models (after download)
├── documents/              # Test documents
├── ground_truth.json       # Annotated test dataset
├── requirements.txt        # Python dependencies
├── install.sh             # Linux/Mac installation
└── README.md              # This file
```

---

## For Developers

### Building from Source

**Prerequisites:**
- Python 3.10+
- PyInstaller (Windows) or py2app (macOS)
- Inno Setup (Windows) or create-dmg (macOS)

**Build:**
```bash
# Windows
cd build/windows
build_windows.bat
build_installer.bat

# macOS
cd build/macos
./build_macos.sh
./build_dmg.sh
```

See `docs/BUILD_INSTRUCTIONS.md` for detailed build documentation.

### Running Tests

```bash
# All tests
python scripts/run_all_tests.py

# Specific suite
pytest tests/core/
pytest tests/integration/

# With coverage
pytest tests/ --cov=src --cov-report=html
```

**Test Stats:** 67/67 passing (100%)

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd classmark

# Install dependencies
pip install -r requirements.txt

# Download models
python src/utils/download_models.py

# Run tests
python scripts/run_all_tests.py

# Run GUI from source
python src/gui/classmark_gui.py
```

---

## Architecture

### Detection Layers

1. **Fast Keyword Detection** - Common classification keywords
2. **Fast Pattern Matching** - Aho-Corasick automaton (27 patterns)
3. **Regex Patterns** - Complex patterns requiring lookahead (15 patterns)
4. **Fuzzy Matching** - Handle typos and variations (85% similarity)
5. **Context Analysis** - Surrounding text analysis
6. **Visual Detection** - LayoutLMv3 multimodal analysis (optional)

### Two-Stage Architecture

**Fast Path (Stage 1):**
- Pattern matching only
- ~0.04ms per document
- 2.2M documents/minute

**Slow Path (Stage 2):**
- Visual detection with LayoutLMv3
- ~3-12ms per document
- 5K-20K documents/minute
- Triggered for ambiguous cases

**Early Exit Thresholds:**
- Confidence >95% → Accept (skip slow path)
- Confidence <10% → Reject (skip slow path)
- 10-95% → Route to slow path

---

## Performance

### Accuracy Metrics
```
Precision: 93.02%
Recall:    100.00%
F1 Score:  96.36%
Accuracy:  93.02%
```

Tested on 61 annotated documents (48 synthetic + 13 real declassified docs).

### Throughput
- Fast mode: 2.2 million docs/min
- Hybrid mode: 5,000-20,000 docs/min
- GPU acceleration: 10-50x faster visual detection

### System Requirements

**Minimum:**
- 8GB RAM
- 5GB disk space (15GB with models)
- Dual-core CPU

**Recommended:**
- 16GB RAM
- 20GB disk space
- Quad-core CPU
- NVIDIA GPU (6GB+ VRAM)

---

## Documentation

- **User Guide:** `docs/CLASSMARK_GUI_README.md`
- **Build Instructions:** `docs/BUILD_INSTRUCTIONS.md`
- **Deployment Guide:** `docs/DEPLOYMENT_SUMMARY.md`
- **Project Summary:** `docs/PROJECT_SUMMARY.md`
- **Phase Documentation:** `docs/PHASE3_SUMMARY.md`, `docs/PHASE4_SUMMARY.md`
- **Refactoring Analysis:** `docs/REFACTORING_ANALYSIS.md`

---

## Technology Stack

### AI/ML
- **PyTorch 2.7.1** - Deep learning framework
- **Transformers 4.57.1** - HuggingFace transformers (LayoutLMv3)
- **Ultralytics 8.3.227** - YOLO object detection
- **scikit-learn 1.7.2** - ML utilities and calibration

### Pattern Matching
- **pyahocorasick 2.2.0** - Aho-Corasick automaton
- **rapidfuzz 3.0+** - Fuzzy string matching

### OCR & Document Processing
- **pytesseract** - Tesseract OCR wrapper
- **pdf2image** - PDF to image conversion
- **PyPDF2 3.0+** - PDF text extraction
- **python-docx 1.0+** - DOCX processing
- **pdfplumber 0.10+** - Advanced PDF extraction

### GUI
- **tkinter** - Cross-platform GUI framework (built into Python)

---

## License

See LICENSE file for details.

---

## Support

- **Documentation:** See `docs/` directory
- **Issues:** Check `docs/CLASSMARK_GUI_README.md` troubleshooting section
- **Performance:** Run `python scripts/run_baseline_evaluation.py`

---

## Credits

**Development Team:** Classmark Development Team
**Date:** 2025-11-10
**Version:** 1.0.0

Powered by:
- Microsoft LayoutLMv3 (Document AI Research)
- Ultralytics YOLOv8 (Object Detection)
- HuggingFace Transformers
- Tesseract OCR (Google)

---

## Changelog

### Version 1.0.0 (2025-11-10)
- ✨ Initial production release
- ✅ Full Windows and macOS support
- ✅ Offline operation capability
- ✅ 67/67 tests passing
- ✅ Professional installers
- ✅ Comprehensive documentation
- ✅ Refactored codebase structure

---

**Ready for Production Deployment**
