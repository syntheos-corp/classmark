# Classmark Desktop Application Packaging Guide

## Overview

This guide explains how to build standalone desktop installers for Classmark that end users can install with a double-click. No Python environment, pip, or command line required.

**CRITICAL**: This is a security classification application. The installer MUST include all SOTA components for full accuracy (93% precision, 100% recall).

---

## Prerequisites

### 1. Install Development Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Install build tools
pip install pyinstaller  # Windows
pip install py2app       # macOS
```

### 2. Download All Models (REQUIRED)

**Before building the installer**, you MUST download all AI models:

```bash
python src/utils/download_models.py
```

This downloads (~500MB total):
- **LayoutLMv3-base** (~470MB) - Visual pattern detection
- **YOLOv8n** (~6MB) - Text region detection
- **Tesseract** (system package) - OCR engine

Models are saved to `models/` directory and will be bundled into the installer.

### 3. Verify SOTA Mode

Test that all components work before building:

```bash
python -c "
from src.core.classification_scanner import ClassificationScanner
config = {'fuzzy_matching': True, 'use_visual_detection': True, 'use_gpu': True, 'use_llm': False}
scanner = ClassificationScanner(config)
print(f'Fast matcher: {scanner.fast_matcher is not None}')
print(f'Visual detector: {scanner.visual_detector is not None}')
print(f'Fuzzy matcher: {scanner.fuzzy_matcher is not None}')
"
```

Expected output:
```
✓ Fast pattern matcher initialized (Aho-Corasick: True)
✓ Visual pattern detector available (LayoutLMv3: True)
Fast matcher: True
Visual detector: True
Fuzzy matcher: True
```

---

## Building Windows Installer

### Step 1: Navigate to Build Directory

```bash
cd build/windows
```

### Step 2: Build with PyInstaller

```bash
pyinstaller classmark.spec
```

This creates:
- `dist/Classmark/` - Complete application directory
- `dist/Classmark/Classmark.exe` - Main executable

### Step 3: Create Installer (Optional)

Use **Inno Setup** to create a Windows installer:

1. Download [Inno Setup](https://jrsoftware.org/isinfo.php)
2. Edit `classmark_installer.iss`
3. Compile the installer
4. Output: `Output/Classmark_Setup_1.0.0.exe`

### Step 4: Test the Installer

1. Install on a clean Windows machine (no Python, no dependencies)
2. Run `Classmark.exe`
3. Verify all SOTA components load:
   - No "Fast pattern matcher not available" warning
   - No "Visual pattern detector not available" warning
4. Process test documents and verify detection accuracy

---

## Building macOS Application

### Step 1: Navigate to Build Directory

```bash
cd build/macos
```

### Step 2: Build with py2app

```bash
# Build standalone app
python setup_macos.py py2app

# Or for development/testing (alias mode)
python setup_macos.py py2app -A
```

This creates:
- `dist/Classmark.app` - macOS application bundle

### Step 3: Create DMG Installer (Optional)

```bash
# Create a DMG for distribution
hdiutil create -volname "Classmark" -srcfolder dist/Classmark.app -ov -format UDZO dist/Classmark.dmg
```

### Step 4: Test the Application

1. Test on a clean macOS machine (no Python, no dependencies)
2. Open `Classmark.app`
3. Verify all SOTA components load (same checks as Windows)
4. Process test documents and verify detection accuracy

---

## What Gets Bundled

### Python Runtime
- Embedded Python interpreter
- No system Python required

### All Dependencies
- **Pattern Matching**: pyahocorasick, rapidfuzz
- **ML/AI**: torch, transformers, ultralytics
- **OCR**: pytesseract, Pillow, pdf2image
- **Document Processing**: PyPDF2, python-docx, pdfplumber
- **Scientific**: numpy, scipy, scikit-learn

### Pre-Downloaded Models
- **LayoutLMv3-base** (470MB) - For visual pattern detection
- **YOLOv8n** (6MB) - For text region detection
- Models are included in the installer, no download needed at runtime

### Application Code
- All modules from `src/core/`, `src/gui/`, `src/config/`
- Test data and documentation

---

## Installer Size

- **Windows**: ~800MB (with all models and dependencies)
- **macOS**: ~800MB (universal binary for Intel + Apple Silicon)

Size breakdown:
- Models: 500MB
- PyTorch: 200MB
- Other dependencies: 100MB

---

## End User Experience

### Installation
1. **Windows**: Double-click `Classmark_Setup.exe` → Next → Install
2. **macOS**: Open `Classmark.dmg` → Drag to Applications

### First Launch
1. Click application icon
2. GUI opens immediately (no console, no terminal)
3. All SOTA components loaded and ready
4. No internet connection required

### Usage
1. Click "Browse" to select document folder
2. Click "Scan" to process documents
3. View results in real-time
4. Check generated logs and moved files

---

## Troubleshooting

### "Fast pattern matcher not available"
**Cause**: pyahocorasick not installed before build
**Fix**: Run `pip install pyahocorasick`, rebuild

### "Visual pattern detector not available"
**Cause**: Models not downloaded or not bundled
**Fix**: Run `python src/utils/download_models.py`, rebuild

### Large installer size
**Expected**: 800MB is normal for ML applications with bundled models
**Alternative**: Offer a "lite" version without visual detection (~300MB)

### Slow startup on first run
**Cause**: Model loading and GPU initialization
**Expected**: 5-10 seconds first launch, <2 seconds after

### GPU not detected
**Windows**: Check CUDA drivers installed
**macOS**: Metal is used automatically on Apple Silicon

---

## Distribution Checklist

Before releasing the installer:

- [ ] Models downloaded (`models/layoutlmv3-base/`, `models/yolo/yolov8n.pt`)
- [ ] All dependencies installed (`pip list` shows all requirements)
- [ ] SOTA mode verified (fast matcher + visual detector working)
- [ ] Installer built successfully
- [ ] Tested on clean machine (no Python, no dev tools)
- [ ] All features work: pattern matching, visual detection, fuzzy matching
- [ ] Baseline accuracy achieved: 93% precision, 100% recall on test data
- [ ] License file included
- [ ] Documentation included
- [ ] Version number updated

---

## Security Notes

**This application handles classified documents. The installer must:**

1. ✅ Include ALL SOTA components (degraded mode unacceptable)
2. ✅ Work completely offline (no internet downloads after install)
3. ✅ Provide consistent, validated accuracy
4. ✅ Include audit logging
5. ✅ Support secure deletion of processed files

**Never distribute an installer without:**
- Full SOTA mode (fast matcher + visual detector)
- Pre-downloaded models
- Validation on clean test machine
- Accuracy verification on baseline test data

---

## Support

For build issues:
- Check `build/windows/` or `build/macos/` logs
- Verify all prerequisites installed
- Test in development mode first (`python src/gui/classmark_gui.py`)

For runtime issues:
- Check application logs
- Verify models are bundled (`dist/Classmark/models/`)
- Test with sample documents first
