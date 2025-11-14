# Classmark Build Instructions

This document provides step-by-step instructions for building Classmark installers for Windows and macOS.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Preparing for Build](#preparing-for-build)
3. [Building for Windows](#building-for-windows)
4. [Building for macOS](#building-for-macos)
5. [Testing the Build](#testing-the-build)
6. [Distribution](#distribution)

---

## Prerequisites

### Common Prerequisites

1. **Python 3.10 or later**
   ```bash
   python --version  # Should be 3.10+
   ```

2. **All dependencies installed**
   ```bash
   pip install -r requirements.txt
   ```

3. **Models downloaded (optional but recommended)**
   ```bash
   python download_models.py
   ```
   - Downloads ~600 MB of AI models
   - One-time operation
   - Can be included in installer or downloaded by users

### Windows-Specific Prerequisites

1. **Windows 10/11 (64-bit)**

2. **PyInstaller**
   ```cmd
   pip install pyinstaller
   ```

3. **Inno Setup 6** (for installer creation)
   - Download from: https://jrsoftware.org/isdl.php
   - Install to default location: `C:\Program Files (x86)\Inno Setup 6\`

4. **Optional: UPX** (for smaller executables)
   - Download from: https://upx.github.io/
   - Add to PATH

### macOS-Specific Prerequisites

1. **macOS High Sierra (10.13) or later**

2. **Xcode Command Line Tools**
   ```bash
   xcode-select --install
   ```

3. **py2app**
   ```bash
   pip3 install py2app
   ```

4. **create-dmg** (optional, for better DMG creation)
   ```bash
   brew install create-dmg
   ```

---

## Preparing for Build

### 1. Clean Previous Builds

**Windows:**
```cmd
rmdir /s /q build dist
```

**macOS/Linux:**
```bash
rm -rf build dist Output
```

### 2. Verify Tests Pass

```bash
# Run all unit tests
python -m pytest

# Or run individual test suites
python test_fast_pattern_matcher.py
python test_visual_pattern_detector.py
python test_hybrid_classifier.py
```

All tests should pass before building.

### 3. Update Version Information

Edit the following files to update version numbers:

- `classmark_gui.py` - Line ~30
- `classmark.spec` - VERSION constant
- `setup_macos.py` - VERSION constant
- `classmark_installer.iss` - MyAppVersion define

### 4. Optional: Download Models

To include models in the installer (recommended for offline installation):

```bash
python download_models.py
```

This creates a `models/` directory with:
- LayoutLMv3 model (~500 MB)
- YOLOv8n model (~6 MB)
- Configuration file

**Note:** Including models increases installer size from ~500 MB to ~1.5 GB but enables fully offline installation.

---

## Building for Windows

### Step 1: Build Application

```cmd
build_windows.bat
```

Or manually:

```cmd
pyinstaller classmark.spec
```

**Expected output:**
- `dist/Classmark/` directory containing:
  - `Classmark.exe`
  - All dependencies
  - Supporting files

**Build time:** 5-15 minutes depending on system

### Step 2: Test Application

```cmd
cd dist\Classmark
Classmark.exe
```

Verify:
- Application launches
- UI displays correctly
- Can select folders
- Can process sample documents

### Step 3: Build Installer

```cmd
build_installer.bat
```

Or manually with Inno Setup:

```cmd
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" classmark_installer.iss
```

**Expected output:**
- `Output/ClassmarkSetup.exe`

**Build time:** 2-5 minutes

**Installer size:**
- Without models: ~500 MB
- With models: ~1.5 GB

### Step 4: Test Installer

1. Run `Output\ClassmarkSetup.exe`
2. Follow installation wizard
3. Launch Classmark from Start Menu
4. Verify functionality

---

## Building for macOS

### Step 1: Build Application Bundle

```bash
./build_macos.sh
```

Or manually:

```bash
python3 setup_macos.py py2app
```

**For development/testing (faster):**
```bash
./build_macos.sh alias
```

**Expected output:**
- `dist/Classmark.app` bundle

**Build time:** 5-15 minutes depending on system

### Step 2: Test Application

```bash
open dist/Classmark.app
```

Verify:
- Application launches
- UI displays correctly
- Can select folders
- Can process sample documents

### Step 3: Build DMG Installer

```bash
./build_dmg.sh
```

**Expected output:**
- `Output/Classmark-1.0.0.dmg`
- `Output/Classmark-1.0.0.dmg.sha256` (checksum)

**Build time:** 2-5 minutes

**DMG size:**
- Without models: ~500 MB
- With models: ~1.5 GB

### Step 4: Test DMG

1. Open `Output/Classmark-1.0.0.dmg`
2. Drag Classmark.app to Applications
3. Launch from Applications
4. Verify functionality

---

## Testing the Build

### Functional Testing

1. **Launch Test:**
   - Application starts without errors
   - GUI displays correctly
   - All buttons are functional

2. **Folder Selection Test:**
   - Can browse and select input folder
   - Can browse and select output folder
   - Paths display correctly

3. **Processing Test:**
   - Process sample documents
   - Progress bar updates
   - Log displays real-time status
   - Processing completes successfully

4. **File Operations Test:**
   - Files are moved correctly (if enabled)
   - Directory structure is preserved
   - Log file is created

5. **Settings Test:**
   - Settings dialog opens
   - Can change sensitivity
   - Can toggle options
   - Settings are saved

### Performance Testing

Create a test folder with 100 sample documents:

```bash
# Time the processing
# Should complete in 2-10 seconds for 100 documents
```

Expected performance:
- Fast mode: ~50-200 docs/second
- Visual detection: ~5-20 docs/second

### Offline Testing

1. Disconnect from internet
2. Launch application
3. Process documents
4. Verify all features work

If models are included in installer, everything should work offline.

---

## Distribution

### Windows Distribution

**Files to distribute:**
- `Output/ClassmarkSetup.exe` - Main installer

**Optional files:**
- `CLASSMARK_GUI_README.md` - User documentation
- `README.md` - Project overview

**Distribution channels:**
- Direct download
- Internal network share
- USB drive

**Checksums:**
```cmd
certutil -hashfile Output\ClassmarkSetup.exe SHA256 > Output\ClassmarkSetup.exe.sha256
```

### macOS Distribution

**Files to distribute:**
- `Output/Classmark-1.0.0.dmg` - Main installer
- `Output/Classmark-1.0.0.dmg.sha256` - Checksum

**Optional files:**
- `CLASSMARK_GUI_README.md` - User documentation
- `README.md` - Project overview

**Distribution channels:**
- Direct download
- Internal network share
- USB drive

### Code Signing (Optional but Recommended)

**Windows:**
```cmd
signtool sign /f certificate.pfx /p password /t http://timestamp.digicert.com Output\ClassmarkSetup.exe
```

**macOS:**
```bash
codesign --deep --force --verify --verbose --sign "Developer ID Application" dist/Classmark.app
```

### Notarization (macOS, for public distribution)

```bash
# Create a zip
ditto -c -k --keepParent dist/Classmark.app Classmark.zip

# Submit for notarization
xcrun notarytool submit Classmark.zip --apple-id your@email.com --team-id TEAMID --wait

# Staple the ticket
xcrun stapler staple dist/Classmark.app
```

---

## Troubleshooting Build Issues

### Windows Issues

**Error: "PyInstaller not found"**
```cmd
pip install pyinstaller
```

**Error: "UPX is not available"**
- Edit `classmark.spec`, set `upx=False`

**Error: "Module not found"**
- Check `hiddenimports` in `classmark.spec`
- Add missing module to list

**Error: "Inno Setup not found"**
- Install from https://jrsoftware.org/isdl.php
- Or manually compile with ISCC.exe

### macOS Issues

**Error: "py2app not found"**
```bash
pip3 install py2app
```

**Error: "No module named 'setuptools'"**
```bash
pip3 install setuptools
```

**Error: "Code signature invalid"**
```bash
# Remove quarantine attribute
xattr -cr dist/Classmark.app
```

**Error: "Application damaged"**
- Application needs to be signed
- Or: Right-click → Open (first time)

### Common Issues

**Large executable size:**
- This is normal for bundled Python applications
- PyTorch alone is ~500 MB
- Use UPX compression to reduce size

**Slow startup:**
- First launch extracts files (one-time)
- Subsequent launches are faster
- Consider one-directory mode instead of one-file

**Missing dependencies:**
- Check requirements.txt
- Verify all imports in spec file
- Test in clean virtual environment

---

## Build Automation (CI/CD)

### GitHub Actions Example

```yaml
name: Build Classmark

on:
  push:
    tags:
      - 'v*'

jobs:
  build-windows:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pyinstaller
      - name: Build
        run: pyinstaller classmark.spec
      - name: Upload artifact
        uses: actions/upload-artifact@v2
        with:
          name: classmark-windows
          path: dist/Classmark/

  build-macos:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip3 install -r requirements.txt
          pip3 install py2app
      - name: Build
        run: python3 setup_macos.py py2app
      - name: Upload artifact
        uses: actions/upload-artifact@v2
        with:
          name: classmark-macos
          path: dist/Classmark.app
```

---

## Release Checklist

- [ ] All tests pass
- [ ] Version numbers updated
- [ ] Models downloaded (if including)
- [ ] Windows build completes successfully
- [ ] Windows installer created
- [ ] Windows installer tested
- [ ] macOS build completes successfully
- [ ] macOS DMG created
- [ ] macOS DMG tested
- [ ] User documentation updated
- [ ] Release notes written
- [ ] Checksums generated
- [ ] (Optional) Binaries signed
- [ ] (Optional) macOS app notarized
- [ ] Distribution files uploaded

---

## Support

For build issues or questions:
- Check troubleshooting section above
- Review PyInstaller documentation: https://pyinstaller.org/
- Review py2app documentation: https://py2app.readthedocs.io/
- Open issue on GitHub

**Build Date:** 2025-11-10
**Document Version:** 1.0.0
