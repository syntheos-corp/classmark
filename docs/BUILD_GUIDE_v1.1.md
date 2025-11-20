# Classmark v1.1 Build Guide

**Version:** 1.1.0
**Date:** 2025-11-19
**Changes:** Added ttkbootstrap for modern UI themes and platform-native improvements

---

## Overview

This guide covers building Classmark v1.1 with the new UI/UX improvements, including ttkbootstrap integration for modern themes, dark mode support, keyboard shortcuts, and system notifications.

---

## Prerequisites

### General Requirements

**Python Version:**
- Python 3.10 or later (tested with 3.10, 3.11, 3.12, 3.13)

**Dependencies:**
```bash
pip install -r requirements.txt
```

**Key New Dependency:**
- `ttkbootstrap>=1.10.1` - Modern themes for tkinter

### Platform-Specific Build Tools

**Windows:**
- PyInstaller 5.0+
- Inno Setup (for installer creation)
- UPX (optional, for compression)

**macOS:**
- py2app 0.28+
- create-dmg (for DMG creation)
- Xcode Command Line Tools

---

## What's New in v1.1 Builds

### Bundled Components

**New in v1.1:**
- `ttkbootstrap` package and all themes
- `src/gui/platform_utils.py` module
- Enhanced platform detection
- Theme assets and configurations

**Updated:**
- `src/gui/classmark_gui.py` with modern UI
- Settings dialog with theme selection
- Keyboard shortcuts support
- System notification integration

---

## Building for Windows

### Step 1: Prepare Environment

```bash
# Navigate to Windows build directory
cd build/windows

# Install dependencies (if not already installed)
pip install -r ../../requirements.txt

# Verify ttkbootstrap is installed
python -c "import ttkbootstrap; print(f'ttkbootstrap version: {ttkbootstrap.__version__}')"
```

### Step 2: Download Models (if not bundled)

```bash
cd ../..
python src/utils/download_models.py
```

This downloads:
- LayoutLMv3 model (~500MB)
- YOLOv8n model (~6MB)

### Step 3: Build Executable

```bash
cd build/windows

# Clean build (recommended for new version)
build_windows.bat clean

# Or regular build
build_windows.bat
```

**Build Time:** 10-20 minutes depending on system

**Output:** `dist/Classmark/Classmark.exe` (and supporting files)

### Step 4: Test the Build

```bash
cd dist/Classmark
Classmark.exe
```

**Test Checklist:**
- [ ] Application launches without errors
- [ ] Modern theme is applied (ttkbootstrap)
- [ ] Dark mode detection works (change Windows theme)
- [ ] Keyboard shortcuts work (Ctrl+O, Ctrl+S, Ctrl+R, Escape)
- [ ] Settings dialog shows theme selection
- [ ] Theme can be changed (Auto/Light/Dark)
- [ ] System notifications work after processing
- [ ] Fonts are correct (Segoe UI for text, Consolas for logs)
- [ ] All original functionality works (scanning, logging, etc.)

### Step 5: Create Installer (Optional)

```bash
# Back to build directory
cd ../../

# Create installer with Inno Setup
build_installer.bat
```

**Output:** `ClassmarkSetup.exe`

---

## Building for macOS

### Step 1: Prepare Environment

```bash
# Navigate to macOS build directory
cd build/macos

# Install dependencies (if not already installed)
pip3 install -r ../../requirements.txt

# Verify ttkbootstrap is installed
python3 -c "import ttkbootstrap; print(f'ttkbootstrap version: {ttkbootstrap.__version__}')"
```

### Step 2: Download Models (if not bundled)

```bash
cd ../..
python3 src/utils/download_models.py
```

### Step 3: Build Application Bundle

```bash
cd build/macos

# Make script executable (first time only)
chmod +x build_macos.sh

# Clean build (recommended for new version)
./build_macos.sh clean

# Or regular build
./build_macos.sh
```

**Build Time:** 10-20 minutes depending on system

**Output:** `dist/Classmark.app`

### Step 4: Test the Build

```bash
open dist/Classmark.app
```

**Test Checklist:**
- [ ] Application launches without errors
- [ ] Modern theme is applied (ttkbootstrap)
- [ ] Dark mode detection works (change macOS appearance)
- [ ] Keyboard shortcuts work (⌘O, ⌘S, ⌘R, Escape, ⌘,)
- [ ] Settings dialog shows theme selection
- [ ] Theme can be changed (Auto/Light/Dark)
- [ ] System notifications work (macOS Notification Center)
- [ ] Fonts are correct (San Francisco for text, SF Mono for logs)
- [ ] All original functionality works
- [ ] Works on both Intel and Apple Silicon

### Step 5: Create DMG (Optional)

```bash
# Make script executable (first time only)
chmod +x build_dmg.sh

# Create DMG installer
./build_dmg.sh
```

**Output:** `Classmark-1.1.0.dmg`

---

## Build Configuration Updates

### Windows (PyInstaller)

**File:** `build/windows/classmark.spec`

**Changes:**
```python
# Added ttkbootstrap data files
datas += collect_data_files('ttkbootstrap', include_py_files=True)

# Added ttkbootstrap imports
hiddenimports = [
    # ... existing imports ...
    'ttkbootstrap',
    'ttkbootstrap.constants',
    'ttkbootstrap.themes',
    'ttkbootstrap.style',
    'ttkbootstrap.window',
    'src.gui.platform_utils',
    'winreg',  # For Windows platform detection
]

# Added ttkbootstrap submodules
hiddenimports += collect_submodules('ttkbootstrap')
```

### macOS (py2app)

**File:** `build/macos/setup_macos.py`

**Changes:**
```python
'packages': [
    # ... existing packages ...
    'ttkbootstrap',  # Added
    'src.gui.platform_utils',  # Added
],
'includes': [
    # ... existing includes ...
    'tkinter.simpledialog',  # Added
    'ttkbootstrap.constants',  # Added
    'ttkbootstrap.themes',  # Added
    'ttkbootstrap.style',  # Added
    'ttkbootstrap.window',  # Added
],
```

---

## Troubleshooting

### Issue: ttkbootstrap not found

**Symptom:** Build completes but app crashes with `ModuleNotFoundError: No module named 'ttkbootstrap'`

**Solution:**
1. Ensure ttkbootstrap is installed: `pip install ttkbootstrap>=1.10.1`
2. Clean build: Delete `build/` and `dist/` directories
3. Rebuild with updated spec file

### Issue: Theme files missing

**Symptom:** App launches but themes don't apply, falls back to standard tkinter

**Solution:**
- Verify `collect_data_files('ttkbootstrap', include_py_files=True)` is in spec
- Check that `dist/` folder contains `ttkbootstrap/` directory with themes
- Rebuild with clean build

### Issue: Platform detection fails

**Symptom:** Dark mode detection doesn't work, or platform-specific features missing

**Windows:**
- Ensure `winreg` is in hiddenimports
- Check Windows registry permissions

**macOS:**
- Verify `subprocess` module is available
- Check that `defaults` command works in terminal

### Issue: Notifications don't work

**Windows:**
- PowerShell must be available
- Toast notification permissions may need user approval

**macOS:**
- Application must have notification permissions
- Check System Preferences → Notifications

**Linux:**
- Install `libnotify-bin`: `sudo apt install libnotify-bin`

### Issue: Large Build Size

**Expected Sizes:**
- Windows: 800MB - 1.2GB (with models)
- macOS: 900MB - 1.3GB (with models)

**Size increased by ~50-100MB due to ttkbootstrap**

**To Reduce Size:**
- Don't bundle models (download on first run)
- Enable UPX compression
- Exclude unnecessary packages in spec file

---

## Testing the Builds

### Functional Testing

**Core Features:**
1. Launch application
2. Select input folder with test documents
3. Select output folder
4. Start processing
5. Verify results are correct
6. Check log file is created

**New v1.1 Features:**
1. Check theme matches system (Auto mode)
2. Change theme to Dark, verify UI updates
3. Change theme to Light, verify UI updates
4. Test all keyboard shortcuts
5. Complete a scan, verify notification appears
6. Open settings, verify all options present
7. Change settings, verify they persist

### Cross-Platform Testing

**Windows:**
- Test on Windows 10 and Windows 11
- Test in both Light and Dark system themes
- Verify Segoe UI font is used
- Test with and without admin privileges

**macOS:**
- Test on Intel Mac
- Test on Apple Silicon (M1/M2)
- Test on macOS 10.13 through latest
- Test in Light and Dark modes
- Verify San Francisco font is used

### Performance Testing

**Startup Time:**
- v1.0: ~2-3 seconds
- v1.1: ~2.5-3.5 seconds (slight increase due to theme loading)

**Memory Usage:**
- v1.0: ~200MB idle
- v1.1: ~210MB idle (+5% for ttkbootstrap)

**Processing Speed:**
- Should be identical to v1.0
- No performance regression in core scanning

---

## Deployment Checklist

### Pre-Release

- [ ] All dependencies installed and tested
- [ ] Models downloaded and bundled (or download script works)
- [ ] Build scripts updated with ttkbootstrap
- [ ] Version numbers updated (1.1.0)
- [ ] Documentation updated

### Build

- [ ] Clean build performed
- [ ] All hiddenimports and data files included
- [ ] Build completes without errors or warnings
- [ ] Output size is reasonable

### Testing

- [ ] Functional testing complete
- [ ] All new v1.1 features work
- [ ] Cross-platform testing done
- [ ] Performance acceptable
- [ ] No regressions from v1.0

### Distribution

- [ ] Installers created (EXE for Windows, DMG for macOS)
- [ ] Installers tested on fresh systems
- [ ] README and documentation included
- [ ] License file included
- [ ] Version info correct in About dialog

---

## Build Automation (CI/CD)

### GitHub Actions Example

```yaml
name: Build Classmark v1.1

on: [push, pull_request]

jobs:
  build-windows:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pyinstaller
      - name: Build Windows executable
        run: |
          cd build/windows
          pyinstaller classmark.spec
      - name: Upload artifact
        uses: actions/upload-artifact@v2
        with:
          name: classmark-windows
          path: build/windows/dist/Classmark/

  build-macos:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip3 install -r requirements.txt
          pip3 install py2app
      - name: Build macOS app
        run: |
          cd build/macos
          python3 setup_macos.py py2app
      - name: Upload artifact
        uses: actions/upload-artifact@v2
        with:
          name: classmark-macos
          path: build/macos/dist/Classmark.app
```

---

## Version Comparison

| Feature | v1.0 | v1.1 |
|---------|------|------|
| tkinter GUI | ✓ | ✓ |
| Batch Processing | ✓ | ✓ |
| Dark Mode | ✗ | ✓ |
| Modern Themes | ✗ | ✓ |
| Keyboard Shortcuts | ✗ | ✓ |
| System Notifications | ✗ | ✓ |
| Platform Detection | Basic | Enhanced |
| Native Fonts | ✗ | ✓ |
| Theme Selection | ✗ | ✓ |
| Build Size | ~800MB | ~900MB |
| Startup Time | 2-3s | 2.5-3.5s |

---

## Support

### Build Issues

If you encounter build issues:

1. Check this guide's Troubleshooting section
2. Verify all dependencies are installed
3. Try a clean build
4. Check PyInstaller/py2app documentation
5. Review build logs for specific errors

### Runtime Issues

If the built application has runtime issues:

1. Test with `python src/gui/classmark_gui.py` directly
2. Check that all files are bundled correctly
3. Verify ttkbootstrap is in the bundle
4. Check platform detection is working
5. Review application logs

---

## Summary

Building Classmark v1.1 with ttkbootstrap integration is straightforward if you follow these steps:

1. **Install dependencies** (including ttkbootstrap)
2. **Update build scripts** (already done in this version)
3. **Clean build** to ensure all changes take effect
4. **Test thoroughly** on target platforms
5. **Create installers** for distribution

The build process is similar to v1.0, with the main addition being ttkbootstrap bundling. The spec files have been updated to handle this automatically.

**Estimated Total Time:**
- Windows build: 15-25 minutes
- macOS build: 15-25 minutes
- Testing: 30-60 minutes per platform

---

**Author:** Classmark Development Team
**Date:** 2025-11-19
**Version:** 1.1.0
