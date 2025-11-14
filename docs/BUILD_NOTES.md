# Build Notes for Classification Scanner Desktop App

## Important Build Information

### Platform-Specific Builds

PyInstaller creates platform-specific executables:
- **Windows**: Build on Windows → Creates `.exe` file
- **Linux**: Build on Linux → Creates Linux binary
- **macOS**: Build on macOS → Creates macOS app

**You cannot cross-compile.** To create a Windows .exe, you must build on Windows.

### Current Environment: WSL/Linux

This project was developed in WSL (Windows Subsystem for Linux). To build a **Windows executable**, you need to:

1. **Transfer files to Windows**
   - Files are already accessible at: `D:\Common_Resources\classmark\`
   - Open Windows PowerShell or Command Prompt
   - Navigate to: `cd D:\Common_Resources\classmark`

2. **Run the build script on Windows**
   ```cmd
   build.bat
   ```

3. **Result**: `dist\ClassificationScanner.exe` (~50-150 MB)

---

## Build Instructions by Platform

### Windows

#### Prerequisites
- Python 3.8+ installed
- Added to PATH during installation

#### Steps
```cmd
cd D:\Common_Resources\classmark
build.bat
```

**Output**: `dist\ClassificationScanner.exe`

#### Manual Build (if script fails)
```cmd
pip install -r requirements.txt
pip install pyinstaller
pyinstaller classification_scanner_gui.spec
```

---

### Linux

#### Prerequisites
```bash
sudo apt install python3 python3-pip python3-tk
```

#### Steps
```bash
cd /path/to/classmark
chmod +x build.sh
./build.sh
```

**Output**: `dist/ClassificationScanner`

---

### macOS

#### Prerequisites
```bash
brew install python3 python-tk
```

#### Steps
```bash
cd /path/to/classmark
chmod +x build.sh
./build.sh
```

**Output**: `dist/ClassificationScanner`

---

## Build Issues and Solutions

### Issue: "Cannot import name 'tarfile'"
**Solution**: Install backports.tarfile
```bash
pip install backports.tarfile
```

### Issue: Build takes very long (>10 minutes)
**Cause**: Anaconda Python includes many extra packages
**Solutions**:
1. Use system Python instead of Anaconda
2. Wait for build to complete (may take 10-20 minutes)
3. Use `--onefile` flag for smaller but slower executable

### Issue: Large executable size (>500 MB)
**Cause**: Anaconda environment or unnecessary dependencies
**Solutions**:
1. Use virtual environment with minimal dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   pip install pyinstaller
   pyinstaller classification_scanner_gui.spec
   ```

2. Exclude unnecessary packages in spec file

### Issue: "Module not found" errors when running executable
**Solution**: Add missing modules to `hiddenimports` in `.spec` file

---

## Expected Build Output

### File Sizes (approximate)
- **Minimal build** (system Python): 50-80 MB
- **Standard build**: 100-150 MB
- **Anaconda build**: 200-500 MB (includes extra dependencies)

### Build Time
- **Fast machine**: 2-5 minutes
- **Standard machine**: 5-10 minutes
- **Slow machine or Anaconda**: 10-20 minutes

### Directory Structure After Build
```
classmark/
├── dist/
│   └── ClassificationScanner(.exe)  ← THE EXECUTABLE
├── build/
│   └── classification_scanner_gui/  (temporary build files)
├── classification_scanner_gui.spec
└── ... (source files)
```

---

## Distribution

### What to Distribute
Only distribute the executable: `dist/ClassificationScanner` or `dist/ClassificationScanner.exe`

### What End Users Need
- **Nothing!** The executable is self-contained
- No Python installation required
- No dependencies required
- Just double-click to run

### Optional for LLM Features
If users want LLM verification:
1. They can use the built-in installer in the GUI
2. Or manually install from https://ollama.ai

---

## Testing the Build

### Before Distribution
Test the executable on a clean machine (or VM) without Python:

```bash
# Copy executable to test machine
# Double-click or run from terminal
./ClassificationScanner  # Linux/Mac
ClassificationScanner.exe  # Windows
```

### Verify
- ✓ Application launches
- ✓ GUI displays correctly
- ✓ Can browse and select directories
- ✓ Can run scans
- ✓ Can export reports
- ✓ LLM status checking works
- ✓ No errors in console (if running from terminal)

---

## Reducing Build Size (Optional)

If the executable is too large, try:

### 1. Use UPX Compression
Already enabled in spec file. To disable:
```python
upx=False  # in .spec file
```

### 2. Exclude Unnecessary Packages
Edit `classification_scanner_gui.spec`:
```python
excludes=[
    'matplotlib',
    'scipy',
    'pandas',
    'jupyter',
    # Add more as needed
]
```

### 3. Use Directory Build Instead
Uncomment COLLECT section in spec file for folder distribution instead of single file.

---

## CI/CD Build Automation

### GitHub Actions Example
```yaml
name: Build Executables

on: [push, pull_request]

jobs:
  build-windows:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt pyinstaller
      - run: pyinstaller classification_scanner_gui.spec
      - uses: actions/upload-artifact@v3
        with:
          name: ClassificationScanner-Windows
          path: dist/ClassificationScanner.exe

  build-linux:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: |
          sudo apt-get update
          sudo apt-get install -y python3-tk
      - run: pip install -r requirements.txt pyinstaller
      - run: pyinstaller classification_scanner_gui.spec
      - uses: actions/upload-artifact@v3
        with:
          name: ClassificationScanner-Linux
          path: dist/ClassificationScanner

  build-macos:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt pyinstaller
      - run: pyinstaller classification_scanner_gui.spec
      - uses: actions/upload-artifact@v3
        with:
          name: ClassificationScanner-macOS
          path: dist/ClassificationScanner
```

---

## Support

If you encounter build issues:
1. Check this document for common solutions
2. Verify Python version (3.8+)
3. Try building in a clean virtual environment
4. Check PyInstaller documentation: https://pyinstaller.org

---

## Build Information

- **PyInstaller Version**: 6.0+
- **Python Version**: 3.8+
- **Spec File**: `classification_scanner_gui.spec`
- **Main Script**: `classification_scanner_gui.py`
- **Backend**: `classification_scanner.py`
