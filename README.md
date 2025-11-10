# Classification Marking Scanner - Desktop Edition

A desktop application for scanning documents for US Government security classification markings.

## Features

- **Multi-format support**: PDF, Word (DOCX), Text (TXT, MD, LOG, CSV), JSON
- **Pattern matching**: CAPCO-compliant classification marking detection
- **Fuzzy matching**: Handles variations and typos
- **LLM verification**: Optional verification via Ollama (local LLM) with **built-in downloader**
- **User-friendly GUI**: Simple point-and-click interface with tooltips and help
- **One-click LLM setup**: Install Ollama and download models directly from the GUI
- **Automatic file quarantine**: Move flagged files to a separate output folder
- **Batch processing**: Scan multiple files with parallel workers
- **Detailed reports**: Export results as JSON or CSV

## Supported Classification Levels

- TOP SECRET (with compartments like SCI, NOFORN, etc.)
- SECRET
- CONFIDENTIAL
- CUI (Controlled Unclassified Information)
- Portion markings: (TS), (S), (C), (U)
- Classification authority blocks

## System Requirements

### To Run the Executable (End Users)
- **Windows**: Windows 10 or higher (64-bit)
- **Linux**: Any modern distribution (64-bit)
- **macOS**: macOS 10.14 or higher
- **RAM**: 4 GB minimum, 8 GB recommended
- **Storage**: 200 MB free space

### To Build from Source (Developers)
- Python 3.8 or higher
- pip (Python package manager)
- 500 MB free space for dependencies

## Quick Start (End Users)

### Windows
1. Download `ClassificationScanner.exe`
2. Double-click to run
3. Click "Browse" to select a folder
4. Configure scan options (optional)
5. Click "Start Scan"
6. Review results and export reports if needed

### Linux/macOS
1. Download `ClassificationScanner`
2. Make executable: `chmod +x ClassificationScanner`
3. Run: `./ClassificationScanner`
4. Follow the same steps as Windows

## Building from Source

### Prerequisites

1. **Install Python 3.8+**
   - Windows: Download from [python.org](https://www.python.org/downloads/)
   - Linux: `sudo apt install python3 python3-pip` (Ubuntu/Debian)
   - macOS: `brew install python3`

2. **Clone or download this repository**

### Build Steps

#### Windows

1. Open Command Prompt in the project directory
2. Run the build script:
   ```cmd
   build.bat
   ```
3. Wait for the build to complete (5-10 minutes)
4. Find the executable in `dist\ClassificationScanner.exe`

#### Linux/macOS

1. Open Terminal in the project directory
2. Run the build script:
   ```bash
   ./build.sh
   ```
3. Wait for the build to complete (5-10 minutes)
4. Find the executable in `dist/ClassificationScanner`

### Manual Build (Alternative)

If the build scripts don't work, you can build manually:

```bash
# Install dependencies
pip install -r requirements.txt
pip install pyinstaller

# Build executable
pyinstaller classification_scanner_gui.spec

# Executable will be in dist/ folder
```

## Usage Guide

### GUI Application

1. **Select Directory**
   - Click "Browse" button
   - Navigate to the folder containing documents to scan
   - Click "Select Folder"

2. **Configure Output (Optional)**
   - **Move flagged files to output folder**: Enable to automatically quarantine classified files
   - **Output Directory**: Select destination folder for flagged files
   - Files will be moved with preserved directory structure
   - Automatic collision handling with timestamps

3. **Configure Scan Options**
   - **Scan subdirectories recursively**: Enable to scan all subfolders
   - **Enable fuzzy matching**: Helps detect variations and typos
   - **Use LLM verification**: Requires Ollama installed (see below)
   - **Sensitivity**: High (30%), Medium (50%), or Low (70%) confidence threshold
   - **Parallel Workers**: Number of files to scan simultaneously (1-8)

4. **Start Scan**
   - Click "Start Scan" button
   - Progress bar will show scanning activity
   - Results appear in the text area

5. **Review Results and Move Files**
   - Flagged files are shown in red
   - If file moving is enabled, you'll be prompted to confirm before moving
   - Move report shows successfully moved files and any errors
   - Click "Export JSON Report" or "Export CSV Report" to save results
   - Click "Clear Results" to reset

### Command Line Interface (Alternative)

You can also use the CLI version:

```bash
python classification_scanner.py /path/to/documents

# With options
python classification_scanner.py /path/to/documents \
    --sensitivity high \
    --fuzzy-matching \
    --workers 4 \
    --output report.json
```

## Optional: LLM Verification

For enhanced accuracy, you can enable LLM verification. The GUI includes a **built-in LLM downloader** for easy setup:

### Easy Setup (Recommended)

1. **Launch the GUI**
   ```bash
   python3 classification_scanner_gui.py
   ```

2. **In the "LLM Setup" section:**
   - Click "Install Ollama" button (downloads ~100-200 MB)
   - Follow the installation wizard
   - Click "Download Model" button (downloads ~4.7 GB)
   - Wait for download to complete (10-30 minutes)

3. **Enable in Scan Options**
   - Check "Use LLM verification" option
   - Scan will be slower but more accurate

### Manual Setup (Alternative)

If you prefer manual installation:

1. **Install Ollama**
   - Download from [ollama.ai](https://ollama.ai)
   - Follow installation instructions for your OS

2. **Download Model**
   ```bash
   ollama pull qwen3:8b
   ```

3. **The GUI will auto-detect** the installation

## Understanding Results

### Confidence Scores
- **90-100%**: Very high confidence (official marking format)
- **70-89%**: High confidence (likely official)
- **50-69%**: Medium confidence (review recommended)
- **30-49%**: Low confidence (possible false positive)

### Match Types
- **pattern**: Direct regex pattern match
- **fuzzy**: Fuzzy string match (handles typos)
- **structural**: Classification authority block detected

### Locations
- **header**: Found in document header
- **footer**: Found in document footer
- **body**: Found in document body
- **metadata**: Found in file metadata

## False Positives

The scanner minimizes false positives by:
- Context analysis around matches
- Filtering common phrases like "the secret to success"
- Requiring official indicators for high confidence

However, you should always review results, especially those with confidence < 70%.

## Troubleshooting

### "No module named X" error
**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### GUI doesn't start
**Solution**: Make sure tkinter is installed
```bash
# Ubuntu/Debian
sudo apt install python3-tk

# macOS (usually included)
brew install python-tk
```

### Build fails with PyInstaller
**Solution**: Update PyInstaller
```bash
pip install --upgrade pyinstaller
```

### "Access denied" on Windows
**Solution**: Run as administrator or disable antivirus temporarily

### Large executable size
**Solution**: This is normal. The executable includes Python runtime and all dependencies (~50-100 MB).

## Technical Details

### Architecture
- **Frontend**: Tkinter (Python standard library)
- **Backend**: Custom classification scanner
- **Threading**: Scan runs in separate thread to keep GUI responsive
- **Packaging**: PyInstaller creates single-file executable

### Dependencies
- PyPDF2 - PDF text extraction
- python-docx - Word document processing
- pdfplumber - Enhanced PDF layout (optional)
- rapidfuzz - Fuzzy string matching (optional)
- tqdm - Progress bars (optional)
- ollama - LLM verification (optional)

### Performance
- **Small scan** (< 100 files): < 1 minute
- **Medium scan** (100-1000 files): 1-5 minutes
- **Large scan** (1000+ files): 5-30 minutes

Use parallel workers to speed up large scans.

## Security & Privacy

- **100% local**: No data sent to external servers
- **No internet required**: Works completely offline (except LLM verification)
- **No telemetry**: No usage tracking or analytics
- **Open source**: Review the code yourself

## License

This tool is provided as-is for authorized use only. Use only for:
- Authorized security testing
- Defensive security operations
- Document review and compliance
- Educational purposes

Do NOT use for:
- Unauthorized access to systems
- Malicious purposes
- Circumventing security controls

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review error messages in the GUI status bar
3. Run CLI version for more detailed error messages
4. Check that all dependencies are installed

## Version History

### v2.2 Desktop Edition (2025-01-05)
- Added comprehensive help system (tooltips, Help menu, Quick Start Guide)
- Added automatic file quarantine feature
- Files with classification markings can be moved to output folder
- Preserves directory structure during file moves
- Improved user interface with context-sensitive help

### v2.1 Desktop Edition (2025-01-05)
- Initial desktop GUI release
- Removed image/OCR support for simpler deployment
- Added PyInstaller packaging
- Improved user experience

### v2.0 SOTA Edition
- Command-line interface
- Image OCR support
- Multi-layer detection

## Credits

Developed by Claude Code
Classification patterns based on CAPCO standards and Executive Order 13526
