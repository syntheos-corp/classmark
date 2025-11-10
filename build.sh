#!/bin/bash
# Build script for Linux/Mac
# This script builds the Classification Scanner executable

echo "========================================"
echo "Classification Scanner - Build Script"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "[1/4] Installing dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install pyinstaller

echo ""
echo "[2/4] Cleaning previous builds..."
rm -rf build dist __pycache__

echo ""
echo "[3/4] Building executable with PyInstaller..."
python3 -m PyInstaller classification_scanner_gui.spec

echo ""
echo "[4/4] Checking build result..."
if [ -f "dist/ClassificationScanner" ]; then
    echo ""
    echo "========================================"
    echo "BUILD SUCCESSFUL!"
    echo "========================================"
    echo ""
    echo "Executable location: dist/ClassificationScanner"
    echo ""
    echo "You can now distribute the file:"
    echo "  dist/ClassificationScanner"
    echo ""
else
    echo ""
    echo "========================================"
    echo "BUILD FAILED!"
    echo "========================================"
    echo ""
    echo "Please check the error messages above."
    echo ""
    exit 1
fi
