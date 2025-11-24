#!/bin/bash
################################################################################
# Classmark macOS Build Script
#
# This script builds the macOS application bundle using py2app
#
# Usage:
#   chmod +x build_macos.sh
#   ./build_macos.sh [clean|alias]
#
# Options:
#   clean - Clean build directories before building
#   alias - Build in alias mode (for development/testing)
#
# Author: Classmark Development Team
# Date: 2025-11-10
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo "========================================================================"
echo "Classmark macOS Build Script"
echo "========================================================================"
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Python 3 is not installed"
    exit 1
fi

# Check if py2app is installed
python3 -c "import py2app" 2>/dev/null || {
    echo -e "${YELLOW}[WARNING]${NC} py2app is not installed"
    echo "Installing py2app..."
    pip3 install py2app
}

# Clean build directories if requested
if [ "$1" == "clean" ]; then
    echo -e "${BLUE}Cleaning build directories...${NC}"
    rm -rf build dist
    echo -e "${GREEN}[OK]${NC} Build directories cleaned"
    echo ""
fi

# Check if requirements are installed
echo -e "${BLUE}Checking dependencies...${NC}"
python3 -c "import torch, transformers, ultralytics, tkinter, ttkbootstrap" 2>/dev/null || {
    echo -e "${YELLOW}[WARNING]${NC} Some dependencies may be missing"
    echo "Please run: pip3 install -r requirements.txt"
    echo ""
    echo "Note: ttkbootstrap is required for modern UI themes"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
}

# Build mode
BUILD_MODE=""
if [ "$1" == "alias" ] || [ "$2" == "alias" ]; then
    BUILD_MODE="-A"
    echo -e "${BLUE}Building in alias mode (for development)...${NC}"
else
    echo -e "${BLUE}Building standalone application...${NC}"
fi

# Build the application
echo ""
echo "This may take several minutes (5-15 minutes depending on your system)"
echo ""

python3 setup_macos.py py2app $BUILD_MODE

if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}[ERROR]${NC} Build failed"
    echo "Please check the error messages above"
    exit 1
fi

echo ""
echo "========================================================================"
echo -e "${GREEN}Build Complete!${NC}"
echo "========================================================================"
echo ""
echo "Output: dist/Classmark.app"
echo ""
echo "To test the application:"
echo "  open dist/Classmark.app"
echo ""
echo "To create a DMG installer:"
echo "  ./build_dmg.sh"
echo ""
