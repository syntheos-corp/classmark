#!/bin/bash
################################################################################
# Classmark macOS DMG Installer Build Script
#
# This script creates a macOS disk image (.dmg) installer for Classmark
#
# Prerequisites:
#   1. Build the application first: ./build_macos.sh
#   2. (Optional) Download models: python3 download_models.py
#
# Usage:
#   chmod +x build_dmg.sh
#   ./build_dmg.sh
#
# Output:
#   Output/Classmark-1.0.0.dmg
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

# Configuration
APP_NAME="Classmark"
VERSION="1.0.0"
DMG_NAME="${APP_NAME}-${VERSION}"
DMG_DIR="dmg_temp"
OUTPUT_DIR="Output"

echo ""
echo "========================================================================"
echo "Classmark macOS DMG Installer Build Script"
echo "========================================================================"
echo ""

# Check if application is built
if [ ! -d "dist/${APP_NAME}.app" ]; then
    echo -e "${RED}[ERROR]${NC} Application not built"
    echo ""
    echo "Please build the application first:"
    echo "  ./build_macos.sh"
    echo ""
    exit 1
fi

echo -e "${GREEN}[OK]${NC} Application found: dist/${APP_NAME}.app"
echo ""

# Check if models are downloaded
if [ -f "models/models_config.json" ]; then
    echo -e "${GREEN}[OK]${NC} Models found - will be included in DMG"
    echo "     DMG size: ~1-2 GB with models"
    INCLUDE_MODELS=true
else
    echo -e "${YELLOW}[WARNING]${NC} Models not found"
    echo "          DMG will be created without models"
    echo "          Users will need to download models after installation"
    echo "          DMG size: ~500 MB without models"
    echo ""
    INCLUDE_MODELS=false
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Clean up old temp directory
if [ -d "$DMG_DIR" ]; then
    echo -e "${BLUE}Cleaning up old temp directory...${NC}"
    rm -rf "$DMG_DIR"
fi

# Create temporary directory for DMG contents
echo -e "${BLUE}Creating DMG structure...${NC}"
mkdir -p "$DMG_DIR"

# Copy application
echo -e "${BLUE}Copying application...${NC}"
cp -R "dist/${APP_NAME}.app" "$DMG_DIR/"

# Copy models if available
if [ "$INCLUDE_MODELS" = true ]; then
    echo -e "${BLUE}Copying models...${NC}"
    mkdir -p "$DMG_DIR/${APP_NAME}.app/Contents/Resources/models"
    cp -R models/* "$DMG_DIR/${APP_NAME}.app/Contents/Resources/models/"
fi

# Create Applications symlink for easy installation
echo -e "${BLUE}Creating Applications symlink...${NC}"
ln -s /Applications "$DMG_DIR/Applications"

# Copy README and documentation
if [ -f "README.md" ]; then
    cp "README.md" "$DMG_DIR/"
fi

# Create a nice looking DMG background (optional)
# You can add a custom background image here:
# mkdir -p "$DMG_DIR/.background"
# cp background.png "$DMG_DIR/.background/"

# Calculate DMG size (app size + 100MB buffer)
APP_SIZE=$(du -sm "dist/${APP_NAME}.app" | cut -f1)
DMG_SIZE=$((APP_SIZE + 100))

if [ "$INCLUDE_MODELS" = true ]; then
    MODELS_SIZE=$(du -sm models | cut -f1)
    DMG_SIZE=$((DMG_SIZE + MODELS_SIZE))
fi

echo ""
echo -e "${BLUE}Creating DMG image...${NC}"
echo "Size: ${DMG_SIZE}MB"
echo ""

# Remove old DMG if exists
if [ -f "$OUTPUT_DIR/${DMG_NAME}.dmg" ]; then
    rm "$OUTPUT_DIR/${DMG_NAME}.dmg"
fi

# Create DMG
hdiutil create -volname "$APP_NAME" \
    -srcfolder "$DMG_DIR" \
    -ov \
    -format UDZO \
    -size ${DMG_SIZE}m \
    "$OUTPUT_DIR/${DMG_NAME}.dmg"

if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}[ERROR]${NC} DMG creation failed"
    exit 1
fi

# Clean up
echo -e "${BLUE}Cleaning up...${NC}"
rm -rf "$DMG_DIR"

echo ""
echo "========================================================================"
echo -e "${GREEN}DMG Build Complete!${NC}"
echo "========================================================================"
echo ""
echo "DMG: $OUTPUT_DIR/${DMG_NAME}.dmg"
echo ""
echo "To test the installer:"
echo "  1. Open $OUTPUT_DIR/${DMG_NAME}.dmg"
echo "  2. Drag ${APP_NAME}.app to Applications folder"
echo "  3. Launch ${APP_NAME} from Applications"
echo ""

if [ "$INCLUDE_MODELS" = false ]; then
    echo -e "${YELLOW}IMPORTANT:${NC} Models were NOT included in this DMG"
    echo "Users will need to download models after installation:"
    echo "  1. Open Terminal"
    echo "  2. Navigate to /Applications/${APP_NAME}.app/Contents/Resources"
    echo "  3. Run: python3 download_models.py"
    echo ""
fi

# Optional: Create checksums
echo -e "${BLUE}Creating checksums...${NC}"
cd "$OUTPUT_DIR"
shasum -a 256 "${DMG_NAME}.dmg" > "${DMG_NAME}.dmg.sha256"
echo -e "${GREEN}[OK]${NC} SHA256: $(cat ${DMG_NAME}.dmg.sha256)"
cd ..

echo ""
echo "Distribution files ready in $OUTPUT_DIR/"
echo ""
