#!/usr/bin/env python3
"""
One-Click Dependency Setup for Classmark

Automatically downloads and configures all required dependencies:
- Poppler binaries (for PDF processing)
- AI models (optional, downloaded on first use)

Run this after pip install to complete setup.

Author: Classmark Development Team
Date: 2025-01-18
"""

import sys
import platform
from pathlib import Path

print("=" * 70)
print("Classmark Dependency Setup")
print("=" * 70)
print()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Step 1: Setup Poppler (Windows only)
if platform.system() == "Windows":
    print("[1/2] Setting up Poppler binaries for PDF processing...")
    print()

    try:
        from src.utils.setup_poppler import setup_poppler

        poppler_path = setup_poppler()

        if poppler_path:
            print()
            print("✓ Poppler setup complete!")
        else:
            print()
            print("⚠ Poppler setup incomplete. PDF processing may not work.")
            print("  You can manually download from:")
            print("  https://github.com/oschwartz10612/poppler-windows/releases")

    except Exception as e:
        print(f"✗ Error setting up Poppler: {e}")
        print("  PDF processing may not work without Poppler.")

    print()
else:
    print("[1/2] Poppler setup (Linux/Mac)")
    print("Please install poppler-utils via package manager:")
    print("  Ubuntu/Debian: sudo apt-get install poppler-utils")
    print("  Mac: brew install poppler")
    print()

# Step 2: Setup Tesseract OCR (Windows only)
if platform.system() == "Windows":
    print("[2/3] Setting up Tesseract OCR...")
    print()

    try:
        from src.utils.setup_tesseract import setup_tesseract

        tesseract_path = setup_tesseract()

        if tesseract_path:
            print()
            print("✓ Tesseract setup complete!")
        else:
            print()
            print("⚠ Tesseract not found. OCR features will be limited.")
            print("  Install from: https://github.com/UB-Mannheim/tesseract/wiki")

    except Exception as e:
        print(f"✗ Error checking Tesseract: {e}")
        print("  OCR features may not work without Tesseract.")

    print()
else:
    print("[2/3] Tesseract setup (Linux/Mac)")
    print("Please install tesseract-ocr via package manager:")
    print("  Ubuntu/Debian: sudo apt-get install tesseract-ocr")
    print("  Mac: brew install tesseract")
    print()

# Step 3: Check AI models
print("[3/3] Checking AI models...")
print()
print("AI models will be downloaded automatically on first use.")
print("To pre-download models now, run:")
print("  python src/utils/download_models.py")
print()

print("=" * 70)
print("Setup Complete!")
print("=" * 70)
print()
print("Next steps:")
print("  1. Run GUI: python src/gui/classmark_gui.py")
print("  2. Or run CLI: python src/core/classification_scanner.py --help")
print()
