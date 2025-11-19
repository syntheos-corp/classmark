#!/usr/bin/env python3
"""
Automatic Poppler Binary Setup for Windows

Downloads and configures poppler-utils binaries for pdf2image.
This ensures end users don't need to manually install poppler.

Author: Classmark Development Team
Date: 2025-01-18
"""

import os
import sys
import platform
import zipfile
import shutil
from pathlib import Path
import urllib.request
import ssl

# Poppler release info
POPPLER_VERSION = "24.08.0-0"
POPPLER_WINDOWS_URL = f"https://github.com/oschwartz10612/poppler-windows/releases/download/v{POPPLER_VERSION}/Release-{POPPLER_VERSION}.zip"

# Installation paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
EXTERNAL_DIR = PROJECT_ROOT / "external"
POPPLER_DIR = EXTERNAL_DIR / "poppler"
POPPLER_BIN_DIR = POPPLER_DIR / "Library" / "bin"


def is_windows():
    """Check if running on Windows"""
    return platform.system() == "Windows"


def poppler_is_installed():
    """Check if poppler is already installed in the project"""
    # Check if pdftoppm exists in our external directory
    pdftoppm_path = POPPLER_BIN_DIR / "pdftoppm.exe"
    return pdftoppm_path.exists()


def download_poppler():
    """Download poppler binaries for Windows"""
    print(f"Downloading Poppler {POPPLER_VERSION} for Windows...")
    print(f"URL: {POPPLER_WINDOWS_URL}")

    # Create directories
    EXTERNAL_DIR.mkdir(exist_ok=True)

    # Download file
    zip_path = EXTERNAL_DIR / f"poppler-{POPPLER_VERSION}.zip"

    try:
        # Create SSL context that doesn't verify certificates (for corporate proxies)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        # Download with progress
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, (downloaded / total_size) * 100) if total_size > 0 else 0
            print(f"\rDownloading: {percent:.1f}% ({downloaded // 1024 // 1024}MB / {total_size // 1024 // 1024}MB)", end='')

        urllib.request.urlretrieve(
            POPPLER_WINDOWS_URL,
            zip_path,
            reporthook=show_progress
        )
        print("\n✓ Download complete!")

        return zip_path

    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print("\nAlternative: Download manually from:")
        print(POPPLER_WINDOWS_URL)
        print(f"Then extract to: {POPPLER_DIR}")
        return None


def extract_poppler(zip_path):
    """Extract poppler binaries"""
    print(f"Extracting poppler to {POPPLER_DIR}...")

    try:
        # Remove old installation if exists
        if POPPLER_DIR.exists():
            shutil.rmtree(POPPLER_DIR)

        # Extract
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract to temporary directory first
            temp_dir = EXTERNAL_DIR / "temp_poppler"
            zip_ref.extractall(temp_dir)

            # Find the actual poppler directory (it's usually nested)
            extracted_dirs = list(temp_dir.glob("poppler-*"))
            if extracted_dirs:
                actual_poppler_dir = extracted_dirs[0]
                shutil.move(str(actual_poppler_dir), str(POPPLER_DIR))
            else:
                # If not nested, rename temp dir
                shutil.move(str(temp_dir), str(POPPLER_DIR))

        # Clean up
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        zip_path.unlink()

        print("✓ Extraction complete!")
        return True

    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False


def verify_installation():
    """Verify poppler installation"""
    pdftoppm_path = POPPLER_BIN_DIR / "pdftoppm.exe"

    if pdftoppm_path.exists():
        print(f"✓ Poppler installed successfully at: {POPPLER_BIN_DIR}")
        return True
    else:
        print(f"✗ Poppler installation failed - pdftoppm.exe not found")
        return False


def get_poppler_path():
    """
    Get the path to poppler binaries.

    Returns:
        Path to poppler bin directory, or None if not installed
    """
    if POPPLER_BIN_DIR.exists():
        return str(POPPLER_BIN_DIR)
    return None


def setup_poppler(force_reinstall=False):
    """
    Main setup function - downloads and installs poppler if needed.

    Args:
        force_reinstall: If True, reinstall even if already present

    Returns:
        Path to poppler bin directory, or None on failure
    """
    if not is_windows():
        print("Poppler auto-setup is only needed on Windows.")
        print("On Linux/Mac, install via package manager:")
        print("  Ubuntu/Debian: sudo apt-get install poppler-utils")
        print("  Mac: brew install poppler")
        return None

    # Check if already installed
    if poppler_is_installed() and not force_reinstall:
        print(f"✓ Poppler already installed at: {POPPLER_BIN_DIR}")
        return str(POPPLER_BIN_DIR)

    print("=" * 60)
    print("Poppler Setup for Classmark")
    print("=" * 60)

    # Download
    zip_path = download_poppler()
    if not zip_path:
        return None

    # Extract
    if not extract_poppler(zip_path):
        return None

    # Verify
    if verify_installation():
        return str(POPPLER_BIN_DIR)

    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Setup Poppler binaries for Classmark")
    parser.add_argument("--force", action="store_true", help="Force reinstall even if already present")
    args = parser.parse_args()

    result = setup_poppler(force_reinstall=args.force)

    if result:
        print("\n" + "=" * 60)
        print("Setup Complete!")
        print("=" * 60)
        print(f"Poppler binaries: {result}")
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("Setup Failed")
        print("=" * 60)
        sys.exit(1)
