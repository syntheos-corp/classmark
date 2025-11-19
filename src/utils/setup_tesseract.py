#!/usr/bin/env python3
"""
Automatic Tesseract OCR Setup for Windows

Downloads and configures Tesseract OCR binaries.
This ensures end users don't need to manually install Tesseract.

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
import tempfile
import subprocess

# Tesseract portable release info
# Using a portable build that includes tesseract.exe + tessdata
TESSERACT_VERSION = "5.3.3"
# We'll use the official UB Mannheim installer and extract it
TESSERACT_WINDOWS_URL = "https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.3.3.20231005.exe"

# Alternative: Direct download of portable build (if available)
# For now, we'll download the installer and provide extraction instructions
# In a future update, we can host a pre-extracted portable build

# Installation paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
EXTERNAL_DIR = PROJECT_ROOT / "external"
TESSERACT_DIR = EXTERNAL_DIR / "tesseract"
TESSERACT_EXE = TESSERACT_DIR / "tesseract.exe"
TESSDATA_DIR = TESSERACT_DIR / "tessdata"


def is_windows():
    """Check if running on Windows"""
    return platform.system() == "Windows"


def tesseract_is_installed():
    """Check if tesseract is already installed in the project"""
    # Check if tesseract.exe exists
    if not TESSERACT_EXE.exists():
        return False

    # Check if tessdata directory exists with eng.traineddata
    eng_data = TESSDATA_DIR / "eng.traineddata"
    return eng_data.exists()


def download_tesseract():
    """Download Tesseract installer for Windows"""
    print(f"Downloading Tesseract {TESSERACT_VERSION} for Windows...")
    print(f"This includes tesseract.exe + English language data (~90MB)")
    print(f"URL: {TESSERACT_WINDOWS_URL}")
    print()

    # Create directories
    EXTERNAL_DIR.mkdir(exist_ok=True)

    # Download file
    installer_path = EXTERNAL_DIR / f"tesseract-{TESSERACT_VERSION}-setup.exe"

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
            TESSERACT_WINDOWS_URL,
            installer_path,
            reporthook=show_progress
        )
        print("\n✓ Download complete!")

        return installer_path

    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print("\nAlternative: Download manually from:")
        print(TESSERACT_WINDOWS_URL)
        print(f"Then run setup manually")
        return None


def install_tesseract(installer_path):
    """Install Tesseract from installer to our external directory"""
    print(f"Installing Tesseract to {TESSERACT_DIR}...")

    try:
        # Remove old installation if exists
        if TESSERACT_DIR.exists():
            print("Removing old installation...")
            shutil.rmtree(TESSERACT_DIR)

        # Create tesseract directory
        TESSERACT_DIR.mkdir(parents=True, exist_ok=True)

        # Run installer silently with custom directory
        # NSIS installer supports /S (silent) and /D (directory) flags
        install_cmd = [
            str(installer_path),
            '/S',  # Silent install
            f'/D={str(TESSERACT_DIR)}'  # Custom directory (must be last parameter)
        ]

        print("Running silent installer...")
        print(f"Command: {' '.join(install_cmd)}")

        # Run installer and wait for completion
        result = subprocess.run(
            install_cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        # NSIS installers may return before fully complete, so wait a bit
        import time
        print("Waiting for installation to complete...")
        time.sleep(5)

        # Verify installation
        if TESSERACT_EXE.exists():
            print("✓ Installation complete!")

            # Clean up installer
            installer_path.unlink()

            return True
        else:
            print("✗ Installation verification failed - tesseract.exe not found")
            print(f"Expected at: {TESSERACT_EXE}")
            return False

    except subprocess.TimeoutExpired:
        print("✗ Installation timed out")
        return False
    except Exception as e:
        print(f"✗ Installation failed: {e}")
        return False


def verify_installation():
    """Verify Tesseract installation"""
    if not TESSERACT_EXE.exists():
        print(f"✗ tesseract.exe not found at: {TESSERACT_EXE}")
        return False

    eng_data = TESSDATA_DIR / "eng.traineddata"
    if not eng_data.exists():
        print(f"✗ English language data not found at: {eng_data}")
        return False

    print(f"✓ Tesseract installed successfully!")
    print(f"  Executable: {TESSERACT_EXE}")
    print(f"  Language data: {TESSDATA_DIR}")

    return True


def get_tesseract_path():
    """
    Get the path to tesseract executable and configure TESSDATA_PREFIX.

    Priority:
    1. Bundled version in external/tesseract/ (auto-downloaded)
    2. System PATH
    3. Common install locations

    Returns:
        Path to tesseract.exe, or None if not installed
    """
    # Priority 1: Check bundled version first (our auto-downloaded version)
    if TESSERACT_EXE.exists():
        tesseract_path = str(TESSERACT_EXE)
        _configure_tessdata(tesseract_path)
        return tesseract_path

    # Priority 2: Check system PATH
    system_tesseract = shutil.which("tesseract")
    if system_tesseract:
        _configure_tessdata(system_tesseract)
        return system_tesseract

    # Priority 3: Check common install locations
    common_paths = [
        "C:/Program Files/Tesseract-OCR/tesseract.exe",
        "C:/Program Files (x86)/Tesseract-OCR/tesseract.exe",
        Path.home() / "AppData/Local/Tesseract-OCR/tesseract.exe",
        Path.home() / "AppData/Local/Programs/Tesseract-OCR/tesseract.exe"
    ]

    for path in common_paths:
        if Path(path).exists():
            tesseract_path = str(path)
            _configure_tessdata(tesseract_path)
            return tesseract_path

    return None


def _configure_tessdata(tesseract_exe_path):
    """
    Configure TESSDATA_PREFIX environment variable for Tesseract.

    Args:
        tesseract_exe_path: Path to tesseract.exe
    """
    # Get directory containing tesseract.exe
    tesseract_dir = Path(tesseract_exe_path).parent

    # tessdata should be in same directory or in parent
    tessdata_paths = [
        tesseract_dir / "tessdata",
        tesseract_dir.parent / "tessdata"
    ]

    for tessdata_path in tessdata_paths:
        if tessdata_path.exists():
            # Set environment variable
            os.environ["TESSDATA_PREFIX"] = str(tessdata_path)
            print(f"  Set TESSDATA_PREFIX: {tessdata_path}")
            return

    # If not found, try to set to default location
    # Tesseract usually expects tessdata to be in the same dir as the exe
    default_tessdata = tesseract_dir / "tessdata"
    os.environ["TESSDATA_PREFIX"] = str(default_tessdata)
    print(f"  Warning: tessdata not found, set to: {default_tessdata}")


def setup_tesseract(force_reinstall=False, auto_download=True):
    """
    Main setup function - downloads and installs Tesseract if needed.

    Args:
        force_reinstall: If True, reinstall even if already present
        auto_download: If True, automatically download and install Tesseract

    Returns:
        Path to tesseract executable, or None on failure
    """
    if not is_windows():
        print("Tesseract auto-setup is only needed on Windows.")
        print("On Linux/Mac, install via package manager:")
        print("  Ubuntu/Debian: sudo apt-get install tesseract-ocr")
        print("  Mac: brew install tesseract")
        return None

    # Check if already installed (bundled or system)
    if not force_reinstall:
        tesseract_path = get_tesseract_path()
        if tesseract_path:
            print(f"✓ Tesseract found: {tesseract_path}")
            return tesseract_path

    # Check if our bundled version exists
    if tesseract_is_installed() and not force_reinstall:
        print(f"✓ Tesseract already bundled at: {TESSERACT_DIR}")
        return str(TESSERACT_EXE)

    # Auto-download if enabled
    if not auto_download:
        print("Tesseract OCR not found!")
        print()
        print("To enable OCR for scanned documents, install Tesseract:")
        print()
        print("Option 1 (Recommended): Auto-install")
        print("  Run: python src/utils/setup_tesseract.py")
        print()
        print("Option 2: Manual install via installer")
        print("  Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        print()
        print("Option 3: Install via package manager")
        print("  winget install UB-Mannheim.Tesseract")
        print()
        return None

    # Auto-download and install
    print("=" * 60)
    print("Tesseract Auto-Setup for Classmark")
    print("=" * 60)
    print()

    # Download
    installer_path = download_tesseract()
    if not installer_path:
        return None

    # Install
    if not install_tesseract(installer_path):
        return None

    # Verify
    if verify_installation():
        return str(TESSERACT_EXE)

    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Setup Tesseract OCR for Classmark")
    parser.add_argument("--force", action="store_true", help="Force reinstall even if already present")
    parser.add_argument("--no-auto", action="store_true", help="Don't auto-download, just check for existing installation")
    args = parser.parse_args()

    result = setup_tesseract(
        force_reinstall=args.force,
        auto_download=not args.no_auto
    )

    if result:
        print("\n" + "=" * 60)
        print("Setup Complete!")
        print("=" * 60)
        print(f"Tesseract: {result}")
        print(f"Language data: {TESSDATA_DIR}")
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("Setup Incomplete")
        print("=" * 60)
        print("OCR features will not work until Tesseract is installed.")
        print("\nTo auto-install, run: python src/utils/setup_tesseract.py")
        sys.exit(1)
