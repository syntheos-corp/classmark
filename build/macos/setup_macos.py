#!/usr/bin/env python3
"""
py2app Setup Script for Classmark (macOS)

This script builds a standalone macOS application bundle (.app) for Classmark.
The application includes all dependencies and can run completely offline
after models are downloaded.

Prerequisites:
    pip install py2app

Usage:
    # Build in alias mode (for development/testing)
    python setup_macos.py py2app -A

    # Build standalone application
    python setup_macos.py py2app

Output:
    dist/Classmark.app

Author: Classmark Development Team
Date: 2025-11-10
"""

from setuptools import setup
import os
import sys

# Application name and version
APP_NAME = 'Classmark'
VERSION = '1.0.0'
DESCRIPTION = 'SOTA Classification Marking Detection System'
AUTHOR = 'Classmark Development Team'

# Main script
APP = ['../../src/gui/classmark_gui.py']

# Data files to include
DATA_FILES = []

# Add pre-downloaded models (CRITICAL for offline operation)
# These models must be downloaded before building the installer
models_dir = '../../models'
if os.path.exists(models_dir):
    # Bundle LayoutLMv3 model (~500MB)
    layoutlm_path = os.path.join(models_dir, 'layoutlmv3-base')
    if os.path.exists(layoutlm_path):
        DATA_FILES.append((layoutlm_path, 'models/layoutlmv3-base'))
        print(f"✓ Bundling LayoutLMv3 model from {layoutlm_path}")
    else:
        print(f"⚠ Warning: LayoutLMv3 model not found at {layoutlm_path}")

    # Bundle YOLO model (~6MB)
    yolo_path = os.path.join(models_dir, 'yolo', 'yolov8n.pt')
    if os.path.exists(yolo_path):
        DATA_FILES.append((yolo_path, 'models/yolo'))
        print(f"✓ Bundling YOLO model from {yolo_path}")
    else:
        print(f"⚠ Warning: YOLO model not found at {yolo_path}")
else:
    print(f"⚠ Warning: Models directory not found at {models_dir}")
    print("  Run 'python src/utils/download_models.py' before building")

# Add documentation files
for doc_file in ['README.md', 'PROJECT_SUMMARY.md', 'PHASE3_SUMMARY.md', 'PHASE4_SUMMARY.md']:
    doc_path = os.path.join('../..', doc_file)
    if os.path.exists(doc_path):
        DATA_FILES.append(doc_path)

# Add ground truth data if exists
gt_path = '../../ground_truth.json'
if os.path.exists(gt_path):
    DATA_FILES.append(gt_path)

# Options for py2app
OPTIONS = {
    'py2app': {
        'argv_emulation': False,  # Don't emulate sys.argv
        'packages': [
            # Core Python packages
            'tkinter',
            'json',
            'csv',
            'datetime',
            'dataclasses',
            'enum',
            'typing',
            'pathlib',

            # PyTorch and related
            'torch',
            'torchvision',

            # Transformers and HuggingFace
            'transformers',
            'transformers.models',
            'transformers.models.layoutlmv3',
            'tokenizers',

            # Ultralytics YOLO
            'ultralytics',
            'ultralytics.nn',
            'ultralytics.yolo',

            # OCR dependencies
            'pytesseract',
            'PIL',
            'pdf2image',

            # Pattern matching
            'ahocorasick',

            # Document processing
            'PyPDF2',
            'docx',
            'pdfplumber',
            'docx2python',

            # Scientific computing
            'numpy',
            'scipy',
            'sklearn',
            'sklearn.linear_model',
            'sklearn.isotonic',
            'sklearn.metrics',

            # Utilities
            'rapidfuzz',
            'tqdm',
            'requests',

            # ttkbootstrap (modern themes)
            'ttkbootstrap',

            # Classmark modules (in src/)
            'src',
            'src.core',
            'src.core.classification_scanner',
            'src.core.fast_pattern_matcher',
            'src.core.visual_pattern_detector',
            'src.core.yolo_ocr',
            'src.core.simple_ocr',
            'src.core.hybrid_classifier',
            'src.core.calibrated_scorer',
            'src.core.evaluation',
            'src.config',
            'src.config.offline_config',
            'src.gui',
            'src.gui.classmark_gui',
            'src.gui.platform_utils',
        ],
        'includes': [
            # Additional modules to include
            'tkinter.ttk',
            'tkinter.filedialog',
            'tkinter.messagebox',
            'tkinter.scrolledtext',
            'tkinter.simpledialog',

            # ttkbootstrap submodules
            'ttkbootstrap.constants',
            'ttkbootstrap.themes',
            'ttkbootstrap.style',
            'ttkbootstrap.window',
        ],
        'excludes': [
            # Exclude unnecessary packages
            'matplotlib',
            'IPython',
            'notebook',
            'jupyter',
            'pytest',
            'setuptools',
        ],
        'frameworks': [
            # macOS frameworks if needed
        ],
        'resources': DATA_FILES,
        'plist': {
            'CFBundleName': APP_NAME,
            'CFBundleDisplayName': APP_NAME,
            'CFBundleIdentifier': 'com.classmark.app',
            'CFBundleVersion': VERSION,
            'CFBundleShortVersionString': VERSION,
            'NSHumanReadableCopyright': f'Copyright © 2025 {AUTHOR}',
            'NSHighResolutionCapable': True,
            'LSMinimumSystemVersion': '10.13',  # macOS High Sierra or later
            'CFBundleDocumentTypes': [
                {
                    'CFBundleTypeName': 'PDF Document',
                    'CFBundleTypeRole': 'Viewer',
                    'LSHandlerRank': 'Alternate',
                    'LSItemContentTypes': ['com.adobe.pdf'],
                },
                {
                    'CFBundleTypeName': 'Word Document',
                    'CFBundleTypeRole': 'Viewer',
                    'LSHandlerRank': 'Alternate',
                    'LSItemContentTypes': ['org.openxmlformats.wordprocessingml.document'],
                },
                {
                    'CFBundleTypeName': 'Text Document',
                    'CFBundleTypeRole': 'Viewer',
                    'LSHandlerRank': 'Alternate',
                    'LSItemContentTypes': ['public.plain-text'],
                },
            ],
        },
        # 'iconfile': 'classmark.icns',  # Add icon file if available
        'arch': 'universal2',  # Build for Intel and Apple Silicon
        'optimize': 2,  # Optimize bytecode
        'compressed': True,  # Compress files
        'semi_standalone': False,  # Full standalone app
        'site_packages': True,  # Include site-packages
    }
}

# Setup configuration
setup(
    name=APP_NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    app=APP,
    data_files=DATA_FILES,
    options=OPTIONS,
    setup_requires=['py2app'],
)
