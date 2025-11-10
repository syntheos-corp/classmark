#!/usr/bin/env python3
"""
Configuration and Constants for Classification Scanner

This module contains all configuration values, constants, and default settings
used throughout the application.

Author: Claude Code
Date: 2025-01-05
Version: 2.2
"""

# Application Information
APP_NAME = "Classification Marking Scanner"
APP_VERSION = "2.2"
APP_DATE = "2025-01-05"
APP_AUTHOR = "Claude Code"

# Window Configuration
WINDOW_TITLE = f"{APP_NAME}"
WINDOW_GEOMETRY = "900x700"
WINDOW_MIN_WIDTH = 800
WINDOW_MIN_HEIGHT = 600

# GUI Layout
PADDING_STANDARD = "10"
PADDING_SMALL = "5"
PADDING_FRAME = "2"

# Font Configuration
FONT_TITLE = ("Arial", 16, "bold")
FONT_SUBTITLE = ("Arial", 9)
FONT_NORMAL = ("Arial", 9)
FONT_ITALIC = ("Arial", 9, "italic")
FONT_RESULTS = ("Courier New", 9)
FONT_RESULTS_BOLD = ("Courier New", 9, "bold")
FONT_TOOLTIP = ("Arial", 9, "normal")

# Colors
COLOR_TOOLTIP_BG = "#ffffe0"
COLOR_TOOLTIP_FG = "#000000"
COLOR_FLAGGED = "red"
COLOR_CLEAN = "green"
COLOR_WARNING = "orange"
COLOR_STATUS_FG = "gray"

# Sensitivity Thresholds
SENSITIVITY_THRESHOLDS = {
    'high': 0.3,
    'medium': 0.5,
    'low': 0.7
}

# Workers Configuration
WORKERS_MIN = 1
WORKERS_MAX = 8
WORKERS_DEFAULT = 1

# Scan Configuration Defaults
DEFAULT_RECURSIVE = True
DEFAULT_FUZZY_MATCHING = True
DEFAULT_USE_LLM = False
DEFAULT_SENSITIVITY = "high"
DEFAULT_MOVE_FLAGGED = False

# LLM Configuration
LLM_MODEL_NAME = "qwen3:8b"
LLM_MODEL_SIZE_GB = 5.2
LLM_INSTALLER_SIZE_MB = "100-200"

# Ollama Download URLs
OLLAMA_URLS = {
    'Windows': 'https://ollama.com/download/OllamaSetup.exe',
    'Darwin': 'https://ollama.com/download/Ollama-darwin.zip',  # macOS
    'Linux': 'https://ollama.com/install.sh'
}

# Progress and Timing
PROGRESS_CHECK_INTERVAL_MS = 100
TOOLTIP_DELAY_MS = 500
LLM_CHECK_TIMEOUT_SEC = 5
LLM_LIST_TIMEOUT_SEC = 10
LLM_DOWNLOAD_TIMEOUT_MIN = 30

# File Operations
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
REPORT_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

# Security Limits
MAX_FILE_SIZE_MB = 100  # Maximum file size to process (in megabytes)
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Export Configuration
EXPORT_JSON_DEFAULT = "classification_scan"
EXPORT_CSV_DEFAULT = "classification_scan"

# Text Content
TEXT_ABOUT = f"""{APP_NAME}
Desktop Edition

Version: {APP_VERSION}
Date: {APP_DATE}

A desktop application for scanning documents
for US Government security classification markings.

Supports: PDF, DOCX, TXT, JSON files
Features: Pattern matching, fuzzy matching,
optional LLM verification, automatic file quarantine

Developed by: {APP_AUTHOR}
Based on: CAPCO standards and Executive Order 13526

License: For authorized use only
"""

TEXT_QUICK_START = """Quick Start Guide

1. SELECT DIRECTORY
   Click 'Browse' and select the folder containing
   documents you want to scan.

2. CONFIGURE OPTIONS (Optional)
   • Scan subdirectories - Include all subfolders
   • Fuzzy matching - Detect typos and variations
   • LLM verification - Use AI for better accuracy
   • Sensitivity - High/Medium/Low confidence threshold
   • Workers - Number of parallel scan processes

3. START SCAN
   Click 'Start Scan' button and wait for results.

4. REVIEW RESULTS
   Flagged files appear in red with details about
   classification markings found.

5. EXPORT REPORTS (Optional)
   Save results as JSON or CSV for further analysis.

TIPS:
• Use High sensitivity for comprehensive scanning
• Enable fuzzy matching to catch variations
• Use multiple workers for faster scanning of
  many files (1-8, default: 1)
"""

TEXT_REPORT_ISSUE = """Report an Issue

If you encounter a bug or have a feature request:

1. Check if the issue already exists
2. Gather information:
   • What were you trying to do?
   • What happened instead?
   • Any error messages?
   • Your operating system

3. Contact your system administrator or
   development team with this information.

For technical issues:
• Check console output for detailed errors
• Verify all prerequisites are installed
• Try running the CLI version for more details
"""

# Tooltip Text
TOOLTIPS = {
    'dir_entry': "Selected directory containing documents to scan",
    'browse_btn': "Click to select a folder containing documents to scan",
    'move_flagged_cb': "When enabled, files containing classification markings will be moved to the output folder",
    'output_entry': "Destination folder for flagged files",
    'output_browse_btn': "Select the folder where flagged files will be moved",
    'recursive_cb': "When enabled, scans all subfolders within the selected directory",
    'fuzzy_cb': "Detects classification markings even with typos or variations (e.g., 'SECR3T' for 'SECRET')",
    'llm_cb': "Uses a local AI model to verify markings and reduce false positives\n(Install Ollama first using the buttons below)",
    'sensitivity_combo': "Detection sensitivity:\n• High (30% confidence) - More results, some false positives\n• Medium (50% confidence) - Balanced\n• Low (70% confidence) - Fewer results, high confidence only",
    'workers_spin': "Number of files to scan simultaneously (1-8)\nHigher = faster for many files, but uses more CPU",
    'install_ollama_btn': "Download and install Ollama (local LLM runtime)\nRequired for LLM verification feature (~100-200 MB)",
    'download_model_btn': "Download the AI model for classification verification\nRequires ~4.7 GB and takes 10-30 minutes",
    'scan_button': "Begin scanning the selected directory for classification markings\nMake sure to select a directory first",
    'stop_button': "Stop the currently running scan",
    'help_button': "Open the Quick Start Guide to learn how to use this application",
    'results_text': "Detailed scan results will appear here\nRed text indicates files with classification markings found",
    'export_json_btn': "Save scan results as JSON file\nDetailed format for programmatic analysis",
    'export_csv_btn': "Save scan results as CSV file\nSpreadsheet-compatible format for easy review",
    'clear_btn': "Clear the results display and reset for a new scan",
}

# Status Bar Messages
STATUS_MESSAGES = {
    'ready': "Ready to scan. Click 'Browse' to select a directory or go to Help > Quick Start Guide",
    'browse': "Select a folder containing documents to scan for classification markings",
    'recursive': "Enable to scan all subfolders within the selected directory",
    'fuzzy': "Fuzzy matching helps detect classification markings with typos or variations",
    'llm': "LLM verification uses AI to improve accuracy and reduce false positives (requires Ollama)",
    'sensitivity': "High sensitivity finds more matches but may include false positives; Low sensitivity is more strict",
    'workers': "More workers = faster scanning of many files, but uses more CPU (recommended: 2-4)",
    'scan': "Start scanning the selected directory with current options",
    'export_json': "Export detailed scan results in JSON format for programmatic analysis",
    'export_csv': "Export scan results in CSV format for viewing in spreadsheet applications",
}

# Supported File Extensions
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.txt', '.text', '.md', '.log', '.json'}

# Result Display Configuration
MAX_MATCHES_DISPLAY = 3
MAX_FILES_IN_CONFIRMATION = 10
RESULTS_SEPARATOR = "=" * 80
RESULTS_SUBSEPARATOR = "-" * 80
