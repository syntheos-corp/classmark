#!/usr/bin/env python3
"""
Test script for GUI application
Verifies that the GUI can be instantiated and key methods exist
"""

import sys
import tkinter as tk

# Don't actually display GUI in test
import os
os.environ['DISPLAY'] = ':99'  # Fake display for testing

try:
    from classification_scanner_gui import ScannerGUI

    print("Testing GUI application...")
    print("")

    # Create root window (won't display)
    root = tk.Tk()
    root.withdraw()  # Hide the window

    # Create GUI instance
    app = ScannerGUI(root)

    print("✓ GUI instantiated successfully")

    # Check that key attributes exist
    attributes_to_check = [
        'selected_dir',
        'recursive',
        'fuzzy_matching',
        'use_llm',
        'sensitivity',
        'workers',
        'is_scanning',
        'scan_results',
        'progress_queue',
    ]

    for attr in attributes_to_check:
        if hasattr(app, attr):
            print(f"✓ Attribute '{attr}' exists")
        else:
            print(f"✗ Attribute '{attr}' missing")
            sys.exit(1)

    # Check that key methods exist
    methods_to_check = [
        'browse_directory',
        'start_scan',
        'stop_scan',
        'run_scan',
        'check_progress',
        'display_results',
        'export_json',
        'export_csv',
        'clear_results',
    ]

    for method in methods_to_check:
        if hasattr(app, method) and callable(getattr(app, method)):
            print(f"✓ Method '{method}' exists")
        else:
            print(f"✗ Method '{method}' missing")
            sys.exit(1)

    print("")
    print("="*50)
    print("All GUI tests passed!")
    print("="*50)
    print("")
    print("The GUI application is ready to use.")
    print("Run it with: python3 classification_scanner_gui.py")

except Exception as e:
    print(f"✗ Error testing GUI: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
