#!/usr/bin/env python3
"""
Test script for GUI application structure
Verifies that the GUI class is properly defined without instantiating it
"""

import sys
import inspect

try:
    # Import without running
    import classification_scanner_gui as gui_module

    print("Testing GUI application structure...")
    print("")

    # Check that ScannerGUI class exists
    if not hasattr(gui_module, 'ScannerGUI'):
        print("✗ ScannerGUI class not found")
        sys.exit(1)

    print("✓ ScannerGUI class exists")

    ScannerGUI = gui_module.ScannerGUI

    # Check that __init__ method exists
    if not hasattr(ScannerGUI, '__init__'):
        print("✗ __init__ method not found")
        sys.exit(1)

    print("✓ __init__ method exists")

    # Check that key methods exist
    required_methods = [
        'create_widgets',
        'browse_directory',
        'start_scan',
        'stop_scan',
        'run_scan',
        'check_progress',
        'display_results',
        'scan_complete',
        'scan_error',
        'export_json',
        'export_csv',
        'clear_results',
    ]

    for method_name in required_methods:
        if not hasattr(ScannerGUI, method_name):
            print(f"✗ Method '{method_name}' missing")
            sys.exit(1)

        # Check that it's actually a method
        method = getattr(ScannerGUI, method_name)
        if not callable(method):
            print(f"✗ '{method_name}' is not callable")
            sys.exit(1)

        print(f"✓ Method '{method_name}' defined")

    # Check that main function exists
    if not hasattr(gui_module, 'main'):
        print("✗ main() function not found")
        sys.exit(1)

    print("✓ main() function exists")

    # Get method signatures to verify they look reasonable
    init_sig = inspect.signature(ScannerGUI.__init__)
    print(f"\n✓ __init__ signature: {init_sig}")

    print("")
    print("="*60)
    print("All GUI structure tests passed!")
    print("="*60)
    print("")
    print("The GUI application is properly structured.")
    print("")
    print("To run the GUI:")
    print("  python3 classification_scanner_gui.py")
    print("")
    print("To build executable:")
    print("  Windows: build.bat")
    print("  Linux/Mac: ./build.sh")

except Exception as e:
    print(f"✗ Error testing GUI structure: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
