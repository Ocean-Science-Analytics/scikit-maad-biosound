#!/usr/bin/env python3
"""
Entry point for the scikit-maad marine acoustics GUI application.
This script sets up the proper paths and launches the main GUI.

Usage:
    python main.py              # Normal operation
    python main.py --debug      # Enable debug test_outputs
    python main.py --verbose    # Enable verbose processing test_outputs
"""

import argparse

def main():
    parser = argparse.ArgumentParser(description='Scikit-MAAD Acoustic Indices GUI')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug test_outputs for troubleshooting')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose processing test_outputs')
    
    args = parser.parse_args()
    
    # Set global debug flags that modules can import
    from src.utils import debug_utils
    debug_utils.configure_debug(debug=args.debug, verbose=args.verbose)
    
    # Import and run the main GUI
    from src.gui import main_gui

if __name__ == "__main__":
    main()