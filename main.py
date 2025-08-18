#!/usr/bin/env python3
"""
Entry point for the scikit-maad marine acoustics GUI application.
This script sets up the proper paths and launches the main GUI.

Usage:
    python main.py              # Normal operation
    python main.py --debug      # Enable debug output
    python main.py --verbose    # Enable verbose processing output
"""

import os
import sys
import argparse
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def main():
    parser = argparse.ArgumentParser(description='Scikit-MAAD Acoustic Indices GUI')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug output for troubleshooting')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose processing output')
    
    args = parser.parse_args()
    
    # Set global debug flags that modules can import
    import gui.debug_config
    gui.debug_config.DEBUG_MODE = args.debug
    gui.debug_config.VERBOSE_MODE = args.verbose
    
    # Import and run the main GUI
    import gui.main_gui

if __name__ == "__main__":
    main()