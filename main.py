#!/usr/bin/env python3
"""
Entry point for the scikit-maad marine acoustics GUI application.
This script sets up the proper paths and launches the main GUI.
"""

import os
import sys
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Import and run the main GUI
if __name__ == "__main__":
    # Import the GUI module (this will start the GUI since it ends with mainloop())
    import gui.main_gui