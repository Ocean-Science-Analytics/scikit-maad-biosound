#!/usr/bin/env python3
"""
Debug configuration module for controlling output verbosity.

Global flags that can be set by main.py and used throughout the application:
- DEBUG_MODE: Controls [DEBUG] messages for troubleshooting multiprocessing, data flow, etc.
- VERBOSE_MODE: Controls [STANDALONE] and other verbose processing messages
"""

# Default to quiet operation
DEBUG_MODE = False
VERBOSE_MODE = False

def debug_print(message):
    """Print debug message only if DEBUG_MODE is enabled"""
    if DEBUG_MODE:
        print(message)

def verbose_print(message):
    """Print verbose message only if VERBOSE_MODE is enabled"""
    if VERBOSE_MODE:
        print(message)