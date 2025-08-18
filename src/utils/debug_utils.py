#!/usr/bin/env python3
"""
Shared debug utilities for controlling output verbosity across the application.

This module provides debug and verbose printing functions that can be used
by both GUI and processing modules without creating circular dependencies.

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

def set_debug_mode(enabled):
    """Enable or disable debug mode"""
    global DEBUG_MODE
    DEBUG_MODE = enabled

def set_verbose_mode(enabled):
    """Enable or disable verbose mode"""
    global VERBOSE_MODE
    VERBOSE_MODE = enabled

def configure_debug(debug=False, verbose=False):
    """Configure both debug and verbose modes at once"""
    global DEBUG_MODE, VERBOSE_MODE
    DEBUG_MODE = debug
    VERBOSE_MODE = verbose