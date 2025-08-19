#!/usr/bin/env python3
"""
Debug configuration module for controlling output verbosity.

DEPRECATED: This module now imports from utils.debug_utils for compatibility.
New code should import directly from utils.debug_utils.

Global flags that can be set by main.py and used throughout the application:
- DEBUG_MODE: Controls [DEBUG] messages for troubleshooting multiprocessing, data flow, etc.
- VERBOSE_MODE: Controls [STANDALONE] and other verbose processing messages
"""

# Import from shared utilities for backward compatibility
