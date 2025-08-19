# Plan to Address PR Review Issues

## Summary
This document outlines the plan to fix critical showstoppers and potential issues identified in the PR review of the scikit-maad-biosound project.

## Phase 1: Critical Showstoppers (Must Fix)

### 1. Dependency Management with uv
**Issue**: No formal dependency management (missing pyproject.toml, requirements.txt, etc.)
**Fix**:
- Create `pyproject.toml` with proper project metadata
- List all required dependencies (numpy, pandas, matplotlib, scikit-maad, etc.)
- Add development dependencies for testing
- Configure uv workspace if needed

### 2. Fix Import Path Issues
**Issue**: Brittle `sys.path.insert()` calls in `main.py:44` and `core_processing.py:14`
**Fix**:
- Remove all `sys.path.insert()` calls from main.py and core_processing.py 
- Convert project to proper Python package structure
- Add `__init__.py` files where needed
- Use relative imports throughout the codebase
- Update main.py to use proper module imports

## Phase 2: Potential Issues (High Priority)

### 3. Decouple GUI Dependencies
**Issue**: `standalone_processor.py:23` imports from `gui.debug_config`, causing multiprocessing issues
**Fix**:
- Move debug/verbose printing to a shared utility module
- Remove gui.debug_config imports from processing modules
- Create standalone debug configuration that works with or without GUI
- Ensure multiprocessing works independently of GUI
Michelle's notes: just wanted to make it clear that we previously separated the GUI from the core processing code, because of potential issues with processing code calling GUI code during multiprocessing. This is an issue for mac with tkinter. Perhaps we could fix this by moving the debug code to a separate module, and then we can import that module in both places - please discuss with Michelle.

### 4. Add Input Validation
**Issue**: No validation of inputs, could cause crashes
**Fix**:
- Validate directory existence before processing
- Add bounds checking for frequency ranges (ensure low < high, positive values)
- Validate WAV file naming convention before parsing
- Add parameter validation for sensitivity/gain values

### 5. Improve Platform Compatibility
**Issue**: macOS-specific multiprocessing code in `core_processing.py:50-58`
**Fix**:
- Add proper platform detection and fallback handling
- Test multiprocessing behavior on different platforms
- Add error handling for unsupported multiprocessing methods
- Document platform-specific requirements

## Phase 3: Testing & Verification

### 6. Comprehensive Testing
**Purpose**: Validate all fixes work correctly
**Tasks**:
- Test installation process with clean uv environment
- Verify all import paths work correctly
- Test multiprocessing on different platforms
- Validate error handling with invalid inputs
- Ensure GUI still works after refactoring

## Implementation Order

1. **pyproject.toml** (enables proper installation)
2. **Import fixes** (prevents runtime failures)  
3. **GUI decoupling** (fixes multiprocessing reliability)
4. **Input validation** (prevents crashes from bad data)
5. **Platform compatibility** (ensures broad usability)
6. **Testing** (validates all fixes work)

## Priority Rationale

This plan prioritizes the fixes that prevent the software from working at all, then addresses reliability and robustness issues. Each phase builds on the previous one to ensure a stable, installable, and maintainable codebase.

## Files to Modify

- `pyproject.toml` (new)
- `main.py`
- `src/processing/core_processing.py`
- `src/processing/standalone_processor.py`
- `src/utils/` (new debug utility)
- Various `__init__.py` files
- Test files (for validation)