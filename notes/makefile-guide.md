# Makefile Guide

## What is a Makefile?

A **Makefile** is a configuration file used with the `make` command to automate building, testing, and managing software projects. Originally created for compiling C/C++ programs, Makefiles are now widely used across many programming languages including Python.

### Key Concepts

- **Targets**: Named tasks you want to run (like `test`, `lint`, `clean`)
- **Commands**: Shell commands that execute when a target is run
- **Dependencies**: Other targets that must run first
- **Variables**: Reusable values throughout the Makefile

### Basic Syntax
```makefile
target: dependencies
	command1
	command2
```

**Important**: Commands must be indented with **tabs**, not spaces.

## Our Project's Makefile

Our Makefile provides convenient shortcuts for common development tasks. Here's what each target does:

### Available Commands

Run `make help` to see all available targets:

```bash
make help
```

### Core Development Commands

#### `make test`
Runs the comprehensive test suite:
```bash
make test
# Equivalent to: python tests/test_suite.py
```

#### `make test-performance`
Runs performance benchmarking:
```bash
make test-performance
# Equivalent to: python performance_test.py test_wav_files/ performance_output/
```

#### `make install`
Installs the package in development mode:
```bash
make install
# Equivalent to: 
# uv pip install -e .
# uv pip install -e ".[dev]"
```

### Code Quality Commands

#### `make lint`
Checks code for style and quality issues:
```bash
make lint
# Equivalent to: ruff check src/ tests/ scripts/ --statistics
```

#### `make format-check`
Checks if code needs formatting (without changing files):
```bash
make format-check
# Equivalent to: ruff format --check src/ tests/ scripts/
```

#### `make format`
Automatically formats code:
```bash
make format
# Equivalent to: ruff format src/ tests/ scripts/
```

#### `make fix`
Auto-fixes linting issues AND formats code:
```bash
make fix
# Equivalent to:
# ruff check src/ tests/ scripts/ --fix
# ruff format src/ tests/ scripts/
```

#### `make fix-unsafe`
Fixes all issues including potentially unsafe changes:
```bash
make fix-unsafe
# Equivalent to:
# ruff check src/ tests/ scripts/ --fix --unsafe-fixes
# ruff format src/ tests/ scripts/
```

### Utility Commands

#### `make clean`
Removes build artifacts and cache files:
```bash
make clean
# Removes: __pycache__, *.pyc, *.pyo, build/, dist/, etc.
```

#### `make all-checks`
Runs the full quality pipeline:
```bash
make all-checks
# Runs: lint, format-check, and test
```

#### `make mypy`
Runs type checking:
```bash
make mypy
# Equivalent to: mypy src/
```

### Legacy Tool Commands (for comparison)

#### `make black-check` / `make flake8`
Included for comparing with the older tools we replaced with Ruff:
```bash
make black-check  # Check formatting with Black
make flake8       # Check with flake8
```

## How to Use Our Makefile

### Daily Development Workflow

1. **Before starting work**:
   ```bash
   make test           # Ensure everything works
   ```

2. **During development**:
   ```bash
   make lint           # Check for issues
   make format         # Format your code
   ```

3. **Before committing**:
   ```bash
   make all-checks     # Run everything
   make fix           # Auto-fix any issues
   ```

4. **When cleaning up**:
   ```bash
   make clean         # Remove temporary files
   ```

### Example Session

```bash
# Check current state
make lint

# Fix formatting and linting issues
make fix

# Run tests to ensure nothing broke
make test

# Final check before commit
make all-checks
```

## Advantages of Using Make

1. **Consistency**: Everyone on the team runs commands the same way
2. **Documentation**: The Makefile serves as living documentation of common tasks
3. **Simplicity**: Short, memorable commands instead of long CLI commands
4. **Automation**: Can chain multiple commands together
5. **Cross-platform**: Works on Linux, macOS, and Windows (with make installed)

## Technical Details

### The `.PHONY` Declaration
```makefile
.PHONY: help install test lint format fix clean
```
This tells make that these targets don't create files with the same names.

### Variable Usage
We could add variables like:
```makefile
SRC_DIR = src/
TEST_DIR = tests/
PYTHON = python
```

### Command Chaining
Our `all-checks` target runs multiple commands:
```makefile
all-checks: lint format-check test
	@echo "All checks completed!"
```

The dependencies (`lint format-check test`) run first, then the echo command.

## When to Update the Makefile

Add new targets when you:
- Create new test types
- Add new development tools
- Want to automate common tasks
- Need complex command combinations

The Makefile makes our development workflow more efficient and helps ensure consistent code quality across the project.