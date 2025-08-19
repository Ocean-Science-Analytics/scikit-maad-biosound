# Quick Start for Existing Users

## One-time setup (only do this once per computer)

1. **Update to latest code:**
   ```bash
   cd scikit-maad-biosound
   git pull
   ```

2. **Setup everything:**

   **macOS/Linux or Git Bash on Windows:**
   ```bash
   make setup
   ```
   
   **Windows Command Prompt/PowerShell:**
   ```bash
   # Install uv if you don't have it:
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # Then install dependencies:
   uv pip install -e .
   ```
   
   This installs uv (if you don't have it), creates a virtual environment, and installs all dependencies. You only need to do this once on your computer.

## Every time you want to run the GUI

**macOS/Linux or Git Bash on Windows:**
```bash
make gui
```

**Windows Command Prompt/PowerShell:**
```bash
python main.py
```

## If you want to update to the latest version

```bash
git pull
make install  # only if there are new dependencies (or "uv pip install -e ." on Windows)
make gui      # (or "python main.py" on Windows)
```

## Alternative: Run without make commands

If you prefer to run directly (after the one-time setup above):

```bash
python main.py
```

This works from command line or any IDE (VS Code, PyCharm, etc.) - just open main.py and click run.

---

The full README has more details about all the features and advanced usage, but these are all the commands you need to get started!