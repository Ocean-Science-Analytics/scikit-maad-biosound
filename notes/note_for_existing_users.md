# Quick Start for existing users

## One-time setup (only do this once per computer)

1. **Get the code:**
   ```bash
   git clone [repository-url]
   cd scikit-maad-biosound
   ```

2. **Setup everything:**
   ```bash
   make setup
   ```
   
   This installs uv (if you don't have it), creates a virtual environment, and installs all dependencies. You only need to do this once on your computer.

## Every time you want to run the GUI

```bash
make gui
```

That's it!

## If you want to update to the latest version

```bash
git pull
make install  # only if there are new dependencies
make gui
```

## Alternative: Run without make commands

If you prefer to run directly (after the one-time setup above):

```bash
python main.py
```

This works from command line or any IDE (VS Code, PyCharm, etc.) - just open main.py and click run.

---

The full README has more details about all the features and advanced usage, but these 3 commands are all you need to get started!