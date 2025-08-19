.PHONY: help setup install test lint format fix clean

help:  ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

setup:  ## Complete first-time setup (installs uv if needed, creates venv, installs dependencies)
	@echo "🔧 Setting up scikit-maad-biosound..."
	@command -v uv >/dev/null 2>&1 || { echo "Installing uv..."; curl -LsSf https://astral.sh/uv/install.sh | sh; }
	@echo "Creating virtual environment..."
	uv venv
	@echo "Installing project and dependencies..."
	uv pip install -e .
	uv pip install -e ".[dev]"
	@echo "✅ Setup complete! You can now run: make gui"

install:  ## Install the package in development mode
	uv pip install -e .
	uv pip install -e ".[dev]"

test:  ## Run all tests
	python tests/test_suite.py

test-performance:  ## Run performance tests (requires test data)
	python scripts/performance_test.py test_wav_files/ performance_output/

test-performance-fresh:  ## Run performance tests with fresh sample data
	rm -rf test_wav_files/
	python scripts/generate_samples.py -o test_wav_files
	python scripts/performance_test.py test_wav_files/ performance_output/

generate-samples:  ## Generate sample marine acoustic WAV files for testing
	python scripts/generate_samples.py

gui:  ## Launch the marine acoustic analysis GUI
	scikit-maad-gui

validate:  ## Validate calculations against scikit-maad
	python scripts/validate_calculations.py

lint:  ## Check code with Ruff linter
	ruff check src/ tests/ scripts/ --statistics

format-check:  ## Check if code needs formatting
	ruff format --check src/ tests/ scripts/

format:  ## Format code with Ruff formatter
	ruff format src/ tests/ scripts/

fix:  ## Auto-fix linting issues and format code
	ruff check src/ tests/ scripts/ --fix
	ruff format src/ tests/ scripts/

fix-unsafe:  ## Auto-fix all issues including unsafe fixes
	ruff check src/ tests/ scripts/ --fix --unsafe-fixes
	ruff format src/ tests/ scripts/

clean:  ## Clean build artifacts and cache files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/

mypy:  ## Run mypy type checking
	mypy src/

black-check:  ## Check formatting with Black (if you want to compare)
	black --check src/ tests/ scripts/

flake8:  ## Run flake8 (if you want to compare)
	flake8 src/ tests/ scripts/

all-checks: lint format-check test  ## Run all checks (lint, format, tests)
	@echo "All checks completed!"