#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = CourtPressGER
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python3
VENV_NAME = .venv
VENV_ACTIVATE = $(VENV_NAME)/bin/activate
PYTHON = $(VENV_NAME)/bin/python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync

## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	source $(VENV_ACTIVATE) && ruff check courtpressger
	source $(VENV_ACTIVATE) && mypy courtpressger

## Format source code with ruff
.PHONY: format
format:
	source $(VENV_ACTIVATE) && ruff format courtpressger

## Run tests
.PHONY: test
test:
	source $(VENV_ACTIVATE) && pytest -v tests/

## Run data cleaning pipeline
.PHONY: clean_data
clean_data: requirements
	$(PYTHON_INTERPRETER) notebooks/bereinigung.ipynb

## Generate descriptive statistics
.PHONY: descriptive
descriptive: requirements
	$(PYTHON_INTERPRETER) notebooks/deskriptiv.ipynb

## Generate synthetic prompts
.PHONY: synthetic
synthetic: requirements
	$(PYTHON_INTERPRETER) notebooks/synthetic_prompts.ipynb

## Synchronize the environment with dependencies
.PHONY: sync
sync:
	source $(VENV_ACTIVATE) && uv sync

## Download the German courts dataset
.PHONY: download_data
download_data:
	source $(VENV_ACTIVATE) && $(PYTHON) -m courtpressger.main --download

## Remove Python file artifacts
.PHONY: clean
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[co]" -delete
	find . -type f -name "*.so" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +

# Virtuelle Umgebungen
.PHONY: venv venv-cpu venv-gpu clean-venv

# Erstellt eine CPU-spezifische virtuelle Umgebung
venv-cpu:
	@echo "Erstelle CPU-spezifische virtuelle Umgebung..."
	@uv venv --python $(PYTHON_VERSION) .venv-cpu
	@. .venv-cpu/bin/activate && \
		uv pip install --upgrade pip && \
		uv pip install -e ".[cpu]"
	@echo ">>> CPU-Umgebung erstellt. Aktivieren mit:"
	@echo ">>> source .venv-cpu/bin/activate"

# Erstellt eine GPU-spezifische virtuelle Umgebung
venv-gpu:
	@echo "Erstelle GPU-spezifische virtuelle Umgebung..."
	@uv venv --python $(PYTHON_VERSION) .venv-gpu
	@. .venv-gpu/bin/activate && \
		uv pip install --upgrade pip && \
		uv pip install torch --index-url https://download.pytorch.org/whl/cu118 && \
		uv pip install -e ".[gpu]"
	@echo ">>> GPU-Umgebung erstellt. Aktivieren mit:"
	@echo ">>> source .venv-gpu/bin/activate"

# Standard-Target für venv (CPU als Standard)
venv: venv-cpu

# Löscht alle virtuellen Umgebungen
clean-venv:
	@echo "Lösche virtuelle Umgebungen..."
	@rm -rf .venv-cpu .venv-gpu

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

## Display help information
.PHONY: help
help:
	@echo "Available commands:"
	@grep -E '^## [a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' 