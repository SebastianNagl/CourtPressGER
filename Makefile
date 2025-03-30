#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = CourtPressGER
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python3
VENV_NAME = .venv
VENV_ACTIVATE = $(VENV_NAME)/bin/activate
PYTHON = $(VENV_NAME)/bin/python
SHELL := /bin/bash

# Standardwerte für synthetische Prompt-Generierung
DEFAULT_MODEL = claude-3-7-sonnet-20250219
DEFAULT_INPUT = data/interim/cleaned.csv
DEFAULT_OUTPUT = data/processed/cases_prs_synth_prompts.csv
DEFAULT_CHECKPOINT_DIR = data/checkpoints
DEFAULT_SAVE_INTERVAL = 1  # Speichere nach jedem Batch

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Make sure uv is installed first: curl -LsSf https://astral.sh/uv/install.sh | sh

## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync

## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	. $(VENV_ACTIVATE) && ruff check courtpressger
	. $(VENV_ACTIVATE) && mypy courtpressger

## Format source code with ruff
.PHONY: format
format:
	. $(VENV_ACTIVATE) && ruff format courtpressger

## Run tests
.PHONY: test
test:
	. $(VENV_ACTIVATE) && pytest -v tests/

## Run only CSV-related tests
.PHONY: test-csv
test-csv:
	. $(VENV_ACTIVATE) && pytest -v tests/test_cleaner.py tests/test_csv_validator.py

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
	. $(VENV_ACTIVATE) && python -m courtpressger.synthetic_prompts.cli sync --help

## Download the German courts dataset
.PHONY: download_data
download_data:
	. $(VENV_ACTIVATE) && python -m courtpressger.synthetic_prompts.cli download --help

## Validate CSV checkpoints
.PHONY: validate-csv
validate-csv:
	. $(VENV_ACTIVATE) && python -m courtpressger.synthetic_prompts.cli validate --help

## Fix format errors in a CSV file (use FILE=path/to/file.csv)
.PHONY: fix-csv
fix-csv:
	@if [ -z "$(FILE)" ]; then \
		echo "Bitte den Dateipfad angeben, z.B. 'make fix-csv FILE=checkpoints/file.csv'"; \
	else \
		. $(VENV_ACTIVATE) && $(PYTHON) -m courtpressger.synthetic_prompts.cli fix --file $(FILE); \
	fi

## Repair damaged CSV structure (use FILE=path/to/file.csv)
.PHONY: repair-csv
repair-csv:
	@if [ -z "$(FILE)" ]; then \
		echo "Bitte den Dateipfad angeben, z.B. 'make repair-csv FILE=checkpoints/file.csv'"; \
	else \
		. $(VENV_ACTIVATE) && $(PYTHON) -m courtpressger.synthetic_prompts.cli repair --file $(FILE); \
	fi

## Clean all checkpoints from API errors
.PHONY: clean-checkpoints
clean-checkpoints:
	. $(VENV_ACTIVATE) && python -m courtpressger.synthetic_prompts.cli clean --help

## Sanitize API responses in checkpoints
.PHONY: sanitize-csv
sanitize-csv:
	. $(VENV_ACTIVATE) && python -m courtpressger.synthetic_prompts.cli sanitize --help

## Clean CSV data (use FILE=path/to/file.csv)
.PHONY: clean-csv
clean-csv:
	@if [ -z "$(FILE)" ]; then \
		echo "Bitte den Dateipfad angeben, z.B. 'make clean-csv FILE=checkpoints/file.csv'"; \
	else \
		. $(VENV_ACTIVATE) && $(PYTHON) -m courtpressger.synthetic_prompts.cli clean-csv --file $(FILE); \
	fi

## Führe alle CSV-bezogenen Befehle aus
.PHONY: all-csv
all-csv: validate-csv fix-csv repair-csv clean-checkpoints sanitize-csv clean-csv

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
	@grep -E '^##' $(MAKEFILE_LIST) | grep -v "^## -----" | sed -e 's/## //g' | sort

.PHONY: install dev all-csv

install:
	$(PYTHON) -m pip install -e .

dev:
	$(PYTHON) -m pip install -e ".[dev]"

all-csv: validate-csv fix-csv repair-csv clean-checkpoints sanitize-csv clean-csv

## Remove Python file artifacts
.PHONY: clean-pyc
clean-pyc:
	find . -name '*.pyc' -delete
	find . -name '*.pyo' -delete
	find . -name '*~' -delete
	find . -name '__pycache__' -type d -delete

## Remove test and coverage artifacts
.PHONY: clean-test
clean-test:
	rm -f .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache

## Clean all artifacts
.PHONY: clean
clean: clean-pyc clean-test

## Display help message
.PHONY: help
help:
	@echo "Available commands:"
	@grep -E '^##' $(MAKEFILE_LIST) | grep -v "^## -----" | sed -e 's/## //g' | sort 