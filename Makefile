#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = CourtPressGER
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python3
VENV_NAME = .venv
PYTHON = python3
SHELL := /bin/bash

# Standardwerte für synthetische Prompt-Generierung
DEFAULT_MODEL = claude-3-7-sonnet-20250219
DEFAULT_INPUT = data/interim/cleaned.csv
DEFAULT_OUTPUT = data/processed/cases_prs_synth_prompts.csv
DEFAULT_CHECKPOINT_DIR = data/checkpoints
DEFAULT_BATCH_SIZE = 10      # Batchgröße standardmäßig 10
DEFAULT_SAVE_INTERVAL = 1    # Speichere nach jedem Batch

#################################################################################
# HINWEIS                                                                       #
#################################################################################
# Aktiviere vor der Nutzung der Make-Befehle eine virtuelle Umgebung mit:       #
# source .venv-cpu/bin/activate  ODER  source .venv-gpu/bin/activate            #
#################################################################################

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
	ruff check courtpressger
	mypy courtpressger

## Format source code with ruff
.PHONY: format
format:
	ruff format courtpressger

## Run tests
.PHONY: test
test:
	pytest -v tests/

## Run only CSV-related tests
.PHONY: test-csv
test-csv:
	pytest -v tests/test_cleaner.py tests/test_csv_validator.py

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
synthetic:
	@mkdir -p $(DEFAULT_CHECKPOINT_DIR)
	python -m courtpressger.synthetic_prompts.cli generate \
		--input $(DEFAULT_INPUT) \
		--output $(DEFAULT_OUTPUT) \
		--checkpoint-dir $(DEFAULT_CHECKPOINT_DIR) \
		--batch-size $(DEFAULT_BATCH_SIZE) \
		--save-interval $(DEFAULT_SAVE_INTERVAL)

## Resume synthetic prompt generation from last checkpoint
.PHONY: synthetic-resume
synthetic-resume:
	@if [ -d "$(DEFAULT_CHECKPOINT_DIR)" ]; then \
		checkpoints=$$(ls -v1 $(DEFAULT_CHECKPOINT_DIR)/cases_prs_synth_prompts_*.csv 2>/dev/null || echo ""); \
		if [ -n "$$checkpoints" ]; then \
			last_checkpoint=$$(echo "$$checkpoints" | tail -n1); \
			last_index=$$(basename "$$last_checkpoint" .csv | grep -oE '[0-9]+$$'); \
			echo "Letzter Checkpoint gefunden: $$last_checkpoint (Index: $$last_index)"; \
			python -m courtpressger.synthetic_prompts.cli generate \
				--input $(DEFAULT_INPUT) \
				--output $(DEFAULT_OUTPUT) \
				--checkpoint-dir $(DEFAULT_CHECKPOINT_DIR) \
				--batch-size $(DEFAULT_BATCH_SIZE) \
				--save-interval $(DEFAULT_SAVE_INTERVAL) \
				--start-idx $$last_index; \
		else \
			echo "Keine Checkpoints gefunden in $(DEFAULT_CHECKPOINT_DIR). Starte neu."; \
			$(MAKE) synthetic; \
		fi \
	else \
		echo "Checkpoint-Verzeichnis $(DEFAULT_CHECKPOINT_DIR) nicht gefunden. Starte neu."; \
		$(MAKE) synthetic; \
	fi

## Synchronize the environment with dependencies
.PHONY: sync
sync:
	python -m courtpressger.synthetic_prompts.cli sync --help

## Download the German courts dataset
.PHONY: download_data
download_data:
	python -m courtpressger.synthetic_prompts.cli download --help

## Validate CSV checkpoints
.PHONY: validate-csv
validate-csv:
	python -m courtpressger.synthetic_prompts.cli validate --help

## Fix format errors in a CSV file (use FILE=path/to/file.csv)
.PHONY: fix-csv
fix-csv:
	@if [ -z "$(FILE)" ]; then \
		echo "Bitte den Dateipfad angeben, z.B. 'make fix-csv FILE=checkpoints/file.csv'"; \
	else \
		python -m courtpressger.synthetic_prompts.cli fix --file $(FILE); \
	fi

## Repair damaged CSV structure (use FILE=path/to/file.csv)
.PHONY: repair-csv
repair-csv:
	@if [ -z "$(FILE)" ]; then \
		echo "Bitte den Dateipfad angeben, z.B. 'make repair-csv FILE=checkpoints/file.csv'"; \
	else \
		python -m courtpressger.synthetic_prompts.cli repair --file $(FILE); \
	fi

## Clean all checkpoints from API errors
.PHONY: clean-checkpoints
clean-checkpoints:
	python -m courtpressger.synthetic_prompts.cli clean --help

## Sanitize API responses in checkpoints
.PHONY: sanitize-csv
sanitize-csv:
	python -m courtpressger.synthetic_prompts.cli sanitize --help

## Clean CSV data (use FILE=path/to/file.csv)
.PHONY: clean-csv
clean-csv:
	@if [ -z "$(FILE)" ]; then \
		echo "Bitte den Dateipfad angeben, z.B. 'make clean-csv FILE=checkpoints/file.csv'"; \
	else \
		python -m courtpressger.synthetic_prompts.cli clean-csv --file $(FILE); \
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
	@echo ">>> CPU-Umgebung erstellt. Wichtig: Aktiviere die Umgebung mit:"
	@echo ">>> source .venv-cpu/bin/activate"
	@echo ">>> Erst nach der Aktivierung können Make-Befehle ausgeführt werden."

# Erstellt eine GPU-spezifische virtuelle Umgebung
venv-gpu:
	@echo "Erstelle GPU-spezifische virtuelle Umgebung..."
	@uv venv --python $(PYTHON_VERSION) .venv-gpu
	@. .venv-gpu/bin/activate && \
		uv pip install --upgrade pip && \
		uv pip install torch --index-url https://download.pytorch.org/whl/cu118 && \
		uv pip install -e ".[gpu]"
	@echo ">>> GPU-Umgebung erstellt. Wichtig: Aktiviere die Umgebung mit:"
	@echo ">>> source .venv-gpu/bin/activate"
	@echo ">>> Erst nach der Aktivierung können Make-Befehle ausgeführt werden."

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
	pip install -e .

dev:
	pip install -e ".[dev]"

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