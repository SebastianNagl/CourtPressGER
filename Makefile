#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = CourtPressGER
PYTHON_VERSION = 3.12


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

## Synchronize the environment with dependencies
.PHONY: sync
sync:
	uv sync

## Download the German courts dataset
.PHONY: download_data
download_data:
	python -m courtpressger.main --download

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