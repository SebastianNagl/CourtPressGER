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
DEFAULT_BATCH_SIZE = 20
DEFAULT_SAVE_INTERVAL = 5

# Standardwerte für Evaluation
DEFAULT_EVAL_DATASET = data/generation/mock_models_results.csv
DEFAULT_EVAL_OUTPUT_DIR = data/evaluation

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

## Run generation pipeline tests
.PHONY: test-generation
test-generation:
	pytest -v tests/test_generation_pipeline.py

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

## Generate press releases with all or a specific model
.PHONY: generate
generate:
	@if [ -z "$(MODEL)" ]; then \
		echo "Generiere Pressemitteilungen mit allen verfügbaren Modellen"; \
		python -m courtpressger.generation.cli \
			--dataset $(DEFAULT_OUTPUT) \
			--output-dir data/generation \
			$(if $(LIMIT),--limit $(LIMIT),); \
	else \
		echo "Generiere Pressemitteilungen mit dem spezifischen Modell: $(MODEL)"; \
		python -m courtpressger.generation.cli \
			--dataset $(DEFAULT_OUTPUT) \
			--output-dir data/generation \
			--model $(MODEL) \
			$(if $(LIMIT),--limit $(LIMIT),); \
	fi

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

## Führt alle verfügbaren Evaluationsmethoden für den Standard-Datensatz aus
.PHONY: eval
eval:
	@echo "Führe alle Evaluationsmethoden für den Datensatz $(DEFAULT_EVAL_DATASET) aus..."
	@mkdir -p $(DEFAULT_EVAL_OUTPUT_DIR)
	$(PYTHON_INTERPRETER) -m courtpressger.evaluation.cli \
		--dataset $(DEFAULT_EVAL_DATASET) \
		--output-dir $(DEFAULT_EVAL_OUTPUT_DIR) \
		--evaluate-existing-columns \
		--prompt-column synthetic_prompt \
		--ruling-column judgement \
		--press-column summary \
		--source-text-column judgement \
		--exclude-columns id date judgement subset_name split_name is_announcement_rule matching_criteria synthetic_prompt \
		--bert-score-model bert-base-multilingual-cased \
		--generate-report \
		--report-path reports/evaluation_report.html
	@echo "Evaluierungsergebnisse wurden im Verzeichnis $(DEFAULT_EVAL_OUTPUT_DIR) gespeichert."
	@echo "Ein HTML-Bericht wurde unter reports/evaluation_report.html erstellt."

eval-factual:
	@echo "Führe Evaluationsmethoden mit sachlicher Konsistenzprüfung für den Datensatz $(DEFAULT_EVAL_DATASET) aus..."
	@mkdir -p $(DEFAULT_EVAL_OUTPUT_DIR)
	$(PYTHON_INTERPRETER) -m courtpressger.evaluation.cli \
		--dataset $(DEFAULT_EVAL_DATASET) \
		--output-dir $(DEFAULT_EVAL_OUTPUT_DIR) \
		--evaluate-existing-columns \
		--prompt-column synthetic_prompt \
		--ruling-column judgement \
		--press-column summary \
		--source-text-column judgement \
		--exclude-columns id date judgement subset_name split_name is_announcement_rule matching_criteria synthetic_prompt \
		--bert-score-model bert-base-multilingual-cased \
		--enable-factual-consistency \
		--generate-report \
		--report-path reports/evaluation_factual_report.html
	@echo "Evaluierungsergebnisse wurden im Verzeichnis $(DEFAULT_EVAL_OUTPUT_DIR) gespeichert."
	@echo "Ein HTML-Bericht wurde unter reports/evaluation_factual_report.html erstellt."

## Synchronize the environment with dependencies
.PHONY: sync
sync:
	python -m courtpressger.synthetic_prompts.cli sync --help

## Download the German courts dataset
.PHONY: download_data
download_data:
	python -m courtpressger.synthetic_prompts.cli download --help

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
.PHONY: venv clean-venv

# Erstellt eine virtuelle Umgebung
venv:
	@echo "Erstelle virtuelle Umgebung..."
	@uv venv --python $(PYTHON_VERSION) .venv
	@echo ">>> Virtuelle Umgebung erstellt. Wichtig: Aktiviere die Umgebung mit:"
	@echo ">>> source .venv/bin/activate"
	@echo ">>> Erst nach der Aktivierung können Make-Befehle ausgeführt werden."

# Löscht die virtuelle Umgebung
clean-venv:
	@echo "Lösche virtuelle Umgebung..."
	@rm -rf .venv

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

## Display help information
.PHONY: help
help:
	@echo "Available commands:"
	@grep -E '^##' $(MAKEFILE_LIST) | grep -v "^## -----" | sed -e 's/## //g' | sort

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