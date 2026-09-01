PYTHON ?= python
VENV   := .venv
BIN    := $(VENV)/bin
ifeq ($(OS),Windows_NT)
BIN    := $(VENV)/Scripts
endif
PY     := $(BIN)/python

DATA   := data/tinyshakespeare.txt
ART    := artifacts
TARGET := $(ART)/target.pt
DRAFT  := $(ART)/draft.pt

.DEFAULT_GOAL := help
.PHONY: help setup data train train-target train-draft verify benchmark report \
        test lint typecheck check clean clean-artifacts all

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the virtualenv and install everything (CPU-only torch)
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -e ".[dev,plots]"

data: $(DATA)  ## Download TinyShakespeare

$(DATA):
	@mkdir -p data
	curl -sSfL -o $(DATA) https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

train: train-target train-draft  ## Train both models

train-target: $(DATA)  ## Train the target model (~25 min on one CPU core)
	$(PY) scripts/train.py --preset target --steps 6000

train-draft: $(DATA)  ## Train the draft model (~7 min on one CPU core)
	$(PY) scripts/train.py --preset draft --steps 6000

verify: $(TARGET) $(DRAFT)  ## Prove speculative decoding preserves the distribution
	$(PY) scripts/verify_speculative.py --trials 5000 --gammas 1 2 4 7

benchmark: $(TARGET)  ## Run every benchmark sweep
	$(PY) scripts/benchmark.py --requests 64 --rates 4 16 64 256 --max-batch-size 16

report:  ## Regenerate plots and RESULTS.md from the JSON in artifacts/
	$(PY) scripts/report.py

test:  ## Run the test suite
	$(PY) -m pytest

lint:  ## Lint with ruff
	$(PY) -m ruff check src tests scripts

typecheck:  ## Type-check with mypy (strict)
	$(PY) -m mypy

check: lint typecheck test  ## Everything CI runs

all: data train verify benchmark report check  ## Reproduce the whole project

clean:  ## Remove caches and build products
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

clean-artifacts:  ## Remove trained models, benchmarks, and plots
	rm -rf $(ART)/*.pt $(ART)/*.json $(ART)/benchmark $(ART)/plots
