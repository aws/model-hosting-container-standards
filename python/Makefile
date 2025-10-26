.PHONY: help install format lint test clean all pre-commit-install pre-commit-run ci

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	poetry install --with dev

format:  ## Format code with black and isort
	poetry run black .
	poetry run isort .

lint:  ## Run all linters
	poetry run flake8 . --count --statistics
	#poetry run mypy model_hosting_container_standards/
	poetry run black --check .
	poetry run isort --check-only .

test:  ## Run tests
	poetry run pytest

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .mypy_cache/
	rm -rf .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

pre-commit-install:  ## Install pre-commit hooks
	poetry run pre-commit install

pre-commit-run:  ## Run pre-commit on all files
	poetry run pre-commit run --all-files

all: format lint test  ## Run format, lint, and test

ci: lint test  ## Run CI checks (lint and test without formatting)
