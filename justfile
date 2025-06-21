set dotenv-load

# ----------------
# default settings
# ----------------
# project
PROJECT_NAME := "svsv-llm"
EXAMPLE_DIR := "./examples"
LOGS_DIR := "./logs"
# python
PYTHON_EXEC := "uv run"
PYTHONVERSION := "3.12"
ENVNAME := "llm"
COV_FAIL_UNDER := "100"
APP := "src/svsvllm/__main__.py"
# docker
IMAGE := PROJECT_NAME


# -----------
# utilities
# -----------
init-directories:
	mkdir -p {{LOGS_DIR}}

pre-commit-install:
	{{PYTHON_EXEC}} pre-commit install


# -----------
# install project's dependencies
# -----------
install: pre-commit-install
	uv sync

lock:
	uv lock


# -----------
# testing
# -----------
init-tests: init-directories pre-commit-install

ruff:
	{{PYTHON_EXEC}} ruff check --fix .
	{{PYTHON_EXEC}} ruff format .

black-check:
	{{PYTHON_EXEC}} black --check src tests

black-fix:
	{{PYTHON_EXEC}} black src tests

mypy:
	{{PYTHON_EXEC}} mypy --cache-fine-grained tests
	{{PYTHON_EXEC}} mypy --cache-fine-grained src

pylint:
	{{PYTHON_EXEC}} pylint src

unit-test: init-tests
	{{PYTHON_EXEC}} pytest -m "not integtest" -x --testmon --junitxml=unit-tests.xml --cov=src/ --cov-fail-under {{COV_FAIL_UNDER}} --cov-report xml:unit-tests-cov.xml

integ-test: init-tests
	{{PYTHON_EXEC}} pytest -m "integtest" -x --testmon --junitxml=integ-tests.xml --cov=src/ --cov-report xml:integ-tests-cov.xml

nbmake: init-tests
	{{PYTHON_EXEC}} pytest --nbmake --overwrite {{EXAMPLE_DIR}}

test: ruff mypy unit-test nbmake

tests: test


# -----------
# Git
# -----------
# Run pre-commits manually
pre-commit:
	{{PYTHON_EXEC}} pre-commit run --all-files


# -----------
# UI
# -----------
ui:
	{{PYTHON_EXEC}} python -m svsv


# -----------
# Ollama
# -----------
ollama-clone:
	git clone https://github.com/ggerganov/llama.cpp

mlx-convert path="mlx-community/Mistral-7B-Instruct-v0.3":
	rm -rf mlx_model || echo "no dir"
	{{PYTHON_EXEC}} python -m mlx_lm convert --hf-path {{path}}
	mkdir -p "$(dirname ./llms/"{{path}}")"
	mv mlx_model ./llms/{{path}}
