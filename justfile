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


# -----------
# install project's dependencies
# -----------
install:
	uv sync

lock:
	uv lock


# -----------
# testing
# -----------
init-tests: init-directories

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

test: pylint mypy unit-test nbmake

tests: test


# -----------
# UI
# -----------
ui:
	uv run python -m svsvllm