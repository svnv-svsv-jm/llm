help:
	@cat Makefile

.EXPORT_ALL_VARIABLES:

# create an .env file to override the default settings
-include .env
export $(shell sed 's/=.*//' .env)


.PHONY: build docs

# ----------------
# default settings
# ----------------
# user
LOCAL_USER:=$(shell whoami)
LOCAL_USER_ID:=$(shell id -u)
# project
PROJECT_NAME:=svsv-llm
EXAMPLE_DIR:=./examples
LOGS_DIR:=./logs
# python
PYTHON?=python
PYTHON_EXEC?=python -m
PYTHONVERSION?=3.12
ENVNAME?=llm
PYTEST?=pytest
PYTEST_PLUGINS=
SYSTEM=$(shell python -c "import sys; print(sys.platform)")
COV_FAIL_UNDER=99
APP=src/svsvllm/__main__.py
# poetry
POETRY=poetry
# docker
DOCKER?=docker
DOCKERFILE?=Dockerfile
MEMORY=8g
SHM=4g
CPUS=1
DOCKER_COMMON_FLAGS=--cpus=$(CPUS) --memory=$(MEMORY) --shm-size=$(SHM) --network=host
REGISTRY=registry.gitlab.com/svnv-svsv-jm/chess-rl
IMAGE=$(PROJECT_NAME)
IMAGE_PYTHON=/venv/bin/python


# -----------
# utilities
# -----------
init-directories:
	mkdir -p $(LOGS_DIR)


# -----------
# install project's dependencies
# -----------
create-env:
	pyenv virtualenv $(PYTHONVERSION) $(ENVNAME) || echo "$(ENVNAME) already exists"

activate-env: create-env
	pyenv shell $(ENVNAME)

install-init:
	$(PYTHON_EXEC) pip install --upgrade pip
	$(PYTHON_EXEC) pip install --upgrade poetry poetry-plugin-export
	$(PYTHON_EXEC) poetry self update

install:
	$(PYTHON_EXEC) poetry install --no-cache

lock: install-init
	$(PYTHON_EXEC) poetry lock


# -----------
# testing
# -----------
init-tests: init-directories

mypy: init-tests
	$(PYTHON_EXEC) mypy --cache-fine-grained tests

unit-test: init-tests
	$(PYTHON_EXEC) pytest -m "not integtest" --pylint $(PYTEST_PLUGINS) --junitxml=pytest-results.xml --cov=src/ --cov-fail-under $(COV_FAIL_UNDER)

integ-test: init-tests
	$(PYTHON_EXEC) pytest -m "integtest" --pylint $(PYTEST_PLUGINS) --junitxml=pytest-results.xml --cov=src/

nbmake: init-tests
	$(PYTHON_EXEC) pytest --nbmake --overwrite "$(EXAMPLE_DIR)"

test: mypy unit-test nbmake

tests: test


# -----------
# UI
# -----------
ui: LOG_LEVEL=TRACE
ui:
	LOG_LEVEL=$(LOG_LEVEL) python -m svsvchat


# -----------
# git
# -----------
# Locally delete branches that have been merged
git-clean:
	bash scripts/git-clean.sh

# Squash all commits before rebasing, see https://stackoverflow.com/questions/25356810/git-how-to-squash-all-commits-on-branch
git-squash:
	git reset $(git merge-base main $(git branch --show-current))
	git add -A
	git commit -m "squashed commit"

# Merge to `develop` and delete
git-merge: CURRENT_BRANCH=$(shell git branch --show-current)
git-merge:
	git checkout develop
	git pull
	git merge $(CURRENT_BRANCH)
	git branch -D $(CURRENT_BRANCH)


# ---------------
# remote
# ---------------
# scp -r $(OBJECT) $(REMOTE_HOST):$(REMOTE_PATH_TO_PROJECT)/$(OBJECT)
sync-object: OBJECT=
sync-object:
	rsync -av $(OBJECT) $(REMOTE_HOST):$(REMOTE_PATH_TO_PROJECT)/. --exclude=__pycache__ --exclude=.coverage --exclude=.models --exclude=.pytest_cache

# Sync code
sync-remote:
	$(MAKE) sync-object OBJECT=res
	$(MAKE) sync-object OBJECT=scripts
	$(MAKE) sync-object OBJECT=src
	$(MAKE) sync-object OBJECT=tests
	$(MAKE) sync-object OBJECT=Makefile
	$(MAKE) sync-object OBJECT=poetry.lock
	$(MAKE) sync-object OBJECT=pyproject.toml
	$(MAKE) sync-object OBJECT=tutorials
	$(MAKE) sync-object OBJECT=sandbox

# Run
run-remote: CMD=make tests
run-remote: sync-remote
	ssh $(REMOTE_HOST) -t bash -c "cd $(REMOTE_PATH_TO_PROJECT); make activate-env; make install; $(CMD)"


# -----------
# docker
# -----------
build: TAG=latest
build:
	$(DOCKER) build -t $(REGISTRY)/$(IMAGE):$(TAG) --network=host --build-arg PROJECT_NAME=$(PROJECT_NAME) -f $(DOCKERFILE) .

buildx: TAG=latest
buildx: PLATFORM=linux/amd64
buildx: LOAD_PUSH=--load
buildx:
	$(DOCKER) buildx build --platform $(PLATFORM) $(LOAD_PUSH) --network=host -t $(REGISTRY)/$(IMAGE):$(TAG) --build-arg PROJECT_NAME=$(PROJECT_NAME) -f $(DOCKERFILE) .

buildx-multi: PLATFORM=linux/amd64,linux/arm64
buildx-multi: LOAD_PUSH=--push
buildx-multi: build

push:
	echo $(CI_JOB_TOKEN) | $(DOCKER) login -u $(LOCAL_USER) $(REGISTRY) --password-stdin
	$(DOCKER) image push $(REGISTRY)/$(PROJECT_NAME):latest

bash: CONTAINER_NAME=bash
bash:
	$(DOCKER) run --rm -it $(DOCKER_COMMON_FLAGS) \
		--name $(PROJECT_NAME)-$(CONTAINER_NAME) \
		-t $(REGISTRY)/$(PROJECT_NAME) \
		bash

up: docker-compose.yml
	docker-compose $(DOCKER_COMPOSE_CONTEXT) -p $(PROJECT_NAME) up -d $(SERVICES) $(RECREATE)

up-force: RECREATE=--build --force-recreate
up-force: up

up-remote: DOCKER_COMPOSE_CONTEXT=--context remote
up-remote: up

down: docker-compose.yml
	$(DOCKER)-compose -p $(PROJECT_NAME) down --volumes


# -----------------------------------------
# experiments
# -----------------------------------------
run:
	supervisord -c supervisord.conf

# [DO NOT CALL THIS COMMAND]
exp-base: RANDOM=$(shell bash -c 'echo $$RANDOM')
exp-base: CONTAINER_NAME=exp-$(CONFIG)-$(RANDOM)
exp-base: SCRIPT=experiments/main.py
exp-base: PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
exp-base: DOCKER_FLAGS=-d
exp-base: NOW=$(shell date '+%Y-%m-%d_%H:%M:%S')
exp-base: CMD=$(IMAGE_PYTHON) -u /workdir/$(SCRIPT) --config-name $(CONFIG) $(OVERRIDE)
exp-base: WORKDIR=$(PWD)
exp-base:
	$(DOCKER) run --rm -it $(DOCKER_FLAGS) $(DOCKER_COMMON_FLAGS) \
		--volume $(WORKDIR):/workdir \
		$(GPU_FLAGS) \
		--name $(PROJECT_NAME)-$(CONTAINER_NAME) \
		-t $(REGISTRY)/$(PROJECT_NAME) \
		$(CMD)

# Run experiment
exp: CONFIG=main.yaml
exp: exp-base

# Run experiment with NVIDIA runtime
# exp-gpu: GPU_FLAGS=--runtime=nvidia --gpus all
exp-gpu: GPU_FLAGS=--gpus all
exp-gpu: CONFIG=main.yaml
exp-gpu: exp-base


# ---------------
# HuggingFace
# ---------------
# Clean cache
hf-clean-cache:
	rm -r ~/.cache/huggingface/hub


# ---------------
# remote
# ---------------
# Sync code
sync-code:
	scp -r . $(REMOTE_HOST):$(REMOTE_PATH_TO_PROJECT)/.

# Get lightning_logs from remote machine
sync-exp-logs:
	scp -r $(REMOTE_HOST):$(REMOTE_PATH_TO_PROJECT)/lightning_logs ./lightning_logs
cp-remote-logs: sync-exp-logs


# ----------
# llama.ccp
# ----------
get-llama.ccp:
	git clone git@github.com:ggerganov/llama.cpp.git

# necessary to download gated models
huggingface-login:
	huggingface-cli login --token ${HUGGINGFACE_TOKEN}

# Some useful names:
# - mistralai/Mistral-7B-Instruct-v0.3
# - meta-llama/Meta-Llama-3-8B-Instruct
llama.ccp-download-model: huggingface-login
llama.ccp-download-model: MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3
llama.ccp-download-model:
	mkdir -p models/$(MODEL_NAME)
	huggingface-cli download $(MODEL_NAME) --local-dir models/$(MODEL_NAME)


# --------
# mlx-lm
# --------
mlx-convert: MODEL_PATH=mistralai/Mistral-7B-Instruct-v0.3
mlx-convert:
	python -m mlx_lm.convert --hf-path $(MODEL_PATH) -q
