SHELL := /bin/bash
COMPOSE ?= docker compose
CONFIG ?= config.yaml

.DEFAULT_GOAL := help

.PHONY: help hooks venv deps test lint audit run up up-vllm up-rocm up-cpu up-vulkan down logs pull models clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

hooks: ## Enable the gitmoji commit-msg hook
	git config core.hooksPath .githooks
	git config commit.template .gitmessage
	@echo "✅ Git hooks enabled (gitmoji)."

venv: ## Create a local venv for development
	python3 -m venv .venv && . .venv/bin/activate && pip install -U pip

deps: ## Install development dependencies (without the models)
	pip install -r requirements.txt -r requirements-dev.txt

test: ## Run the test suite
	pytest -q

lint: ## Check style (ruff, if available)
	ruff check app tests || true

audit: ## Audit dependencies for known vulnerabilities (pip-audit)
	pip-audit -r requirements.txt --no-deps --disable-pip || true

run: ## Run the server locally (dev)
	uvicorn app.main:app --host 0.0.0.0 --port $${PORT:-8004} --reload

up: ## Start the stack (CUDA)
	$(COMPOSE) up -d --build

up-vllm: ## Start the stack with the vLLM-Omni service (experimental streaming)
	$(COMPOSE) --profile vllm up -d --build

up-rocm: ## Start with the AMD ROCm override (non-NVIDIA)
	$(COMPOSE) -f docker-compose.yml -f docker-compose.rocm.yml up -d --build

up-cpu: ## Start with the CPU override (non-NVIDIA)
	$(COMPOSE) -f docker-compose.yml -f docker-compose.cpu.yml up -d --build

up-vulkan: ## Start with the Vulkan override (non-NVIDIA)
	$(COMPOSE) -f docker-compose.yml -f docker-compose.vulkan.yml up -d --build

down: ## Stop the stack
	$(COMPOSE) down

logs: ## Follow logs
	$(COMPOSE) logs -f --tail=100

pull: ## Pull official images (vLLM-Omni, etc.)
	$(COMPOSE) --profile vllm pull --ignore-buildable || true

models: ## Pre-download the TTS models into hf_cache
	./scripts/download_models.sh

clean: ## Remove local artifacts
	rm -rf outputs/* logs/* hf_cache/* __pycache__ .pytest_cache
