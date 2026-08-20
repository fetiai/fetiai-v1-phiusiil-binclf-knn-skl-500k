VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PORT ?= 8000
IMAGE ?= fetiai-v1-phiusiil-binclf-knn-skl-500k:local

.DEFAULT_GOAL := help

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

$(VENV): ## Create the virtualenv
	python3 -m venv $(VENV)

install: $(VENV) ## Install pinned dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt -r requirements-dev.txt

selftest: ## Verify the artifact reproduces its recorded golden row
	$(PY) -m server.selftest --golden

namecheck: ## Fail if the parent project's internal library name appears anywhere
	$(PY) -m pytest tests/test_naming.py -q

test: ## Run the full offline suite
	$(PY) -m pytest -q

serve: ## Run the API on http://127.0.0.1:$(PORT)
	$(VENV)/bin/uvicorn server.app:app --host 127.0.0.1 --port $(PORT)

docker-build: ## Build the container image
	docker build -t $(IMAGE) .

docker-test: ## Run the golden-row self-test inside the built image
	docker run --rm $(IMAGE) python -m server.selftest --golden

clean: ## Remove caches
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

.PHONY: help install selftest namecheck test serve docker-build docker-test clean
