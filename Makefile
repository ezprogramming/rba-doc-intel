# Makefile for RBA Document Intelligence Platform
# Production-ready task automation

# Default target
.DEFAULT_GOAL := help

# Environment configuration (can be overridden)
COMPOSE ?= docker compose
APP_SERVICE ?= app
ARGS ?=
MODEL ?= qwen2.5:1.5b
CMD ?= bash
SERVICE ?= app

# Internal variables (not meant to be overridden)
APP_RUN := $(COMPOSE) run --rm $(APP_SERVICE)
UV_RUN := $(APP_RUN) uv run

# Capture extra arguments passed after target name (for advanced usage)
EXTRA_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
$(eval $(EXTRA_ARGS):;@:)

# Declare all phony targets (targets that don't create files)
.PHONY: help bootstrap clean clean-all \
	up up-detached up-models up-embedding down logs \
	ui streamlit llm-pull \
	crawl ingest ingest-reset refresh \
	embeddings embeddings-reset \
	test test-workflow verify-tables verify-tables-doc \
	lint lint-fix format format-check fix \
	export-feedback finetune \
	run exec wait

# Help target - must be first target after .PHONY
help: ## Show this help message
	@echo "RBA Document Intelligence Platform - Available Commands"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*##"; printf "Usage: make \033[36m<target>\033[0m\n\n"} \
		{printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

# =============================================================================
# Setup & Bootstrap
# =============================================================================

bootstrap: ## Build app image and install all dependencies (dev + prod)
	@echo "Building app container..."
	$(COMPOSE) build app
	@echo "Installing dependencies (including dev tools)..."
	$(APP_RUN) uv sync --extra dev
	@echo "✓ Bootstrap complete"

clean: ## Remove Python cache files and temp artifacts
	@echo "Cleaning Python cache..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cache cleaned"

clean-all: down clean ## Stop all services and clean everything
	@echo "Removing Docker volumes..."
	$(COMPOSE) down -v
	@echo "✓ Full cleanup complete"

# =============================================================================
# Docker Service Management
# =============================================================================

up: ## Start all services in foreground (Ctrl+C to stop)
	$(COMPOSE) up

up-detached: ## Start all services in background
	$(COMPOSE) up -d
	@echo "Services started. Use 'make logs' to view logs."

up-models: ## Start only embedding + LLM services
	$(COMPOSE) up -d embedding llm

up-embedding: ## Start only embedding service
	$(COMPOSE) up -d embedding

down: ## Stop and remove all services
	$(COMPOSE) down

logs: ## Follow logs for a service (default: app, override with SERVICE=name)
	$(COMPOSE) logs -f $(SERVICE)

# =============================================================================
# UI & Models
# =============================================================================

ui: ## Start Streamlit UI in foreground
	$(COMPOSE) up app

streamlit: ## Launch Streamlit UI inside running container
	$(UV_RUN) streamlit run app/ui/streamlit_app.py $(ARGS) $(EXTRA_ARGS)

llm-pull: ## Pull Ollama model (default: qwen2.5:1.5b, override with MODEL=name)
	$(COMPOSE) exec llm ollama pull $(MODEL)

# =============================================================================
# Data Pipeline
# =============================================================================

crawl: ## Crawl RBA website for PDFs
	@echo "Starting PDF crawler..."
	$(UV_RUN) scripts/crawler_rba.py $(ARGS) $(EXTRA_ARGS)

ingest: ## Ingest PDFs (extract text + tables)
	@echo "Processing PDFs..."
	$(UV_RUN) python scripts/ingest_documents.py $(ARGS) $(EXTRA_ARGS)

ingest-reset: ## Reset all documents to NEW status
	@echo "Resetting document status..."
	$(UV_RUN) python scripts/ingest_documents.py --reset

ingest-retry: ## Retry all FAILED documents
	@echo "Retrying failed documents..."
	$(UV_RUN) python scripts/ingest_documents.py --retry-failed

embeddings: ## Generate embeddings for chunks
	@echo "Building embeddings..."
	$(UV_RUN) scripts/build_embeddings.py $(ARGS) $(EXTRA_ARGS)

embeddings-reset: ## Rebuild all embeddings from scratch
	@echo "Rebuilding embeddings..."
	$(UV_RUN) scripts/build_embeddings.py --reset

refresh: ## Full refresh: crawl + ingest + embeddings
	@echo "Running full data refresh..."
	$(UV_RUN) python scripts/refresh_pdfs.py $(ARGS) $(EXTRA_ARGS)

# =============================================================================
# ML & Fine-tuning
# =============================================================================

export-feedback: ## Export user feedback as training pairs
	$(UV_RUN) python scripts/export_feedback_pairs.py $(ARGS) $(EXTRA_ARGS)

finetune: ## Fine-tune model with LoRA DPO
	$(UV_RUN) python scripts/finetune_lora_dpo.py $(ARGS) $(EXTRA_ARGS)

# =============================================================================
# Testing & Verification
# =============================================================================

test: ## Run pytest tests
	$(UV_RUN) pytest $(ARGS) $(EXTRA_ARGS)

test-workflow: ## Run end-to-end workflow test
	@echo "Running E2E workflow test..."
	$(UV_RUN) python scripts/test_workflow.py

verify-tables: ## Show table extraction statistics
	$(UV_RUN) python scripts/verify_table_extraction.py stats

verify-tables-doc: ## Verify tables for specific doc (use: ARGS="doc <id>")
	$(UV_RUN) python scripts/verify_table_extraction.py $(ARGS)

# =============================================================================
# Code Quality & CI/CD
# =============================================================================

lint: ## Check code with ruff linter
	$(UV_RUN) ruff check $(ARGS) $(EXTRA_ARGS)

lint-fix: ## Auto-fix safe linting issues
	$(UV_RUN) ruff check --fix

format: ## Format code with ruff
	$(UV_RUN) ruff format $(ARGS) $(EXTRA_ARGS)

format-check: ## Check code formatting (CI-friendly)
	$(UV_RUN) ruff format --check

fix: ## Auto-fix ALL issues (lint + format)
	@echo "Fixing linting issues..."
	$(UV_RUN) ruff check --fix --unsafe-fixes
	@echo "Formatting code..."
	$(UV_RUN) ruff format
	@echo "✓ All fixes applied"

ci: format-check lint test ## Run all CI checks (format + lint + test)
	@echo "✓ All CI checks passed"

# =============================================================================
# Utilities
# =============================================================================

wait: ## Wait for services to be healthy
	$(UV_RUN) python scripts/wait_for_services.py $(ARGS) $(EXTRA_ARGS)

run: ## Run arbitrary command in app container (use: CMD="command")
	$(APP_RUN) $(CMD)

exec: ## Execute command in running container (use: CMD="command")
	$(COMPOSE) exec $(APP_SERVICE) $(CMD)
