COMPOSE ?= docker compose
APP_SERVICE ?= app
APP_RUN := $(COMPOSE) run --rm $(APP_SERVICE)
UV_RUN := $(APP_RUN) uv run
ARGS ?=
MODEL ?= qwen2.5:7b
CMD ?= bash
SERVICE ?= app

.PHONY: help bootstrap up up-detached up-models up-embedding ui down crawl process tables tables-force embeddings embeddings-reset refresh streamlit debug export-feedback finetune test lint format llm-pull logs run exec wait

help: ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*##' Makefile | sort | awk 'BEGIN {FS = ":.*##"}; {printf "%s\t%s\n", $$1, $$2}'

bootstrap: ## Build the app image and sync dependencies inside the container
	$(COMPOSE) build app
	$(APP_RUN) uv sync

up: ## Start the full docker compose stack in the foreground
	$(COMPOSE) up

up-detached: ## Start all services in the background
	$(COMPOSE) up -d

up-models: ## Start only the embedding + LLM services in the background
	$(COMPOSE) up -d embedding llm

up-embedding: ## Start only the embedding service in the background
	$(COMPOSE) up -d embedding

ui: ## Start just the Streamlit app service
	$(COMPOSE) up app

llm-pull: ## Pull the configured Ollama model inside the running llm container
	$(COMPOSE) exec llm ollama pull $(MODEL)

crawl: ## Run scripts/crawler_rba.py inside the app container
	$(UV_RUN) scripts/crawler_rba.py $(ARGS)

process: ## Run scripts/process_pdfs.py inside the app container
	$(UV_RUN) scripts/process_pdfs.py $(ARGS)

tables: ## Run scripts/extract_tables.py to extract structured tables
	$(UV_RUN) scripts/extract_tables.py $(ARGS)

tables-force: ## Re-extract all tables even if they already exist
	$(UV_RUN) scripts/extract_tables.py --force

embeddings: ## Run scripts/build_embeddings.py (set ARGS="--reset" to wipe vectors)
	$(UV_RUN) scripts/build_embeddings.py $(ARGS)

embeddings-reset: ## Shortcut to rebuild embeddings after wiping vectors
	$(UV_RUN) scripts/build_embeddings.py --reset

refresh: ## Run scripts/refresh_pdfs.py convenience wrapper
	$(UV_RUN) python scripts/refresh_pdfs.py $(ARGS)

streamlit: ## Launch the Streamlit UI from inside the container
	$(UV_RUN) streamlit run app/ui/streamlit_app.py $(ARGS)

debug: ## Run scripts/debug_dump.py for ingestion stats
	$(UV_RUN) scripts/debug_dump.py $(ARGS)

export-feedback: ## Export thumbs up/down feedback as preference pairs
	$(UV_RUN) python scripts/export_feedback_pairs.py $(ARGS)

finetune: ## Launch the LoRA DPO fine-tuning script
	$(UV_RUN) python scripts/finetune_lora_dpo.py $(ARGS)

test: ## Run pytest (pass ARGS="tests/..." for a subset)
	$(UV_RUN) pytest $(ARGS)

lint: ## Run ruff check
	$(UV_RUN) ruff check $(ARGS)

format: ## Run ruff format
	$(UV_RUN) ruff format $(ARGS)

logs: ## Follow logs for a specific service (SERVICE=name)
	$(COMPOSE) logs -f $(SERVICE)

run: ## Run an arbitrary command via `docker compose run --rm app`
	$(APP_RUN) $(CMD)

exec: ## Execute a command in the running app container
	$(COMPOSE) exec $(APP_SERVICE) $(CMD)

down: ## Stop and remove all services
	$(COMPOSE) down

wait: ## Wait for Postgres/MinIO/embedding health checks
	$(UV_RUN) python scripts/wait_for_services.py $(ARGS)
