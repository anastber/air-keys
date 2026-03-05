.PHONY: install dev test lint clean help
.DEFAULT_GOAL := help

# Variables
UV := uv
NPM := npm
PYTHON_SRC := backend
FRONTEND_SRC := frontend

help: ## Show this help message
	@echo "AirKeys Development Commands"
	@echo "=============================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies (Python + Node.js)
	@echo "Installing Python dependencies with uv..."
	$(UV) sync
	@echo "Installing Node.js dependencies..."
	cd $(FRONTEND_SRC) && $(NPM) install
	@echo "✅ All dependencies installed!"

dev: ## Start development servers (backend + frontend)
	@echo "Starting development servers..."
	@echo "Backend will run on http://localhost:8000"
	@echo "Frontend will run on http://localhost:3000"
	@echo "Use Ctrl+C to stop both servers"
	@trap 'kill 0' SIGINT; \
	(cd $(FRONTEND_SRC) && $(NPM) run dev) & \
	(. .venv/bin/activate && uvicorn $(PYTHON_SRC).api.main:app --reload --host 0.0.0.0 --port 8000) & \
	wait

test: ## Run all tests (Python + Node.js)
	@echo "Running Python tests..."
	. .venv/bin/activate && pytest $(PYTHON_SRC)/tests/ -v
	@echo "Running Node.js tests..."
	cd $(FRONTEND_SRC) && $(NPM) test
	@echo "✅ All tests passed!"

lint: ## Run linting and formatting (Python + Node.js)
	@echo "Linting Python code with ruff..."
	. .venv/bin/activate && ruff check $(PYTHON_SRC)/
	. .venv/bin/activate && ruff format --check $(PYTHON_SRC)/
	@echo "Linting Node.js code with ESLint..."
	cd $(FRONTEND_SRC) && $(NPM) run lint
	@echo "✅ All linting passed!"

format: ## Format all code (Python + Node.js)
	@echo "Formatting Python code with ruff..."
	. .venv/bin/activate && ruff format $(PYTHON_SRC)/
	@echo "Formatting complete!"

clean: ## Clean build artifacts and caches
	@echo "Cleaning Python cache..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaning Node.js cache..."
	cd $(FRONTEND_SRC) && rm -rf node_modules/.cache 2>/dev/null || true
	@echo "✅ Cleanup complete!"

install-dev: ## Install development dependencies
	$(UV) sync --dev
	@echo "✅ Development dependencies installed!"