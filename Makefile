.PHONY: help install dev test clean docker-build docker-run docker-stop lint format

# Default target
help:
	@echo "AI Manager - Available commands:"
	@echo ""
	@echo "Development:"
	@echo "  install     - Install dependencies"
	@echo "  dev         - Run development server"
	@echo "  test        - Run tests"
	@echo "  lint        - Run linting"
	@echo "  format      - Format code"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run with Docker Compose"
	@echo "  docker-stop  - Stop Docker services"
	@echo ""
	@echo "Database:"
	@echo "  db-init     - Initialize database"
	@echo "  db-migrate  - Run database migrations"
	@echo ""
	@echo "Utilities:"
	@echo "  clean       - Clean up generated files"
	@echo "  example     - Run example training script"

# Development
install:
	pip install -r requirements.txt
	pip install -e .

dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v

lint:
	black --check app/ ai_manager/ tests/
	isort --check-only app/ ai_manager/ tests/
	flake8 app/ ai_manager/ tests/

format:
	black app/ ai_manager/ tests/
	isort app/ ai_manager/ tests/

# Docker
docker-build:
	docker build -t ai-manager .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

# Database
db-init:
	python -c "from app.database import create_tables; create_tables()"

db-migrate:
	alembic upgrade head

# Utilities
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/

example:
	python examples/training_example.py

# Production
prod:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# Setup
setup: install db-init
	@echo "AI Manager setup complete!"
	@echo "Run 'make dev' to start the development server"
	@echo "Visit http://localhost:8000 to access the dashboard" 