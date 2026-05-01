.PHONY: help setup start-db stop-db run-pipeline start-backend test

help:
	@echo ""
	@echo "Family Tree Project - Available Commands"
	@echo "========================================"
	@echo "  make setup          Install Python dependencies"
	@echo "  make start-db       Start FalkorDB in Docker"
	@echo "  make stop-db        Stop FalkorDB"
	@echo "  make run-pipeline   Load CSV data into FalkorDB"
	@echo "  make start-backend  Start the FastAPI backend server"
	@echo "  make test           Run all tests"
	@echo ""

setup:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Done! Copy .env.example to .env and fill in your keys."
	cp -n .env.example .env || true

start-db:
	@echo "🚀 Starting FalkorDB..."
	docker-compose up -d
	@echo "✅ FalkorDB running!"
	@echo "   Redis API : localhost:6379"
	@echo "   Browser UI: http://localhost:3000"

stop-db:
	@echo "🛑 Stopping FalkorDB..."
	docker-compose down

run-pipeline:
	@echo "🔄 Running data pipeline..."
	python pipeline/load_graph.py

start-backend:
	@echo "🚀 Starting FastAPI backend..."
	uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

test:
	@echo "🧪 Running tests..."
	python -m pytest tests/ -v