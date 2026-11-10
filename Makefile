.PHONY: setup migrate seed test test-pg smoke dataset validate-dataset api worker demo

setup:
	pip install -e ".[dev]"

migrate:
	alembic upgrade head

seed:
	python -m scripts.seed

test:
	pytest -q

smoke:
	python -m scripts.smoke

dataset:
	python -m scripts.build_dataset

validate-dataset:
	python -m scripts.validate_dataset

# Needs a scratch PostgreSQL; the schema is DROPPED and recreated per test.
#   make test-pg TEST_DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/scratch
test-pg:
	TEST_DATABASE_URL=$(TEST_DATABASE_URL) pytest tests/pg -q

api:
	uvicorn app.main:app --reload

worker:
	python -m app.execution.worker

demo:
	python -m scripts.demo_comparison
