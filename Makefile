.PHONY: setup lint test demo eval sweep dagster clean

PYTHONPATH := src

setup:
	pipenv install --dev

lint:
	PYTHONPATH=$(PYTHONPATH) pipenv run ruff check src/ tests/
	PYTHONPATH=$(PYTHONPATH) pipenv run ruff format --check src/ tests/

format:
	PYTHONPATH=$(PYTHONPATH) pipenv run ruff format src/ tests/
	PYTHONPATH=$(PYTHONPATH) pipenv run ruff check --fix src/ tests/

test:
	PYTHONPATH=$(PYTHONPATH) pipenv run pytest tests/ -v

demo:
	PYTHONPATH=$(PYTHONPATH) pipenv run python -m finance_rag_eval.cli ingest
	PYTHONPATH=$(PYTHONPATH) pipenv run python -m finance_rag_eval.cli build-index
	PYTHONPATH=$(PYTHONPATH) pipenv run python -m finance_rag_eval.cli query "What is the revenue?"

eval:
	PYTHONPATH=$(PYTHONPATH) pipenv run python -m finance_rag_eval.cli eval

sweep:
	PYTHONPATH=$(PYTHONPATH) pipenv run python -m finance_rag_eval.cli sweep

dagster:
	@echo "Run: PYTHONPATH=$(PYTHONPATH) pipenv run dagster dev -m finance_rag_eval.dagster_app.definitions"

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	rm -rf outputs/
	rm -rf .dagster/

