.PHONY: setup lint format test demo clean

PYTHONPATH := src

setup:
	pipenv install --dev

lint:
	PYTHONPATH=$(PYTHONPATH) pipenv run pylint src/ tests/ --fail-under=9.0 || true
	PYTHONPATH=$(PYTHONPATH) pipenv run black --check src/ tests/

format:
	PYTHONPATH=$(PYTHONPATH) pipenv run black src/ tests/

test:
	PYTHONPATH=$(PYTHONPATH) pipenv run pytest tests/ -v

demo:
	PYTHONPATH=$(PYTHONPATH) pipenv run python -m finance_rag_eval.cli ingest
	PYTHONPATH=$(PYTHONPATH) pipenv run python -m finance_rag_eval.cli build-index
	PYTHONPATH=$(PYTHONPATH) pipenv run python -m finance_rag_eval.cli query "What is the revenue?"

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	rm -rf outputs/
	rm -rf .dagster/

