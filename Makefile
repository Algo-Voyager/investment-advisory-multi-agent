PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: install run test lint ui ingest eval notebook

install:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

run:  ## CLI chat
	$(PY) -m app.cli --client CLT-001 "what do I own?"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check app tests

ui:  ## Streamlit demo UI (arrives in Phase 14)
	$(PY) -m streamlit run ui/streamlit_app.py

ingest:  ## Knowledge-base ingestion (arrives in Phase 5)
	$(PY) -m scripts.ingest_kb --tickers NVDA MSFT --limit 5

eval:  ## Evaluation suite (arrives in Phase 13)
	$(PY) -m evals.report

notebook:
	$(PY) -m jupyterlab notebooks/
