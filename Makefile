.PHONY: dev test smoke

dev:
	.venv/bin/uvicorn backend.main:app --reload --port 8000

test:
	.venv/bin/python -m pytest -m "not pending" -v

smoke:
	.venv/bin/python scripts/llm_smoke_test.py
