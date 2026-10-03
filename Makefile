.PHONY: dev test stream smoke

dev:
	.venv/bin/uvicorn backend.main:app --reload --port 8000

test:
	.venv/bin/python -m pytest -m "not pending" -v

stream:
	curl -N "http://localhost:8000/stream/seed?speed=600&to_minute=60"

smoke:
	.venv/bin/python scripts/llm_smoke_test.py
