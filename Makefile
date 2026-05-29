.PHONY: setup extract extract-test extract-meta extract-meta-test status data-sanity build-run-value build-folds build-profile-cache test lint clean-test demo demo-api demo-web

START ?= 2017-03-15
END   ?= $(shell date +%Y-%m-%d)

setup:
	uv sync

extract:
	uv run python -m data.extract_statcast --start $(START) --end $(END)

extract-test:
	uv run python -m data.extract_statcast \
		--start 2024-04-01 --end 2024-04-03 \
		--output data/raw_test \
		--checkpoint data/raw_test/_checkpoint.json

extract-meta:
	uv run python -m data.extract_game_metadata

extract-meta-test:
	uv run python -m data.extract_game_metadata \
		--raw-dir data/raw_test \
		--output-dir data/game_metadata_test \
		--limit 5

status:
	uv run python -m scripts.status

data-sanity:
	uv run python -m scripts.data_sanity

build-run-value:
	uv run python -m scripts.build_run_value

build-folds:
	uv run python -m scripts.build_folds

build-profile-cache:
	uv run python -m scripts.build_profile_cache

test:
	uv run pytest

lint:
	uv run ruff check .

clean-test:
	rm -rf data/raw_test data/game_metadata_test

# --- Demo ---
#
# The demo is two processes: the FastAPI backend (uvicorn on :8000) and the
# Vite dev server for the React frontend (:5173, proxies /api/* to :8000).
# `make demo` prints how to run them; `make demo-api` and `make demo-web`
# are each a foreground process you run in its own terminal.
#
# Requires the v7 checkpoint at
# ``checkpoints_modal/tiny-fold0-v7/checkpoint_calibrated.pt`` —
# pull from the Modal Volume if it's missing
# (``modal volume get pitchgpt-data checkpoints/tiny-fold0-v7 ...``).

demo:
	@echo ""
	@echo "PitchGPT demo — two processes, two terminals:"
	@echo "  terminal 1:  make demo-api    # FastAPI on :8000"
	@echo "  terminal 2:  make demo-web    # Vite + React on :5173"
	@echo ""
	@echo "Then open http://localhost:5173 in a browser."
	@echo ""

demo-api:
	uv run uvicorn inference.api:app --reload --host 127.0.0.1 --port 8000

demo-web:
	cd frontend && npm run dev
