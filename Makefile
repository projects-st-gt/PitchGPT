.PHONY: setup extract extract-test extract-meta extract-meta-test status data-sanity build-run-value build-folds build-profile-cache test lint clean-test

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
