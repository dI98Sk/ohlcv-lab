# The whole pipeline from an empty clone: make install data layer figures report
# Each target is one step; `make help` lists them.

SYMBOLS := BTCUSDT ETHUSDT SOLUSDT XRPUSDT DOGEUSDT
# Must match ohlcv_lab.report.END: the report uses exactly [2023-10-01, 2026-10-01).
END := 2026-10-01

.PHONY: help install lock data layer figures report bench test lint check

help:  ## list targets
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

install:  ## exact versions from requirements.lock (Python 3.12), then the package itself
	pip install -r requirements.lock
	pip install -e ".[dev]" --no-deps
	nbstripout --install

lock:  ## refresh requirements.lock after changing dependencies
	pip freeze --exclude-editable > requirements.lock

data:  ## raw layer from the Bybit API (reruns download only what is missing)
	ohlcv-fetch --symbols $(SYMBOLS) --interval 60 --start 2023-10-01 --end $(END)
	ohlcv-fetch --symbols BTCUSDT --interval 1 --start 2025-10-01 --end $(END)

layer:  ## clean Parquet layer and quality report
	ohlcv-clean

figures:  ## README figures (docs/img) and the summary table
	ohlcv-report --out docs/img

report:  ## executed report notebook as HTML (reports/report.html)
	jupyter nbconvert --to html --execute notebooks/report.ipynb --output-dir reports

bench:  ## pandas vs Polars vs DuckDB on minute candles
	ohlcv-bench

test:  ## unit tests (synthetic data, no network)
	pytest -q

lint:  ## lint and formatting check, as in CI
	ruff check src tests notebooks/report.ipynb
	ruff format --check src tests notebooks/report.ipynb

check: lint test  ## everything CI runs
