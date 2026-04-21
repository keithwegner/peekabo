PYTHON ?= python

.PHONY: install test lint check coverage build clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

check: lint
	$(PYTHON) -m compileall -q src tests
	$(PYTHON) -m pytest -q

coverage:
	$(PYTHON) -m pytest -q --cov-report=term-missing --cov-report=html

build:
	$(PYTHON) -m build

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
