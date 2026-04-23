PYTHON ?= python
DOCKER_IMAGE ?= peekaboo:dev

.PHONY: install test lint check coverage build docker-build docker-smoke docker-run-synthetic clean

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
	$(PYTHON) -m pytest -q --cov-report=term-missing --cov-report=html --cov-report=xml

build:
	$(PYTHON) -m build

docker-build:
	docker build -t $(DOCKER_IMAGE) .

docker-smoke: docker-build
	docker run --rm $(DOCKER_IMAGE) --help
	docker run --rm --entrypoint python $(DOCKER_IMAGE) examples/generate_synthetic_capture.py --output /tmp/synthetic-demo.pcap
	docker run --rm --entrypoint /bin/sh $(DOCKER_IMAGE) -c "python examples/generate_synthetic_capture.py && peekaboo inspect --config configs/synthetic-demo.yaml"

docker-run-synthetic:
	mkdir -p runs examples/captures
	docker compose run --rm -T generate-synthetic
	docker compose run --rm -T peekaboo run --config configs/synthetic-demo.yaml --force --quiet

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
