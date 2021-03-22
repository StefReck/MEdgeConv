install:
	pip install .

dependencies:
	pip install -r requirements.txt

test:
	pytest

test-cov:
	pytest --cov=./

.PHONY: install dependencies test test-cov
