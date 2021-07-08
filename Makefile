install:
	pip install .

dependencies:
	pip install -Ur requirements.txt

test:
	pytest

test-cov:
	pytest --cov=./

.PHONY: install dependencies test test-cov
