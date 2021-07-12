install:
	pip install .

dependencies:
	pip install -Ur requirements.txt

test:
	pytest

test-cov:
	pytest --cov=./ --cov-report=xml

.PHONY: install dependencies test test-cov
