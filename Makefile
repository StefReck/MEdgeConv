install:
	pip install .

dependencies:
	pip install -r requirements.txt

test:
	pytest

test-cov:
	pytest --cov=./
