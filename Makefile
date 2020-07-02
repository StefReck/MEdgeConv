dependencies:
	pip install -r requirements.txt

test:
	pytest

test-cov:
	py.test --cov medgeconv/ --cov-report term-missing --cov-report html:reports/coverage
