# Makefile to simplify test and build.

.PHONY: all
all: test

.PHONY: all
build-python:
	python setup.py sdist bdist_wheel

.PHONY: clean
clean:
	rm -rf coverage_html_report .coverage
	rm -rf bff.egg-info
	rm -rf venv-dev
	rm -rf dist/

.PHONY: test
test: code lint style coverage

.PHONY: baseline-plot
baseline-plot:
	pytest --mpl-generate-path=tests/baseline tests

.PHONY: code
code:
	pytest --mpl tests

.PHONY: lint
lint:
	pytest --pylint --pylint-rcfile=.pylintrc --pylint-error-types=CWEF

.PHONY: style
style:
	flake8
	mypy bff tests
	pytest --codestyle --docstyle

.PHONY: coverage
coverage:
	rm -rf coverage_html_report .coverage
	pytest --cov=bff tests --cov-report=html:coverage_html_report
