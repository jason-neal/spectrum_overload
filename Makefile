# Python makefile https://krzysztofzuraw.com/blog/2016/makefiles-in-python-projects.html
# Delcare all non-file targets as phony
.PHONY: clean clean-build clean-data data isort lint test
TEST_PATH=./

help:
	@echo "	clean-pyc"
	@echo "		Remove python artifacts."
	@echo "	clean-build"
	@echo "		Remove build artifacts."
	@echo "	isort"
	@echo "		Sort import statements."
	@echo "	lint"
	@echo "		Check style with flake8."
	@echo "	test"
	@echo "		Run py.test"
	@echo "	init"
	@echo "		Initalize by installing requirements"
	@echo "	init-dev"
	@echo "		Initalize by installing normal and dev requirements"
	@echo "	cov"
	@echo "		Produce coverage report"
	@echo "	mypy"
	@echo "		Run type checking with mypy"

clean-pyc:
	find . -name '*.pyc' -exec rm --force {} +
	find . -name '*.pyo' -exec rm --force {} +
	find . -name '*~' -exec rm --force  {} +

clean-build:
	rm --force --recursive build/
	rm --force --recursive dist/
	rm --force --recursive *.egg-info

isort:
	sh -c "isort  --recursive . "

lint:
	flake8 --exclude=.tox

test: clean-pyc
	py.test --verbose --color=yes $(TEST_PATH)

init:
	pip install -r requirements.txt

init-dev:
	pip install -r requirements.txt
	pip install -r requirements_dev.txt

cov: $(module)/*
	py.test --cov=$(module)
	coverage html

mypy:
	# py.test --mypy
	mypy --ignore-missing-imports .
