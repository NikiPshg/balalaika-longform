.PHONY: reproduce test
PYTHON ?= python
reproduce:
	$(PYTHON) scripts/reproduce.py
test:
	$(PYTHON) -m pytest -q
