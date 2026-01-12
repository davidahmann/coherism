.PHONY: install test figures paper cover_letter data_availability clean

VENV ?= .venv
PY := $(VENV)/bin/python

install:
	$(PY) -m pip install -e . --no-build-isolation

test:
	$(PY) -m unittest discover -s tests -p 'test_*.py'

figures:
	$(PY) generate_figures.py

paper:
	latexmk -pdf -outdir=build alfm_bem.tex

cover_letter:
	latexmk -pdf -outdir=build cover_letter.tex

data_availability:
	latexmk -pdf -outdir=build data_availability.tex

clean:
	latexmk -C -outdir=build alfm_bem.tex || true
	latexmk -C -outdir=build cover_letter.tex || true
	latexmk -C -outdir=build data_availability.tex || true
	rm -rf build
