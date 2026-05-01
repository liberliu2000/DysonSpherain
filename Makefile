.PHONY: paper paper-tables paper-figures paper-cases paper-validate

PYTHON ?= .venv312/bin/python

paper-tables:
	$(PYTHON) -m scripts.generate_paper_tables

paper-figures:
	$(PYTHON) -m scripts.generate_paper_figures

paper-cases:
	$(PYTHON) -m scripts.generate_case_studies

paper-validate:
	$(PYTHON) -m scripts.validate_paper_claims

paper: paper-tables paper-figures paper-cases
	cd paper/latex && tectonic main.tex
	$(PYTHON) -m scripts.validate_paper_claims
