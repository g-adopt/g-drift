.PHONY: lint test longtest longtest_output convert_demos data-catalog enrich-metadata docs

lint:
	@echo "Linting module code"
	@python3 -m flake8 gdrift
	@echo "Linting examples and tests"
	@python3 -m flake8 examples tests

data-catalog:
	@echo "Generating data catalog documentation"
	@python3 scripts/generate_data_catalog.py

enrich-metadata:
	@echo "Enriching dataset metadata from HDF5 files and web search"
	@python3 scripts/enrich_datasets_metadata.py --web-search

docs: data-catalog
	@echo "Building documentation with MkDocs"
	@mkdocs build --strict
