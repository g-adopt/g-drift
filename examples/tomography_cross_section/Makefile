.PHONY: notebook data clean test expected

# Convert Python script to notebook
notebook: demo.ipynb

demo.ipynb: demo.py
	jupytext --to notebook --execute demo.py

# Download required datasets (gdrift auto-downloads)
data:
	@echo "No additional data needed - gdrift auto-downloads datasets"

test:
	@echo "No regression tests yet for tomography_cross_section"

expected:
	@echo "No expected values yet for tomography_cross_section"

clean:
	rm -f demo.ipynb
	rm -rf __pycache__
