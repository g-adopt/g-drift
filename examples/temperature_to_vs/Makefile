.PHONY: notebook test data clean expected

# Convert Python script to notebook
notebook: demo.ipynb

demo.ipynb: demo.py
	jupytext --to notebook --execute demo.py

# Run regression tests
test:
	pytest test_demo.py -v

# Regenerate expected values
expected:
	python generate_expected.py

# Download required datasets (gdrift auto-downloads)
data:
	@echo "No additional data needed - gdrift auto-downloads datasets"

clean:
	rm -f demo.ipynb
	rm -rf __pycache__
