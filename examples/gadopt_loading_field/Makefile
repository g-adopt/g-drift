.PHONY: notebook test data clean expected

# Convert Python script to notebook
notebook: demo.ipynb

demo.ipynb: demo.py
	jupytext --to notebook --execute demo.py

# Run regression tests
test:
	pytest test_demo.py -v

# Generate expected values (run with Firedrake-enabled Python)
expected:
	python generate_expected.py

# Download required datasets (gdrift auto-downloads)
data:
	@echo "No additional data needed - gdrift auto-downloads datasets"

clean:
	rm -f demo.ipynb
	rm -f REVEAL.pvd REVEAL_temperature.h5
	rm -rf __pycache__
