.PHONY: notebook data clean test expected

notebook: demo.ipynb

demo.ipynb: demo.py
	jupytext --to notebook --execute demo.py

# The test fixture auto-downloads the GPlates rotation and coastlines
# files from the gadopt CDN into gdrift/data-gplates/ on first run.
# Override with GDRIFT_DEMO_ROTATION_FILE / GDRIFT_DEMO_COASTLINES_FILE
# if you want to point at a different reconstruction.
data:
	@echo "GPlates data auto-downloads from the gadopt CDN on first test run."
	@echo "Override with GDRIFT_DEMO_ROTATION_FILE / GDRIFT_DEMO_COASTLINES_FILE."

test:
	pytest -v test_demo.py

expected:
	python generate_expected.py

clean:
	rm -f demo.ipynb *.pvd expected.pkl
	rm -rf coastlines __pycache__ .pytest_cache
