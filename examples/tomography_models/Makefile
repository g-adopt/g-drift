.PHONY: notebook data clean

# Convert Python script to notebook
notebook: demo.ipynb

demo.ipynb: demo.py
	jupytext --to notebook --execute demo.py

# Download required datasets (gdrift auto-downloads)
data:
	@echo "No additional data needed - gdrift auto-downloads datasets"

clean:
	rm -f demo.ipynb
	rm -rf __pycache__
