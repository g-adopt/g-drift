.PHONY: all notebooks data clean help test

# Demo directories
DEMOS := mantle_solidus anelasticity_corrections temperature_to_vs tomography_models geodynamic_adiabat linearisation gadopt_loading_field

help:
	@echo "Available targets:"
	@echo "  all        - Download data and generate all notebooks"
	@echo "  notebooks  - Convert all Python scripts to notebooks"
	@echo "  test       - Run regression tests for all demos"
	@echo "  data       - Download/prepare all demo data"
	@echo "  clean      - Remove generated notebooks and artifacts"
	@echo ""
	@echo "Demo directories: $(DEMOS)"

# Build everything
all: data notebooks

# Convert all Python scripts to notebooks
notebooks:
	@echo "Converting Python scripts to notebooks..."
	@for dir in $(DEMOS); do \
		if [ -d "$$dir" ] && [ -f "$$dir/Makefile" ]; then \
			echo "Processing $$dir..."; \
			$(MAKE) -C "$$dir" notebook || exit 1; \
		fi \
	done
	@echo "All notebooks generated successfully!"

# Run regression tests
test:
	@for dir in $(DEMOS); do \
		if [ -d "$$dir" ] && [ -f "$$dir/Makefile" ]; then \
			echo "Testing $$dir..."; \
			$(MAKE) -C "$$dir" test || exit 1; \
		fi \
	done

# Download/prepare all demo data
data:
	@echo "Preparing demo data..."
	@for dir in $(DEMOS); do \
		if [ -d "$$dir" ] && [ -f "$$dir/Makefile" ]; then \
			$(MAKE) -C "$$dir" data || true; \
		fi \
	done
	@echo "Data preparation complete!"

# Clean all generated files
clean:
	@echo "Cleaning generated files..."
	@for dir in $(DEMOS); do \
		if [ -d "$$dir" ] && [ -f "$$dir/Makefile" ]; then \
			$(MAKE) -C "$$dir" clean; \
		fi \
	done
	@echo "Clean complete!"
