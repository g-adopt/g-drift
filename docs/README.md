# gdrift Documentation

This directory contains the source files for the gdrift documentation website at [gdrift.gadopt.org](https://gdrift.gadopt.org).

## Structure

```
docs/
├── index.md              # Home page
├── getting-started.md    # Installation and quickstart
├── about.md              # Architecture and design
├── api.md                # API reference (auto-generated from docstrings)
├── user-guide/           # Topic-specific guides
│   ├── index.md
│   ├── datasets.md       # Dataset registry and management
│   ├── profiles.md       # 1D radial profiles
│   ├── seismic.md        # 3D tomography models
│   ├── thermodynamics.md # Temperature-velocity conversions
│   └── anelasticity.md   # Anelastic corrections
├── examples/             # Jupyter notebook demos
│   └── index.md          # Examples landing page
└── CNAME                 # Custom domain configuration
```

## Building Documentation Locally

### Install dependencies
```bash
pip install -e ".[docs,examples]"
```

### Serve documentation locally
```bash
mkdocs serve
```

Then open http://localhost:8000 in your browser.

### Build static site
```bash
mkdocs build
```

Output is in `site/` directory.

## Automatic Deployment

Documentation is automatically built and deployed to GitHub Pages when:

1. Code is pushed to the `main` branch
2. Firedrake tests pass successfully
3. The `docs.yml` workflow runs

See `.github/workflows/docs.yml` for details.

## Writing Documentation

### Markdown

Documentation is written in GitHub-flavored Markdown with extensions:

- **Code blocks**: Syntax highlighting with ` ```python `
- **Admonitions**: `!!! note "Title"`
- **Math**: LaTeX via MathJax `$E = mc^2$`
- **Links**: `[text](path/to/page.md)`

### API Reference

API documentation is auto-generated from docstrings using mkdocstrings:

```markdown
::: gdrift.ClassName
    options:
      members:
        - method1
        - method2
```

### Examples

Example notebooks are generated from Python scripts in `examples/*/demo.py`:

1. Write demo as Python script with `# %%` cell markers
2. Run `make notebooks` in examples directory
3. Notebooks are auto-executed during documentation build

## Custom Domain

The site is served at `gdrift.gadopt.org` via the `CNAME` file.

DNS configuration:
- CNAME record: `gdrift.gadopt.org` → `sghelichkhani.github.io`
- GitHub Pages automatically provisions HTTPS certificate

## Configuration

Site configuration is in `mkdocs.yml` at the repository root:

- **Theme**: Material for MkDocs
- **Plugins**: mkdocstrings (API docs), mkdocs-jupyter (notebooks)
- **Extensions**: Code highlighting, math rendering, admonitions

## Contributing

When adding new documentation:

1. Create or edit Markdown files in this directory
2. Update `mkdocs.yml` navigation if adding new pages
3. Test locally with `mkdocs serve`
4. Commit and push - documentation builds automatically

For API reference updates, edit docstrings in the source code.
