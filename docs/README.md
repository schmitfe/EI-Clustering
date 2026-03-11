# API Documentation

This folder contains the generated static API documentation for:

- `spiketools`
- `plotting`
- `BinaryNetwork`
- `MeanField`
- `sim_config`

Open `index.html` locally to browse the site, or publish this directory via
GitHub Pages.

## Regeneration

From the repository root:

```bash
python scripts/generate_api_docs.py
```

This rebuilds the plotting example assets and then regenerates the `pdoc`
site.

## GitHub Pages

The repository includes a workflow at `.github/workflows/docs.yml` that:

1. installs the documentation dependencies,
2. runs `python scripts/generate_api_docs.py`,
3. uploads `docs/` as the Pages artifact,
4. deploys the generated static site.

The published entry point is `docs/index.html`.
