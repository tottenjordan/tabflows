# Topic Note: Tooling & Environment Management

## Package Resolution Insights with `uv`
- **Default Index**: When running on Google internal workstations, `uv` may attempt to query an internal artifact index (Corp Airlock) that requires specific authentication or flags.
- **PyPI Index Configuration**: Added explicit PyPI default index configuration in `pyproject.toml`:
  ```toml
  [[tool.uv.index]]
  url = "https://pypi.org/simple"
  default = true
  ```
  Or pass `--default-index https://pypi.org/simple` when invoking ad-hoc `uv` commands.

## Pre-commit & Formatting
- Formatting and linting are strictly handled by `ruff` (`uv run ruff check .` and `uv run ruff format .`).
- Static type checking is handled by `ty` (`uv run ty check src/`).
- Test execution is handled by `pytest` (`uv run pytest`).

## Packaging & Distribution
- Packages are built using `hatchling` backend via `uv build`, outputting wheel (`.whl`) and source tarball (`.tar.gz`) packages to `dist/`.
- Distribution wheels can be installed cleanly via `pip install dist/tabflows-*.whl`.

## CI/CD Pipeline
- GitHub Actions workflow (`.github/workflows/ci.yml`) runs on `push` and `pull_request` to `main` and `feat/*`.
- Uses `astral-sh/setup-uv@v5` with Python 3.11.
- Steps: `uv sync` -> `ruff check` -> `ty check` -> `pytest` -> `uv build` -> `python scripts/compile_pipelines_ci.py`.
- Pipeline compilation step (`compile_pipelines_ci.py`) verifies pipeline template compilation without needing GCP credentials.

