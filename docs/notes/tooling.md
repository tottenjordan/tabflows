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
