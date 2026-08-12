# Gemini Workspace Context - `tabflows`

Always refer to [CODE_STANDARDS.md](file:///usr/local/google/home/jordantotten/antigravity/tabflows/CODE_STANDARDS.md) for coding, environment, linting, testing, and git conventions before writing code or modifying the repository environment.

## Project Overview
`tabflows` is a Python library and repository for designing, building, and executing AutoML Tabular classification workflows on Google Cloud Vertex AI Pipelines.

## Core Rules & Conventions
1. **Environment & Commands**: Always use `uv` (`uv run`, `uv add`, `uv sync`). Never execute bare `pip` or `python` commands.
2. **Code Style**: Format and lint using `ruff`. Check types using `ty`. Run tests using `pytest`.
3. **Git Commits**: Never append `Co-Authored-By` trailers.
4. **Documentation Notes**: Maintain topic notes in `docs/notes/` and keep an up-to-date index in `docs/notes/README.md` (< 200 lines).
