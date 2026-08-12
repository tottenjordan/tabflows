# Code Standards & Engineering Guidelines

This document details the mandatory standards and practices for the `tabflows` repository. All contributors and AI assistants MUST strictly adhere to these rules.

## Git & Commits
- **No Co-Authored-By Trailers**: Never add `Co-Authored-By` trailers or metadata when creating git commits or submitting pull requests.

## Python Development & Environment Management
- **Tooling Engine**: Use [`uv`](https://github.com/astral-sh/uv) for all dependency management, package execution, and environment setup.
  - **Never** use bare `pip`, `pip3`, `python`, `python3`, `virtualenv`, or manual `source .venv/bin/activate`.
  - Add dependencies via `uv add <package>` (or `uv add --group dev <package>`).
  - Execute scripts and tools via `uv run <command>`.
- **Linting & Formatting**:
  - Use `ruff` exclusively for linting and code formatting (`uv run ruff check .` and `uv run ruff format .`).
  - Do NOT use legacy tools like `black`, `flake8`, or `isort`.
- **Type Checking**:
  - Use `ty` for fast static type checking (`uv run ty check src/`).
  - Do NOT use `mypy` or `pyright`.
- **Testing**:
  - Use `pytest` for all unit and integration tests (`uv run pytest`).

## Code Structure & Quality
- Target Python version: `>=3.11`.
- Follow modern Python conventions, PEP 604 type union syntax (`X | Y`), standard type annotations (`list[str]`, `dict[str, Any]`), and Pydantic models for structured configuration schemas.
- Maintain clean function documentation and avoid dead or commented-out code.
