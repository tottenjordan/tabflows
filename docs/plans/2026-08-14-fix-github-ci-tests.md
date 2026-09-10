# Fix GitHub Actions CI Tests Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ensure 100% green passing status for GitHub Actions CI workflow runs on PR #4 and `main` by configuring deterministic `--frozen` lockfile installation, adding `enable-cache: true`, setting global CI environment fallbacks (`GCP_PROJECT`, `GCP_LOCATION`, `GCP_BUCKET_URI`, `PYTHONUNBUFFERED`), and hardening pipeline compilation scripts.

**Architecture:** Refine `.github/workflows/ci.yml` step options, update `scripts/compile_pipelines_ci.py` to handle missing credential environments, add unit tests in `tests/test_ci_scripts.py`, and push updated branch to origin.

**Tech Stack:** GitHub Actions (`astral-sh/setup-uv@v5`), Python 3.11, `uv`, `pytest`, `ruff`, `ty`.

---

## Proposed Tasks

### Task 1: Refine `.github/workflows/ci.yml` & Pipeline Compiler

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `scripts/compile_pipelines_ci.py`
- Test: `tests/test_ci_scripts.py`

**Step 1: Write failing unit test**

```python
def test_compile_pipelines_ci_with_frozen_env():
    # Verify compile_pipelines_ci runs cleanly under unbuffered mock env
    pass
```

**Step 2: Run test to verify failure/behavior**
Run: `uv run pytest tests/test_ci_scripts.py`

**Step 3: Update `.github/workflows/ci.yml` & `scripts/compile_pipelines_ci.py`**
Add top-level `env:`, `enable-cache: true`, and `--frozen` flag to `uv sync`.

**Step 4: Run full test suite to verify pass**
Run: `uv run pytest`

**Step 5: Commit**
```bash
git add .github/workflows/ci.yml scripts/compile_pipelines_ci.py tests/test_ci_scripts.py
git commit -m "ci: enforce frozen lockfile sync and caching in GitHub Actions workflow"
```

---

### Task 2: Push to Origin and Verify Remote PR Status

**Files:**
- Modify: `README.md`

**Step 1: Run local verification**
```bash
uv run pytest
uv run ty check src/
uv run ruff check .
uv build
uv run python3 scripts/compile_pipelines_ci.py
```

**Step 2: Push branch to remote origin**
```bash
git push origin feat/deployment-cicd-packaging
```

---

## Verification Plan

### Automated Tests
```bash
uv run pytest
uv run ty check src/
uv run ruff check .
uv build
uv run python3 scripts/compile_pipelines_ci.py
```

### Manual Verification
- Verify green check status on GitHub PR #4 on GitHub web interface.
