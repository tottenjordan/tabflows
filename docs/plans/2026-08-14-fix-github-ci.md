# Fix GitHub Actions CI Workflow Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix failing GitHub Actions CI checks by expanding branch trigger globs (`feat/**`), quoting workflow `on:` key, adding default CI environment variables (`GCP_PROJECT`, `GCP_LOCATION`, `GCP_BUCKET_URI`), setting job permissions, and hardening deterministic dependency synchronization.

**Architecture:** Update `.github/workflows/ci.yml` and `scripts/compile_pipelines_ci.py`, add unit tests for workflow parsing, and verify local execution across `pytest`, `ruff`, `ty`, `uv build`, and pipeline compilation.

**Tech Stack:** GitHub Actions, Python 3.11, PyYAML, `uv`, `pytest`, `ruff`, `ty`.

---

## Proposed Tasks

### Task 1: Harden `.github/workflows/ci.yml` Configuration

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `scripts/compile_pipelines_ci.py`
- Test: `tests/test_ci_scripts.py`

**Step 1: Write failing unit test**

```python
import yaml

def test_github_ci_yaml_structure() -> None:
    workflow_path = project_root / ".github" / "workflows" / "ci.yml"
    with open(workflow_path) as f:
        data = yaml.safe_load(f)
    assert "on" in data or "push" in str(data)
    assert "feat/**" in data["on"]["push"]["branches"] or "feat/**" in data[True]["push"]["branches"]
```

**Step 2: Run test to verify failure**
Run: `uv run pytest tests/test_ci_scripts.py -k test_github_ci_yaml_structure`

**Step 3: Update `.github/workflows/ci.yml`**
Quote `"on":`, add `feat/**` branch patterns, specify `env:` block, and set explicit job permissions (`permissions: contents: read`).

**Step 4: Run test to verify pass**
Run: `uv run pytest tests/test_ci_scripts.py -k test_github_ci_yaml_structure`

**Step 5: Commit**
```bash
git add .github/workflows/ci.yml scripts/compile_pipelines_ci.py tests/test_ci_scripts.py
git commit -m "ci: fix GitHub Actions workflow triggers, environment defaults, and YAML syntax"
```

---

### Task 2: End-to-End Verification & Pull Request Push

**Files:**
- Modify: `README.md`

**Step 1: Run full quality and compilation suite**
```bash
uv run pytest
uv run ty check src/
uv run ruff check .
uv build
uv run python3 scripts/compile_pipelines_ci.py
```

**Step 2: Push changes to remote origin**
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
- Confirm GitHub Actions runner status on GitHub PR #4 shows green passing checks.
