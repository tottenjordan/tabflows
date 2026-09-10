# Bake-off Live Deployment, CI/CD Pipeline, & Package Distribution Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** (1) Implement live model deployment & online latency benchmarking for the completed Bake-off DOE models on Vertex AI, (2) Create GitHub Actions CI workflow (`.github/workflows/ci.yml`), and (3) Configure `tabflows` package building & distribution (`uv build`, `.whl` artifacts).

**Architecture:** Add deployment benchmark script & tutorial notebook 09, configure GitHub Actions workflow for automated testing/linting/typing/pipeline compilation, and update `pyproject.toml` for standard Python package distribution via `uv`.

**Tech Stack:** Python 3.11, Vertex AI Endpoints (`aiplatform.Endpoint`), GitHub Actions (`.github/workflows/ci.yml`), `uv`, `hatchling`, `pytest`, `ruff`, `ty`.

---

## Proposed Tasks

### Task 1: Package Build System & PyPI Metadata Configuration

**Files:**
- Modify: `pyproject.toml`
- Test: `tests/test_package_build.py`

**Step 1: Write failing unit test**

```python
import subprocess


def test_uv_build_package() -> None:
    result = subprocess.run(["uv", "build"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Successfully built" in result.stdout or "dist/" in result.stdout
```

**Step 2: Run test to verify failure**
Run: `uv run pytest tests/test_package_build.py`

**Step 3: Update `pyproject.toml`**
Configure package metadata (`authors`, `readme`, `license`, `classifiers`, `urls`) and build system.

**Step 4: Run test to verify pass**
Run: `uv run pytest tests/test_package_build.py`

**Step 5: Commit**
```bash
git add pyproject.toml tests/test_package_build.py
git commit -m "feat: configure package build metadata and add uv build test"
```

---

### Task 2: GitHub Actions CI/CD Workflow (`.github/workflows/ci.yml`)

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `scripts/compile_pipelines_ci.py`

**Step 1: Create pipeline compilation check script**
Create `scripts/compile_pipelines_ci.py` that verifies `create_tabular_pipeline_job` and `run_skip_architecture_search_pipeline` generate valid Kubeflow JSON specs.

**Step 2: Create GitHub Actions workflow file**
Create `.github/workflows/ci.yml` configuring `astral-sh/setup-uv`, running `uv sync`, `ruff check .`, `ty check src/`, `pytest`, and `python scripts/compile_pipelines_ci.py`.

**Step 3: Run pipeline compilation check script locally**
Run: `uv run python3 scripts/compile_pipelines_ci.py`
Expected: `Pipeline compilation checks passed!`

**Step 4: Commit**
```bash
git add .github/workflows/ci.yml scripts/compile_pipelines_ci.py
git commit -m "ci: add GitHub Actions workflow for automated linting, testing, and pipeline compilation"
```

---

### Task 3: Bake-off Live Deployment & Latency Benchmarking Script

**Files:**
- Create: `scripts/benchmark_bakeoff_deployment.py`
- Test: `tests/test_benchmark_script.py`

**Step 1: Create unit test for benchmark script logic**
Add test in `tests/test_benchmark_script.py` for request latency statistics calculation (p50, p90, p95).

**Step 2: Implement benchmark script**
Create `scripts/benchmark_bakeoff_deployment.py` using `deploy_model_to_endpoint`, `predict_online`, and `cleanup_endpoint` with latency metrics reporting.

**Step 3: Run unit test**
Run: `uv run pytest tests/test_benchmark_script.py`

**Step 4: Commit**
```bash
git add scripts/benchmark_bakeoff_deployment.py tests/test_benchmark_script.py
git commit -m "feat: add Bake-off live deployment and latency benchmarking script"
```

---

### Task 4: Author Notebook 09 - Bake-off Live Deployment & Benchmarking

**Files:**
- Create: `notebooks/09_bakeoff_live_deployment_and_benchmarking.ipynb`

**Step 1: Create notebook file and populate cells**
Author notebook demonstrating retrieving Bake-off DOE models, deploying to Endpoint with 90%/10% traffic split, profiling online latency, and cleaning up.

**Step 2: Execute notebook and save clean outputs**
Run execution script to populate notebook 09 cell outputs.

**Step 3: Commit**
```bash
git add notebooks/09_bakeoff_live_deployment_and_benchmarking.ipynb
git commit -m "feat: add notebook 09 for Bake-off live deployment and online latency benchmarking"
```

---

### Task 5: Repository Documentation & Final Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/notes/README.md`

**Step 1: Update README.md and topic notes**
Document package installation via wheel (`pip install dist/tabflows-*.whl`), CI/CD workflow, benchmark scripts, and notebook 09 summary.

**Step 2: Run full test and lint suite**
```bash
uv run pytest
uv run ty check src/
uv run ruff check .
uv build
```

**Step 3: Commit**
```bash
git add README.md docs/notes/README.md
git commit -m "docs: update README and topic notes index for packaging, CI/CD, and notebook 09"
```

---

## Verification Plan

### Automated Tests
```bash
uv run pytest
uv run ty check src/
uv run ruff check .
uv build
```

### Manual Verification
- Verify `dist/tabflows-*.whl` and `dist/tabflows-*.tar.gz` are generated cleanly by `uv build`.
- Verify GitHub Actions workflow syntax using `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"`.
