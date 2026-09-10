# Fix GCP Project & Auth GoogleAuthError in CI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the 7 failing GitHub CI tests (`GoogleAuthError: Unable to find your project`) by explicitly passing `project=config.project_id` into `aiplatform.PipelineJob` instantiations in `src/tabflows/pipeline.py` and setting `GOOGLE_CLOUD_PROJECT` & `CLOUD_ML_PROJECT_ID` environment variables in `.github/workflows/ci.yml`.

**Architecture:** Pass `project=config.project_id` in `create_tabular_pipeline_job` and `run_skip_architecture_search_pipeline` in `src/tabflows/pipeline.py`, update `.github/workflows/ci.yml` `env:` block with `GOOGLE_CLOUD_PROJECT`, add unit tests in `tests/test_pipeline.py`, and push to origin.

**Tech Stack:** Python 3.11, Vertex AI (`google.cloud.aiplatform`), `google.auth`, `pytest`, `uv`, GitHub Actions.

---

## Root Cause Analysis

In `src/tabflows/pipeline.py` (lines 260 and 299), `aiplatform.PipelineJob` was initialized with `location=config.location` but omitted `project=config.project_id`. When GCP Application Default Credentials (ADC) are not configured (as on GitHub Actions runners), `aiplatform.PipelineJob` triggers `google.auth.default()` to resolve project ID, raising `GoogleAuthError: Unable to find your project`.

---

## Proposed Tasks

### Task 1: Explicitly Pass `project=config.project_id` in PipelineJob Instantiations

**Files:**
- Modify: `src/tabflows/pipeline.py`
- Modify: `.github/workflows/ci.yml`
- Test: `tests/test_pipeline.py`

**Step 1: Write failing unit test**

```python
def test_create_pipeline_job_explicit_project_param(monkeypatch):
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("GCP_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    
    config = TabularPipelineConfig(project_id="explicit-test-proj", bucket_uri="gs://test-bkt")
    job = create_tabular_pipeline_job(config, log_experiment=False)
    assert job.project == "explicit-test-proj"
```

**Step 2: Run test to verify failure**
Run: `uv run pytest tests/test_pipeline.py -k test_create_pipeline_job_explicit_project_param`

**Step 3: Update `src/tabflows/pipeline.py` & `.github/workflows/ci.yml`**
Add `project=config.project_id` to both `aiplatform.PipelineJob(...)` instantiations in `pipeline.py`, and add `GOOGLE_CLOUD_PROJECT` & `CLOUD_ML_PROJECT_ID` to `.github/workflows/ci.yml`.

**Step 4: Run test to verify pass**
Run: `uv run pytest tests/test_pipeline.py -k test_create_pipeline_job_explicit_project_param`

**Step 5: Commit**
```bash
git add src/tabflows/pipeline.py .github/workflows/ci.yml tests/test_pipeline.py
git commit -m "fix: explicitly pass project=config.project_id to PipelineJob to prevent GoogleAuthError in unauthenticated environments"
```

---

### Task 2: Push to Origin and Verify Remote CI Checks

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
- Check GitHub PR #4 on GitHub web interface or `gh run list` to verify 100% green passing status.
