# Complete AutoML Tabular Workflows for Classification Feature Coverage Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Achieve 100% comprehensive coverage of all Vertex AI AutoML Tabular Workflows for Classification features in `tabflows` by adding support for Stratified Data Splitting, Fast Skip-Evaluation Retraining Pipelines (`get_skip_evaluation_pipeline_and_parameters` & `get_distill_skip_evaluation_pipeline_and_parameters`), Custom Ops Model Exports (`export_additional_model_without_custom_ops`), Sample Weighting (`weight_column`), CLI subcommands, and Notebook 10.

**Architecture:** Extend `TabularPipelineConfig`, `pipeline.py`, `cli.py`, add `create_skip_evaluation_pipeline_job`, author tutorial notebook 10, update unit tests, and update repository documentation.

**Tech Stack:** Python 3.11, Vertex AI Tabular Workflows (`google-cloud-pipeline-components>=2.22.0`), `uv`, `pytest`, `ruff`, `ty`.

---

## Proposed Tasks

### Task 1: Core SDK & Config Extensions for Stratified Splits, Skip-Evaluation Pipelines, & Custom Ops Export

**Files:**
- Modify: `src/tabflows/config.py`
- Modify: `src/tabflows/pipeline.py`
- Modify: `src/tabflows/__init__.py`
- Test: `tests/test_config.py`
- Test: `tests/test_pipeline.py`

**Step 1: Write failing unit test**

```python
def test_skip_evaluation_pipeline_creation() -> None:
    config = TabularPipelineConfig(
        project_id="test-proj",
        bucket_uri="gs://test-bkt",
        stratified_split_key="class_col",
        weight_column="sample_weight_col",
        export_additional_model_without_custom_ops=True,
    )
    assert config.stratified_split_key == "class_col"
    assert config.weight_column == "sample_weight_col"
    assert config.export_additional_model_without_custom_ops is True
```

**Step 2: Run test to verify failure**
Run: `uv run pytest tests/test_config.py -k test_skip_evaluation_pipeline_creation`

**Step 3: Implement minimal code**
Update `TabularPipelineConfig` in `src/tabflows/config.py`, add `create_skip_evaluation_pipeline_job` and `create_distill_skip_evaluation_pipeline_job` in `src/tabflows/pipeline.py`, and export in `src/tabflows/__init__.py`.

**Step 4: Run test to verify pass**
Run: `uv run pytest tests/test_pipeline.py`

**Step 5: Commit**
```bash
git add src/tabflows/config.py src/tabflows/pipeline.py src/tabflows/__init__.py tests/test_config.py tests/test_pipeline.py
git commit -m "feat: add stratified split, skip-evaluation pipelines, and no-custom-ops export to SDK"
```

---

### Task 2: CLI Subcommand Options for New Pipeline Variants & Features

**Files:**
- Modify: `src/tabflows/cli.py`
- Test: `tests/test_cli.py`

**Step 1: Write failing unit test**

```python
def test_cli_skip_evaluation_and_stratified_split_flags() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["compile-pipeline", "--stratified-split-key", "class_col", "--skip-evaluation"])
    assert result.exit_code == 0
```

**Step 2: Run test to verify failure**
Run: `uv run pytest tests/test_cli.py -k test_cli_skip_evaluation_and_stratified_split_flags`

**Step 3: Implement minimal code**
Update `tabflows compile-pipeline` and `tabflows run-pipeline` commands in `src/tabflows/cli.py`.

**Step 4: Run test to verify pass**
Run: `uv run pytest tests/test_cli.py -k test_cli_skip_evaluation_and_stratified_split_flags`

**Step 5: Commit**
```bash
git add src/tabflows/cli.py tests/test_cli.py
git commit -m "feat: add CLI options for stratified splits, sample weights, and skip-evaluation pipelines"
```

---

### Task 3: Author Notebook 10 - Complete Feature Showcase

**Files:**
- Create: `notebooks/10_complete_tabular_workflows_feature_showcase.ipynb`

**Step 1: Create notebook file and populate cells**
Author tutorial demonstrating stratified data splits, sample weight columns, fast skip-evaluation pipeline jobs, and no-custom-ops model export.

**Step 2: Execute notebook and save clean cell outputs**
Run execution script to populate notebook 10 cell outputs.

**Step 3: Commit**
```bash
git add notebooks/10_complete_tabular_workflows_feature_showcase.ipynb
git commit -m "feat: add notebook 10 for complete Tabular Workflows feature showcase"
```

---

### Task 4: Repository Documentation & Final Audit

**Files:**
- Modify: `README.md`
- Modify: `docs/notes/README.md`
- Modify: `docs/notes/tabular_workflows.md`

**Step 1: Update README.md and topic notes index**
Document new pipeline functions, stratified splits, sample weights, skip-evaluation pipelines, and notebook 10 summary. Keep `docs/notes/README.md` under 200 lines.

**Step 2: Run full verification suite**
```bash
uv run pytest
uv run ty check src/
uv run ruff check .
uv build
uv run python3 scripts/compile_pipelines_ci.py
```

**Step 3: Commit**
```bash
git add README.md docs/notes/README.md docs/notes/tabular_workflows.md
git commit -m "docs: update README and topic notes index for 100% Tabular Workflows feature coverage"
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
- Verify all 10 notebook tutorials (01 through 10) execute without error.
