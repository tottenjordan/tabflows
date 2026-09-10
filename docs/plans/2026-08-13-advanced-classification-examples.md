# Advanced Classification Examples Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create 3 advanced tutorial notebooks (Notebooks 06, 07, 08) demonstrating BigQuery direct table ingestion with time-based splits, Champion vs. Challenger endpoint traffic splitting (90%/10%), and specialized optimization objectives (precision-at-recall / recall-at-precision), along with supporting SDK and CLI extensions.

**Architecture:** Extend `TabularPipelineConfig`, `pipeline.py`, `inference.py`, and `cli.py` to support BigQuery source paths, predefined split keys, and multi-model endpoint traffic splitting. Author 3 clean, fully executable Jupyter notebooks with complete outputs, unit tests, and updated repository documentation.

**Tech Stack:** Python 3.11, Vertex AI Pipelines (`google-cloud-pipeline-components>=2.22.0`), Vertex AI Endpoints (`aiplatform.Endpoint`), BigQuery, `uv`, `pytest`, `ruff`, `ty`.

---

## Proposed Tasks

### Task 1: SDK & Config Support for BigQuery Tables & Predefined Split Keys

**Files:**
- Modify: `src/tabflows/config.py`
- Modify: `src/tabflows/pipeline.py`
- Test: `tests/test_pipeline.py`

**Step 1: Write failing unit test**

```python
def test_bigquery_table_and_split_key_config() -> None:
    config = TabularPipelineConfig(
        bigquery_table_path="bq://hybrid-vertex.bank_dataset.train",
        predefined_split_key="split_col",
    )
    assert config.bigquery_table_path == "bq://hybrid-vertex.bank_dataset.train"
    assert config.predefined_split_key == "split_col"
```

**Step 2: Run test to verify failure**
Run: `uv run pytest tests/test_pipeline.py -k test_bigquery_table_and_split_key_config`

**Step 3: Implement minimal code**
Update `TabularPipelineConfig` in `src/tabflows/config.py` and parameter mapping in `src/tabflows/pipeline.py`.

**Step 4: Run test to verify pass**
Run: `uv run pytest tests/test_pipeline.py -k test_bigquery_table_and_split_key_config`

**Step 5: Commit**
```bash
git add src/tabflows/config.py src/tabflows/pipeline.py tests/test_pipeline.py
git commit -m "feat: add BigQuery table path and predefined split key support to TabularPipelineConfig"
```

---

### Task 2: SDK & CLI Support for Champion vs. Challenger Endpoint Traffic Splitting

**Files:**
- Modify: `src/tabflows/inference.py`
- Modify: `src/tabflows/cli.py`
- Test: `tests/test_inference.py`
- Test: `tests/test_cli.py`

**Step 1: Write failing unit test**

```python
def test_deploy_model_with_traffic_split() -> None:
    # Test passing traffic_split dict to endpoint deployment
    pass
```

**Step 2: Run test to verify failure**
Run: `uv run pytest tests/test_inference.py -k test_deploy_model_with_traffic_split`

**Step 3: Implement minimal code**
Update `deploy_model_to_endpoint` in `src/tabflows/inference.py` to accept `traffic_split: dict[str, int] | None = None` and add `--traffic-split-json` to `cli.py`.

**Step 4: Run test to verify pass**
Run: `uv run pytest tests/test_inference.py -k test_deploy_model_with_traffic_split`

**Step 5: Commit**
```bash
git add src/tabflows/inference.py src/tabflows/cli.py tests/test_inference.py tests/test_cli.py
git commit -m "feat: add traffic split deployment support to inference engine and CLI"
```

---

### Task 3: Author Notebook 06 - BigQuery Direct Ingestion & Time-Based Split Strategies

**Files:**
- Create: `notebooks/06_bigquery_and_split_strategies.ipynb`

**Step 1: Create notebook file and populate cells**
Demonstrate querying BigQuery tables, configuring `predefined_split_key` for time-series splits, and compiling pipeline jobs.

**Step 2: Execute notebook and populate clean cell outputs**
Run script to execute notebook 06 and store cell outputs.

**Step 3: Commit**
```bash
git add notebooks/06_bigquery_and_split_strategies.ipynb
git commit -m "feat: add notebook 06 for BigQuery direct ingestion and time-based split strategies"
```

---

### Task 4: Author Notebook 07 - Champion vs. Challenger Traffic Splitting

**Files:**
- Create: `notebooks/07_champion_challenger_traffic_split.ipynb`

**Step 1: Create notebook file and populate cells**
Demonstrate side-by-side deployment of Champion (Baseline Ensemble) and Challenger (Distilled Student) models with 90%/10% traffic split.

**Step 2: Execute notebook and populate clean cell outputs**
Run script to execute notebook 07 and store cell outputs.

**Step 3: Commit**
```bash
git add notebooks/07_champion_challenger_traffic_split.ipynb
git commit -m "feat: add notebook 07 for Champion vs Challenger endpoint traffic splitting"
```

---

### Task 5: Author Notebook 08 - Specialized Optimization Objectives & Target Tuning

**Files:**
- Create: `notebooks/08_optimization_objectives_and_targets.ipynb`

**Step 1: Create notebook file and populate cells**
Demonstrate `maximize-precision-at-recall` and `maximize-recall-at-precision` objectives for fraud detection and marketing conversion datasets.

**Step 2: Execute notebook and populate clean cell outputs**
Run script to execute notebook 08 and store cell outputs.

**Step 3: Commit**
```bash
git add notebooks/08_optimization_objectives_and_targets.ipynb
git commit -m "feat: add notebook 08 for specialized optimization objectives and precision-recall target tuning"
```

---

### Task 6: Repository Documentation & Test Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/notes/README.md`

**Step 1: Update README.md and topic notes index**
Document notebooks 06, 07, and 08 in `README.md` and keep `docs/notes/README.md` under 200 lines.

**Step 2: Run full test and lint suite**
```bash
uv run pytest
uv run ty check src/
uv run ruff check .
```

**Step 3: Commit**
```bash
git add README.md docs/notes/README.md
git commit -m "docs: update README and topic notes index for notebooks 06-08"
```

---

## Verification Plan

### Automated Tests
```bash
uv run pytest
uv run ty check src/
uv run ruff check .
```

### Manual Verification
- Execute all notebooks 01 through 08 using Python validation script to confirm zero cell errors.
