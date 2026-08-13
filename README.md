# tabflows

`tabflows` is a Python library and reference repository for designing, building, and executing AutoML Tabular classification workflows and model inference on Google Cloud Vertex AI Pipelines.

---

## Features

- **Vertex AI Tabular Workflows**: Built on Google-managed AutoML Tabular pipeline components (`google-cloud-pipeline-components>=2.22.0`, `google_cloud_pipeline_components.v1.automl.tabular`).
- **Skip Architecture Search**: Accelerates model tuning by reusing Stage 1 hyperparameter tuning results (`stage_1_tuning_result_artifact_uri`) to reduce execution time by ~79.6% and training node-hour costs by ~80.4%.
- **Feature Transform Engine (FTE)**: Supports automated and explicit column-level transformations across `categorical`, `numeric`, `timestamp`, `text_embedding` (`text`), and `auto` data types.
- **Model Distillation**: Supports producing lighter-weight, lower-latency distilled models alongside standard ensemble models.
- **Vertex AI Experiments**: Full-lifecycle experiment tracking of parameters, evaluation metrics, pipeline jobs (`create_tabular_pipeline_job`, `run_skip_architecture_search_pipeline`), DoE campaigns (`run_doe_campaign`), model evaluation metrics (`get_model_evaluation_metrics`), endpoint deployments (`deploy_model_to_endpoint`), and batch prediction jobs (`run_batch_prediction`).
- **Online & Batch Inference**: Built-in support for deploying models to real-time Vertex AI Endpoints (`n1-standard-4`), executing online predictions, and launching non-blocking Batch Prediction jobs against GCS datasets.
- **Automatic `.env` Configuration**: Seamless environment configuration powered by `pydantic-settings` (`BaseSettings`), automatically reading `GCP_PROJECT`, `GCP_LOCATION`, `GCP_BUCKET_URI`, and pipeline parameters from `.env`.
- **CLI & Python SDK Interfaces**: Command-line interface and modular Python SDK for provisioning GCS assets, compiling templates, submitting jobs asynchronously (`job.submit()`), logging experiments (`log-experiment`), and running inference.
- **Modern Python Tooling**: Built with `uv` for dependency management, `ruff` for linting/formatting, `ty` for static type checking, and `pytest` for automated unit testing.

---

## Repository Structure

```text
tabflows/
├── src/tabflows/              # Tabflows Python library
│   ├── config.py              # TabularPipelineConfig schema & .env settings loader
│   ├── pipeline.py            # AutoML Tabular pipeline builders, FTE, & GCS helpers
│   ├── inference.py           # Online endpoint deployment & batch prediction SDK
│   ├── experiments.py         # Model evaluation & Vertex AI Experiment tracking
│   ├── cli.py                 # Click CLI interface (setup / run-automl / inference / fte)
│   └── __init__.py            # Library exports
├── scripts/                   # Helper scripts
│   └── setup_gcp_resources.py # Standalone GCP bucket & asset setup script
├── notebooks/                 # Jupyter notebook tutorials
│   ├── 01_automl_tabular_classification.ipynb
│   ├── 02_automl_tabular_inference.ipynb
│   ├── 03_automl_tabular_evaluation_and_experiments.ipynb
│   ├── 04_skip_architecture_search_benchmarks.ipynb
│   └── 05_feature_transform_engine.ipynb
├── tests/                     # Unit tests
│   ├── test_cli.py            # CLI integration tests
│   ├── test_config.py         # Config schema & .env tests
│   ├── test_experiments.py    # Experiment tracking & metrics tests
│   ├── test_inference.py      # Online & batch inference unit tests
│   └── test_pipeline.py       # Pipeline builder & FTE unit tests
├── docs/notes/                # Persistent technical notes & index (<200 lines)
│   ├── tabular_workflows.md
│   ├── feature_transform_engine.md
│   ├── skip_architecture_search_benchmarks.md
│   ├── inference.md
│   ├── evaluation_and_experiments.md
│   └── tooling.md
├── CODE_STANDARDS.md          # Code & engineering standards
├── GEMINI.md                  # AI workspace context
├── Makefile                   # Standard developer tasks (dev, lint, format, test)
└── pyproject.toml             # uv package definition & tool configs
```

---

## Getting Started

### Prerequisites

- **Python**: `>=3.11`
- **uv**: Installed ([https://github.com/astral-sh/uv](https://github.com/astral-sh/uv))
- **Google Cloud SDK**: Authenticated via Application Default Credentials (`gcloud auth application-default login`) with permissions on your GCP project.

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/tottenjordan/tabflows.git
   cd tabflows
   ```

2. **Install dependencies and setup environment**:
   ```bash
   uv sync --all-groups
   ```

3. **Configure Environment Variables**:
   Copy `.env.example` to `.env` and fill in your GCP project and GCS bucket settings:
   ```bash
   cp .env.example .env
   ```

   Example `.env`:
   ```ini
   GCP_PROJECT=your-gcp-project-id
   GCP_LOCATION=us-central1
   GCP_BUCKET_URI=gs://your-bucket-name
   PIPELINE_ROOT_DIR_NAME=automl_tabular_pipeline
   TARGET_COLUMN=deposit
   PREDICTION_TYPE=classification
   OPTIMIZATION_OBJECTIVE=minimize-log-loss
   SERVING_MACHINE_TYPE=n1-standard-4
   ```

---

## Usage

### 1. Provision GCP Resources

Provision the Cloud Storage bucket (if missing) and upload the initial feature transformation config JSON (`transform_config_unique.json`):

```bash
# Using CLI (reads project and bucket settings from .env automatically)
uv run tabflows setup

# Or using the standalone Python script
uv run python scripts/setup_gcp_resources.py
```

### 2. Command Line Interface (CLI)

Use the `tabflows` CLI to compile templates, submit pipeline jobs, generate FTE configurations, log experiments, or execute model inference. If `--job-id` is omitted, a unique timestamped ID (`automl-tabular-YYYYMMDDHHMMSS`) is generated automatically.

```bash
# Compile standard AutoML Tabular pipeline template locally
uv run tabflows run-automl --compile-only

# Submit standard AutoML Tabular pipeline asynchronously with model distillation enabled
uv run tabflows run-automl --distill --async-mode

# Generate Feature Transform Engine (FTE) JSON configuration
uv run tabflows generate-fte-config \
    --output-path "./transform_config_fte.json" \
    --columns-json '{"age": "numeric", "job": "categorical", "signup_date": "timestamp", "notes": "text"}'

# Submit Skip Architecture Search pipeline (Stage 2 - reusing Stage 1 tuning result artifact URI)
uv run tabflows run-skip-search \
    --tuning-artifact-uri "gs://your-bucket-name/automl_tabular_pipeline/stage1_job_id/automl-tabular-stage-1-tuner/tuning_result_output" \
    --async-mode

# Execute real-time Online Inference
uv run tabflows predict-online \
    --endpoint "projects/.../endpoints/ENDPOINT_ID" \
    --json-instance '{"age": "35", "job": "technician", "marital": "married"}'

# Submit Batch Prediction job against a GCS dataset
uv run tabflows run-batch-predict \
    --model "projects/.../models/MODEL_ID" \
    --gcs-source "gs://your-bucket-name/test_instances.csv"

# Fetch Model Evaluation metrics and feature importance
uv run tabflows get-evaluation --model "projects/.../models/MODEL_ID"

# Log custom experiment run with parameters, metrics, model evaluations, and pipeline job reference
uv run tabflows log-experiment \
    --run-name "my-experiment-run" \
    --model-resource-name "projects/.../models/MODEL_ID" \
    --pipeline-job-id "projects/.../pipelineJobs/JOB_ID" \
    --params-json '{"train_budget_milli_node_hours": 1000, "run_distillation": true}' \
    --metrics-json '{"custom_eval_score": 0.942}'

# List Vertex AI Experiment runs and side-by-side metrics
uv run tabflows list-experiments
```

### 3. Python SDK

```python
from dotenv import load_dotenv
from tabflows import (
    TabularPipelineConfig,
    create_tabular_pipeline_job,
    deploy_model_to_endpoint,
    generate_fte_transformations,
    get_model_evaluation_metrics,
    get_model_feature_attributions,
    list_experiment_runs,
    log_experiment_run,
    predict_online,
    run_batch_prediction,
    run_doe_campaign,
    run_skip_architecture_search_pipeline,
)

# 1. Load .env environment variables and configuration
load_dotenv()
config = TabularPipelineConfig(run_distillation=True)

# 2. Generate Feature Transform Engine (FTE) column transformations
fte_transformations = generate_fte_transformations(
    {
        "age": "numeric",
        "job": "categorical",
        "signup_date": "timestamp",
        "notes": "text",
    }
)

# 3. Create & Submit Stage 1 Pipeline (Full Search with automatic experiment tracking)
stage1_job = create_tabular_pipeline_job(
    config=config,
    job_id="stage1-full-search-run",
    log_experiment=True,
)

# 4. Create & Submit Stage 2 Pipeline (Skip Architecture Search with experiment tracking)
stage2_job = run_skip_architecture_search_pipeline(
    config=config,
    tuning_result_artifact_uri="gs://your-bucket/stage1_job/automl-tabular-stage-1-tuner/tuning_result_output",
    job_id="stage2-skip-search-run",
    log_experiment=True,
)

# 5. Execute Design of Experiments (DoE) Campaign
campaign_results = run_doe_campaign(
    campaign_name="hyperparameter-grid",
    variants=[
        {"name": "v1", "train_budget_milli_node_hours": 1000},
        {"name": "v2", "train_budget_milli_node_hours": 2000},
    ],
    config=config,
)

# 6. Inspect Model Evaluation Metrics & Feature Importance
model_resource_name = "projects/YOUR_PROJECT/locations/us-central1/models/YOUR_MODEL_ID"
metrics = get_model_evaluation_metrics(model=model_resource_name, config=config)
attributions = get_model_feature_attributions(model=model_resource_name, config=config)
print("Log Loss:", metrics.get("logLoss"))
print("ROC AUC:", metrics.get("auRoc"))

# 7. Online Inference (Deploy real-time Endpoint & Predict with experiment tracking)
endpoint = deploy_model_to_endpoint(
    model=model_resource_name,
    config=config,
    log_experiment=True,
)
predictions = predict_online(
    endpoint=endpoint,
    instances=[{"age": "35", "job": "technician", "marital": "married"}],
)
print("Online Predictions:", predictions)

# 8. Batch Inference (with automatic experiment logging)
batch_job = run_batch_prediction(
    model=model_resource_name,
    config=config,
    gcs_source=f"{config.bucket_uri}/test_instances.csv",
    log_experiment=True,
)

# 9. Custom Experiment Run Logging
log_experiment_run(
    run_name="custom-evaluation-run",
    params={"custom_param": "value"},
    metrics={"custom_metric": 0.98},
    model=model_resource_name,
    config=config,
)

# 10. Experiment Tracking (Compare Runs DataFrame)
df = list_experiment_runs(config=config)
print(df)
```

### 4. Jupyter Notebook Tutorials

Register the project environment as a Jupyter kernel and launch the tutorial notebooks:

```bash
# Register Jupyter kernel
uv run python -m ipykernel install --sys-prefix --name tabflows --display-name "Python 3.11 (tabflows)"

# 01. Classification Training Notebook
uv run jupyter notebook notebooks/01_automl_tabular_classification.ipynb

# 02. Online & Batch Inference Notebook
uv run jupyter notebook notebooks/02_automl_tabular_inference.ipynb

# 03. Model Evaluation & Experiment Tracking Notebook
uv run jupyter notebook notebooks/03_automl_tabular_evaluation_and_experiments.ipynb

# 04. Skip Architecture Search & Benchmarks Notebook
uv run jupyter notebook notebooks/04_skip_architecture_search_benchmarks.ipynb

# 05. Feature Transform Engine (FTE) Notebook
uv run jupyter notebook notebooks/05_feature_transform_engine.ipynb
```

#### Notebook Summaries
- **Notebook 01 (`01_automl_tabular_classification.ipynb`)**: End-to-end AutoML Tabular classification pipeline setup, parameter configuration, template compilation, and pipeline job submission (`create_tabular_pipeline_job`) with automatic Vertex AI Experiment tracking.
- **Notebook 02 (`02_automl_tabular_inference.ipynb`)**: Model deployment to real-time Vertex AI Endpoints (`deploy_model_to_endpoint`), online JSON prediction execution (`predict_online`), non-blocking batch prediction jobs (`run_batch_prediction`), and endpoint cleanup, complete with deployment experiment logging.
- **Notebook 03 (`03_automl_tabular_evaluation_and_experiments.ipynb`)**: Comprehensive model evaluation metrics (Log Loss, PR/ROC AUC, confusion matrices), global feature attributions (`get_model_feature_attributions`), DoE campaign execution (`run_doe_campaign`), custom experiment run logging (`log_experiment_run`), and side-by-side run comparisons (`list_experiment_runs`).
- **Notebook 04 (`04_skip_architecture_search_benchmarks.ipynb`)**: Skip Architecture Search Stage 2 execution (`run_skip_architecture_search_pipeline`) reusing Stage 1 tuning result artifacts (`tuning_result_output`), performance benchmarks (-79.6% execution time, -80.4% node-hour savings), and experiment run comparisons.
- **Notebook 05 (`05_feature_transform_engine.ipynb`)**: Feature Transform Engine (FTE) configuration (`generate_fte_transformations`), column data type mappings (`categorical`, `numeric`, `timestamp`, `text_embedding`, `auto`), GCS transform asset management, and pipeline integration.

---

## Development & Testing

Run all quality checks and tests using `make` or `uv`:

```bash
# Install all dependency groups
make dev

# Run code formatting checks (ruff format)
make format

# Run code linter (ruff check) and static type checker (ty check src/)
make lint

# Run unit tests with coverage reporting (pytest)
make test
```

---

## Code Standards

All development strictly follows the rules outlined in [CODE_STANDARDS.md](CODE_STANDARDS.md):
- **`uv`**: Mandatory for dependency & environment execution (`uv run`, `uv add`, `uv sync`). Never use bare `pip` or `python`.
- **`ruff`**: Exclusive tool for linting and code formatting (`line-length = 100`).
- **`ty`**: Type checker for Python code.
- **`pytest`**: Test framework.
- **Git**: Never include `Co-Authored-By` trailers in commit messages or pull requests.
