# tabflows

`tabflows` is a Python library and reference repository for designing, building, and executing AutoML Tabular classification workflows and model inference on Google Cloud Vertex AI Pipelines.

---

## Features

- **Vertex AI Tabular Workflows**: Built on Google-managed AutoML Tabular pipeline components (`google-cloud-pipeline-components>=2.22.0`, `google_cloud_pipeline_components.v1.automl.tabular`).
- **Skip Architecture Search**: Accelerates model tuning by reusing Stage 1 hyperparameter tuning results (`stage_1_tuning_result_artifact_uri`) to reduce execution time and training cost.
- **Online & Batch Inference**: Built-in support for deploying models to real-time Vertex AI Endpoints (`n1-standard-4`), executing online predictions, and launching non-blocking Batch Prediction jobs against GCS datasets.
- **Automatic `.env` Configuration**: Seamless environment configuration powered by `pydantic-settings` (`BaseSettings`), automatically reading `GCP_PROJECT`, `GCP_LOCATION`, `GCP_BUCKET_URI`, and pipeline parameters from `.env`.
- **CLI & Python SDK Interfaces**: Command-line interface and modular Python SDK for provisioning GCS assets, compiling templates, submitting jobs asynchronously (`job.submit()`), and running inference.
- **Modern Python Tooling**: Built with `uv` for dependency management, `ruff` for linting/formatting, `ty` for static type checking, and `pytest` for automated unit testing.

---

## Repository Structure

```text
tabflows/
├── src/tabflows/              # Tabflows Python library
│   ├── config.py              # TabularPipelineConfig schema & .env settings loader
│   ├── pipeline.py            # AutoML Tabular pipeline builders & GCS helpers
│   ├── inference.py           # Online endpoint deployment & batch prediction SDK
│   ├── cli.py                 # Click CLI interface (setup / run-automl / inference)
│   └── __init__.py            # Library exports
├── scripts/                   # Helper scripts
│   └── setup_gcp_resources.py # Standalone GCP bucket & asset setup script
├── notebooks/                 # Jupyter notebook tutorials
│   ├── 01_automl_tabular_classification.ipynb
│   └── 02_automl_tabular_inference.ipynb
├── tests/                     # Unit tests
│   ├── test_config.py         # Config schema & .env tests
│   ├── test_pipeline.py       # Pipeline builder & GCS helper tests
│   └── test_inference.py      # Online & batch inference unit tests
├── docs/notes/                # Persistent technical notes & index (<200 lines)
│   ├── tabular_workflows.md
│   ├── inference.md
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

Use the `tabflows` CLI to compile templates, submit pipeline jobs, or execute model inference. If `--job-id` is omitted, a unique timestamped ID (`automl-tabular-YYYYMMDDHHMMSS`) is generated automatically.

```bash
# Compile standard AutoML Tabular pipeline template locally
uv run tabflows run-automl --compile-only

# Submit standard AutoML Tabular pipeline asynchronously to Vertex AI (Stage 1)
uv run tabflows run-automl --async-mode

# Submit Skip Architecture Search pipeline (Stage 2 - requires Stage 1 tuning result artifact URI)
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
```

### 3. Python SDK

```python
from dotenv import load_dotenv
from tabflows import (
    TabularPipelineConfig,
    cleanup_endpoint,
    deploy_model_to_endpoint,
    predict_online,
    run_batch_prediction,
)

# 1. Load .env environment variables
load_dotenv()
config = TabularPipelineConfig()

# 2. Online Inference (Deploy real-time Endpoint)
model_resource_name = "projects/YOUR_PROJECT/locations/us-central1/models/YOUR_MODEL_ID"
endpoint = deploy_model_to_endpoint(model=model_resource_name, config=config)

# 3. Real-time Prediction
predictions = predict_online(
    endpoint=endpoint,
    instances=[{"age": "35", "job": "technician", "marital": "married"}],
)
print("Online Predictions:", predictions)

# 4. Undeploy & Clean Up Endpoint
cleanup_endpoint(endpoint=endpoint)

# 5. Batch Inference
batch_job = run_batch_prediction(
    model=model_resource_name,
    config=config,
    gcs_source=f"{config.bucket_uri}/test_instances.csv",
)
print("Batch Prediction Job Submitted:", batch_job.resource_name)
```

### 4. Jupyter Notebook Tutorials

Register the project environment as a Jupyter kernel and launch the tutorial notebooks:

```bash
# Register Jupyter kernel
uv run python -m ipykernel install --sys-prefix --name tabflows --display-name "Python 3.11 (tabflows)"

# Launch Classification Training Notebook
uv run jupyter notebook notebooks/01_automl_tabular_classification.ipynb

# Launch Online & Batch Inference Notebook
uv run jupyter notebook notebooks/02_automl_tabular_inference.ipynb
```

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
