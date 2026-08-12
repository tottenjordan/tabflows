# tabflows

`tabflows` is a Python library and reference repository for designing, building, and executing AutoML Tabular classification workflows on Google Cloud Vertex AI Pipelines.

---

## Features

- **Vertex AI Tabular Workflows**: Built on Google-managed AutoML Tabular pipeline components (`google-cloud-pipeline-components>=2.22.0`, `google_cloud_pipeline_components.v1.automl.tabular`).
- **Skip Architecture Search**: Accelerates model tuning by reusing Stage 1 hyperparameter tuning results (`stage_1_tuning_result_artifact_uri`) to reduce execution time and training cost.
- **Automatic `.env` Configuration**: Seamless environment configuration powered by `pydantic-settings` (`BaseSettings`), automatically reading `GCP_PROJECT`, `GCP_LOCATION`, `GCP_BUCKET_URI`, and pipeline parameters from `.env`.
- **CLI & Python SDK Interfaces**: Command-line interface and modular Python SDK for provisioning GCS assets, compiling templates, and submitting jobs asynchronously (`job.submit()`) or synchronously (`job.run()`).
- **Modern Python Tooling**: Built with `uv` for dependency management, `ruff` for linting/formatting, `ty` for static type checking, and `pytest` for automated unit testing.

---

## Repository Structure

```text
tabflows/
├── src/tabflows/              # Tabflows Python library
│   ├── config.py              # TabularPipelineConfig schema & .env settings loader
│   ├── pipeline.py            # AutoML Tabular pipeline builders & GCS helpers
│   ├── cli.py                 # Click CLI interface (tabflows setup / run-automl / run-skip-search)
│   └── __init__.py            # Library exports
├── scripts/                   # Helper scripts
│   └── setup_gcp_resources.py # Standalone GCP bucket & asset setup script
├── notebooks/                 # Jupyter notebook tutorials
│   └── 01_automl_tabular_classification.ipynb
├── tests/                     # Unit tests
│   ├── test_config.py         # Config schema & .env tests
│   └── test_pipeline.py       # Pipeline builder & GCS helper tests
├── docs/notes/                # Persistent technical notes & index (<200 lines)
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

Use the `tabflows` CLI to compile templates or submit jobs to Vertex AI Pipelines. If `--job-id` is omitted, a unique timestamped ID (`automl-tabular-YYYYMMDDHHMMSS`) is generated automatically.

```bash
# Compile standard AutoML Tabular pipeline template locally
uv run tabflows run-automl --compile-only

# Submit standard AutoML Tabular pipeline asynchronously to Vertex AI
uv run tabflows run-automl --async-mode

# Submit Skip Architecture Search pipeline using Stage 1 tuning result artifact
uv run tabflows run-skip-search \
    --tuning-artifact-uri "gs://your-bucket-name/automl_tabular_pipeline/tuning_result_artifact" \
    --async-mode
```

### 3. Python SDK

```python
from dotenv import load_dotenv
from google.cloud import aiplatform
from tabflows import TabularPipelineConfig, build_automl_tabular_pipeline

# 1. Load .env environment variables
load_dotenv()

# 2. Instantiate configuration (loads GCP_PROJECT, GCP_LOCATION, GCP_BUCKET_URI from .env)
config = TabularPipelineConfig()

# 3. Build pipeline template path and parameter values
template_path, parameter_values = build_automl_tabular_pipeline(config)

# 4. Initialize Vertex AI and PipelineJob
aiplatform.init(project=config.project_id, location=config.location)
job = aiplatform.PipelineJob(
    display_name="automl-tabular-run",
    location=config.location,
    template_path=template_path,
    pipeline_root=config.root_dir,
    parameter_values=parameter_values,
    enable_caching=False,
)

# 5. Submit job asynchronously to Vertex AI
job.submit()
print("Vertex AI Pipeline Job submitted successfully!")
```

### 4. Jupyter Notebook Tutorial

Register the project environment as a Jupyter kernel and launch the classification tutorial notebook:

```bash
# Register Jupyter kernel
uv run python -m ipykernel install --sys-prefix --name tabflows --display-name "Python 3.11 (tabflows)"

# Launch Jupyter Notebook
uv run jupyter notebook notebooks/01_automl_tabular_classification.ipynb
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
