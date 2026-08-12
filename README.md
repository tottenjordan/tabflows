# tabflows

`tabflows` is a Python library and reference repository for designing, building, and executing AutoML Tabular classification workflows on Google Cloud Vertex AI Pipelines.

---

## Features

- **Vertex AI Tabular Workflows**: Native support for Google-managed AutoML Tabular pipelines (`google-cloud-pipeline-components>=2.22.0`).
- **Skip Architecture Search**: Fast model tuning by reusing Stage 1 hyperparameter tuning results to save time and cost.
- **Modern Python Architecture**: Built with `uv` for package management, `ruff` for linting/formatting, `ty` for static type checking, and `pytest` for unit testing.
- **CLI & SDK Interfaces**: Command line tools and modular Python SDK for pipeline compilation and execution.

---

## Getting Started

### Prerequisites

- **Python**: `>=3.11`
- **uv**: Installed ([https://github.com/astral-sh/uv](https://github.com/astral-sh/uv))
- **Google Cloud SDK**: Configured with active authentication and project permissions.

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


### Usage

#### 1. Provision GCP Resources

Provision the Cloud Storage bucket and upload initial feature transformation config JSON:

```bash
# Using CLI (reads project and bucket from .env automatically)
uv run tabflows setup

# Or using script
uv run python scripts/setup_gcp_resources.py
```

#### 2. Command Line Interface (CLI)

Compile or launch an AutoML Tabular pipeline using the `tabflows` CLI:

```bash
# Compile standard AutoML Tabular pipeline template
uv run tabflows run-automl --compile-only

# Run Skip Architecture Search pipeline using Stage 1 tuning result artifact
uv run tabflows run-skip-search \
    --tuning-artifact-uri "gs://your-bucket-name/automl_tabular_pipeline/tuning_result_artifact" \
    --compile-only
```

#### 2. Python API

```python
from tabflows import TabularPipelineConfig, build_automl_tabular_pipeline

# Define pipeline configuration
config = TabularPipelineConfig(
    project_id="your-gcp-project-id",
    bucket_uri="gs://your-bucket-name",
    target_column="deposit",
    prediction_type="classification",
)

# Build pipeline template and parameters
template_path, parameter_values = build_automl_tabular_pipeline(config)
print(f"Pipeline compiled to: {template_path}")
```

#### 3. Jupyter Notebook Tutorial

Launch the end-to-end classification example in Jupyter:

```bash
uv run jupyter notebook notebooks/01_automl_tabular_classification.ipynb
```

---

## Development

Run development targets using `make`:

```bash
# Install dependencies
make dev

# Run code formatters (ruff)
make format

# Run linter (ruff) and type checker (ty)
make lint

# Run unit tests (pytest)
make test
```

---

## Code Standards

Refer to [CODE_STANDARDS.md](CODE_STANDARDS.md) for repository conventions and coding rules.
