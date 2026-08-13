# Topic Note: Model Evaluation & Vertex AI Experiment Tracking

## Overview
AutoML Tabular classification workflows on Vertex AI generate model evaluation metrics, confusion matrices, and global feature attributions. `tabflows` provides integrated utilities for full-lifecycle experiment tracking across pipeline job submissions, Design of Experiments (DoE) hyperparameter campaigns, model evaluations, real-time endpoint deployments, and batch prediction jobs.

## Full-Lifecycle Experiment Tracking Architecture
`tabflows` automates logging of execution metadata, parameters, model evaluation metrics, and artifact URIs to Vertex AI Experiments (`automl-tabular-classification-experiments` by default).

### 1. Pipeline Job Tracking
- **`create_tabular_pipeline_job`**: Stage 1 full-search pipeline execution automatically logs `PipelineJob` resource names, target columns, optimization objectives, and training milli-node-hour budgets.
- **`run_skip_architecture_search_pipeline`**: Stage 2 skip-search pipeline logs the reused Stage 1 `tuning_result_output` artifact URI and execution metadata.

### 2. Design of Experiments (DoE) Campaigns
- **`run_doe_campaign`**: Programmatically iterates over a list of parameter variant dictionaries, automatically initiating isolated experiment runs (e.g., `<campaign>-<variant_name>`) to evaluate multi-configuration performance grids.

### 3. Model Evaluation & Feature Importance
- **`get_model_evaluation_metrics`**: Extracts metrics directly from the Vertex AI Model Registry (`logLoss`, `auPrc`, `auRoc`, `confusionMatrix`).
- **`get_model_feature_attributions`**: Retrieves global feature importance weights (`meanAttributions`) derived from Shapley feature explanations.

### 4. Serving & Inference Tracking
- **`deploy_model_to_endpoint`**: Logs deployment experiment runs containing endpoint resource URIs, model resource names, and serving machine specs (`serving_machine_type`).
- **`run_batch_prediction`**: Logs batch prediction experiment runs capturing batch job URIs, input GCS dataset sources, and output prediction prefixes.

---

## Python SDK Usage

### 1. Manual & Custom Experiment Run Logging
```python
from tabflows import log_experiment_run

# Log custom parameters and evaluation metrics
log_experiment_run(
    run_name="manual-tuning-run-01",
    params={"train_budget_milli_node_hours": 1000, "run_distillation": True},
    metrics={"logLoss": 0.321, "auRoc": 0.945},
    model="projects/YOUR_PROJECT/locations/us-central1/models/YOUR_MODEL_ID",
)
```

### 2. Pipeline Execution with Automatic Experiment Tracking
```python
from tabflows import (
    TabularPipelineConfig,
    create_tabular_pipeline_job,
    run_skip_architecture_search_pipeline,
)

config = TabularPipelineConfig()

# Stage 1: Full Search Pipeline (logs run automatically)
job_stage1 = create_tabular_pipeline_job(
    config=config,
    job_id="automl-tabular-stage1",
    log_experiment=True,
)

# Stage 2: Skip Architecture Search Pipeline (logs run automatically)
job_stage2 = run_skip_architecture_search_pipeline(
    config=config,
    tuning_result_artifact_uri="gs://your-bucket/stage1_job/automl-tabular-stage-1-tuner/tuning_result_output",
    job_id="automl-tabular-stage2",
    log_experiment=True,
)
```

### 3. Design of Experiments (DoE) Campaign
```python
from tabflows import run_doe_campaign

variants = [
    {"name": "variant-fast", "train_budget_milli_node_hours": 1000, "run_distillation": False},
    {"name": "variant-distill", "train_budget_milli_node_hours": 2000, "run_distillation": True},
]

summary = run_doe_campaign(
    campaign_name="budget-ab-test",
    variants=variants,
)
```

### 4. Fetching Evaluation Metrics & Feature Importance
```python
from tabflows import get_model_evaluation_metrics, get_model_feature_attributions

model_id = "projects/YOUR_PROJECT/locations/us-central1/models/YOUR_MODEL_ID"

metrics = get_model_evaluation_metrics(model=model_id)
print("Log Loss:", metrics.get("logLoss"))
print("ROC AUC:", metrics.get("auRoc"))

attributions = get_model_feature_attributions(model=model_id)
print("Top Feature Attributions:", attributions)
```

### 5. Deployment & Batch Prediction Tracking
```python
from tabflows import deploy_model_to_endpoint, run_batch_prediction

# Deploy endpoint (logs experiment run automatically)
endpoint = deploy_model_to_endpoint(
    model=model_id,
    log_experiment=True,
)

# Batch Prediction (logs experiment run automatically)
batch_job = run_batch_prediction(
    model=model_id,
    gcs_source="gs://your-bucket/test_instances.csv",
    log_experiment=True,
)
```

### 6. Comparing Experiment Runs DataFrame
```python
from tabflows import list_experiment_runs

df = list_experiment_runs(experiment_name="automl-tabular-classification-experiments")
print(df)
```

---

## CLI Usage

### 1. Retrieve Model Evaluation Metrics
```bash
uv run tabflows get-evaluation --model "projects/YOUR_PROJECT/locations/us-central1/models/YOUR_MODEL_ID"
```

### 2. List Logged Experiment Runs
```bash
uv run tabflows list-experiments --experiment "automl-tabular-classification-experiments"
```

### 3. Log Custom Experiment Run (`log-experiment`)
Log parameters, metrics, model evaluations, and pipeline job references from the command line:

```bash
uv run tabflows log-experiment \
    --run-name "cli-experiment-run-01" \
    --model-resource-name "projects/YOUR_PROJECT/locations/us-central1/models/YOUR_MODEL_ID" \
    --pipeline-job-id "projects/YOUR_PROJECT/locations/us-central1/pipelineJobs/YOUR_JOB_ID" \
    --params-json '{"train_budget_milli_node_hours": 1000, "run_distillation": true}' \
    --metrics-json '{"custom_eval_score": 0.942}' \
    --experiment "automl-tabular-classification-experiments"
```

#### CLI Options
- `--run-name`: (Required) Display name for the experiment run.
- `--model-resource-name`: Model resource name or ID in Vertex AI Model Registry. Automatically extracts and logs primary scalar evaluation metrics.
- `--pipeline-job-id`: Pipeline job ID or resource name associated with the run.
- `--params-json`: JSON string of parameter key-value pairs to log.
- `--metrics-json`: JSON string of custom evaluation metric key-value pairs to log.
- `--experiment`: Target experiment name (defaults to `EXPERIMENT_NAME` in `.env`).
- `--project` / `--location`: Override GCP project and region configuration.
