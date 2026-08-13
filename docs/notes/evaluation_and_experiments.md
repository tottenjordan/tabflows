# Topic Note: Model Evaluation & Vertex AI Experiment Tracking

## Overview
AutoML Tabular classification workflows on Vertex AI generate model evaluation metrics, confusion matrices, and feature attributions. `tabflows` provides utilities for accessing evaluation metrics and tracking training experiments.

## Model Evaluation Metrics
Model evaluations are registered under the trained `aiplatform.Model` resource in Vertex AI Model Registry:
- **Log Loss**: Measures cross-entropy loss (`logLoss`).
- **PR AUC**: Area under the Precision-Recall curve (`auPrc`).
- **ROC AUC**: Area under the Receiver Operating Characteristic curve (`auRoc`).
- **Confusion Matrix**: Classification confusion matrix (`confusionMatrix`).

### Python SDK Usage
```python
from tabflows import (
    get_model_evaluation_metrics,
    get_model_feature_attributions,
)

# Fetch evaluation metrics
metrics = get_model_evaluation_metrics(
    "projects/YOUR_PROJECT/locations/us-central1/models/YOUR_MODEL_ID"
)
print("Log Loss:", metrics.get("logLoss"))
print("ROC AUC:", metrics.get("auRoc"))

# Fetch global feature attributions / importance weights
attributions = get_model_feature_attributions(
    "projects/YOUR_PROJECT/locations/us-central1/models/YOUR_MODEL_ID"
)
```

### CLI Usage
```bash
uv run tabflows get-evaluation --model "projects/YOUR_PROJECT/locations/us-central1/models/YOUR_MODEL_ID"
```

## Vertex AI Experiment Tracking
Vertex AI Experiments track parameters, metrics, and pipeline run metadata:
- **Experiment Name**: Default `automl-tabular-classification-experiments` (configurable via `EXPERIMENT_NAME` in `.env`).
- **Comparing Pipeline Runs**: Side-by-side comparison of standard search vs skip-architecture-search runs.

### Python SDK Usage
```python
from tabflows import list_experiment_runs

df = list_experiment_runs(experiment_name="automl-tabular-classification-experiments")
print(df)
```

### CLI Usage
```bash
uv run tabflows list-experiments
```
