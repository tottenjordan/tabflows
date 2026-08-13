# Topic Note: Vertex AI Tabular Workflows

## Overview
Vertex AI Tabular Workflows are Google-managed Kubeflow Pipelines components for training AutoML Tabular models, supporting hyperparameter tuning, model evaluation, architecture search, and skip-architecture-search execution.

## Package Import & API Evolution
- **Legacy API**: `google_cloud_pipeline_components.experimental.automl.tabular.utils`
  - Parameter for transformation config path: `transform_config_path`
- **Current Stable API**: `google_cloud_pipeline_components.v1.automl.tabular.utils`
  - Parameter for transformation config path: `transformations` (accepts GCS path string or JSON spec)

Key functions available in `google_cloud_pipeline_components.v1.automl.tabular.utils`:
- `get_automl_tabular_pipeline_and_parameters(...)`: Generates pipeline template path and parameter dictionary for standard AutoML tabular training with hyperparameter search.
- `get_skip_architecture_search_pipeline_and_parameters(...)`: Generates pipeline template path and parameter dictionary reusing tuning results from a prior run.

## Key Pipeline Components & Task Names
- `automl-tabular-stage-1-tuner`: Output artifact `tuning_result_output` contains `tuning_result_artifact_uri` required for skip-architecture-search runs.
- `model-upload`: Uploads trained ensemble model artifact to Vertex AI Model Registry.
- `model-evaluation-2`: Output artifact `evaluation_metrics` contains metrics JSON.
- `feature-attribution-2`: Output artifact `feature_attributions` contains feature attributions JSON.

## Dataset & Transformation Specification
- Transformation configurations are JSON files specifying column-level auto transformation specs:
  `[{"auto": {"column_name": "col_1"}}, ...]`
- Transformation configs are stored on GCS and passed to pipeline creation via `transform_config_path`.
- **BigQuery Direct Ingestion**: Pipelines accept BigQuery table paths formatted as `bq://project.dataset.table` via `bigquery_table_path` to avoid intermediate CSV export steps.
- **Data Splitting Strategies**: Supports `predefined_split_key` (for custom/chronological split assignments using `TRAIN`, `VALIDATE`, `TEST` values), `timestamp_split_key`, and `stratified_split_key`.
- **Specialized Optimization Objectives**: Supports `maximize-precision-at-recall` and `maximize-recall-at-precision` objectives using `optimization_objective_recall_value` and `optimization_objective_precision_value` constraints.

