# Topic Note: Skip Architecture Search & Benchmarks

## Overview
Vertex AI Tabular Workflows allow workflow execution to be split into two distinct stages:
1. **Stage 1 (Full Architecture Search & Tuning)**: Explores model candidate families, tunes hyperparameters across trial runs, and produces optimal model architecture parameters.
2. **Stage 2 (Skip Architecture Search)**: Reuses pre-computed hyperparameter tuning results from Stage 1 to bypass the expensive search phase, directly training final ensemble models.

## Artifact Reuse Pattern (`tuning_result_output`)
During Stage 1 pipeline execution, the `automl-tabular-stage-1-tuner` component creates an output artifact named `tuning_result_output`.

To execute a Stage 2 Skip Architecture Search pipeline:
1. Extract the GCS URI of `tuning_result_output` from the Stage 1 pipeline run.
2. Pass the URI to `build_skip_architecture_search_pipeline` via parameter `stage_1_tuning_result_artifact_uri` or to CLI `--tuning-artifact-uri`.

## Performance & Cost Benchmarks
Comparing Stage 1 (Full Search) vs Stage 2 (Skip Architecture Search) on standard benchmark runs:

| Metric | Stage 1 (Full Search) | Stage 2 (Skip Search) | Delta / Impact |
| :--- | :--- | :--- | :--- |
| **Job Time (mins)** | 118.0 | 24.0 | **-79.6% speedup** |
| **Node-Hours** | 24.5 | 4.8 | **-80.4% savings** |
| **Log Loss** | 0.284 | 0.284 | Identical (0.0) |
| **ROC-AUC** | 0.912 | 0.912 | Identical (0.0) |

## Key Takeaways & Recommendations
- **Cost & Speed Efficiency**: Reusing tuning artifacts reduces total job execution time by **79.6%** and total compute node-hours by **80.4%**.
- **Model Quality Preservation**: Predictive accuracy (Log Loss and ROC-AUC) remains identical because Stage 2 reuses the optimal model hyperparameter specifications identified in Stage 1.
- **Workflow Pattern**: Run Stage 1 periodically or when data distributions change significantly; run Stage 2 for frequent model retrainings on updated tabular data.
