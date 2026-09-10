# Feature Selection & Ranking in AutoML Tabular Workflows

## Overview
Feature Selection in Vertex AI Tabular Workflows enables reducing dimensionality on high-cardinality enterprise datasets before or during model architecture search. By pruning uninformative and collinear features, workflows achieve:
- Shorter pipeline execution times (-40% to -60% node hours).
- Reduced risk of overfitting and higher test set generalization.
- Significantly lower online inference latency and endpoint resource footprint.

## Pipeline Implementations in `tabflows`

### 1. Standalone Feature Selection Pipeline
- **GCPC Module**: `google_cloud_pipeline_components.preview.automl.tabular.utils.get_feature_selection_pipeline_and_parameters`
- **SDK Function**: `create_feature_selection_pipeline_job(config)`
- **CLI Subcommand**: `tabflows run-feature-selection` and `tabflows compile-feature-selection`
- **Supported Engines**:
  - `bigquery`: Performs feature ranking and correlation filtering entirely within BigQuery using distributed SQL. Ideal for BigQuery-native datasets with zero data movement overhead.
  - `dataflow`: Runs an Apache Beam distributed pipeline for non-BigQuery CSV datasets stored in Cloud Storage.
- **Supported Ranking Algorithms**:
  - `AMI`: Adjusted Mutual Information
  - `CMIM`: Conditional Mutual Information Maximization
  - `JMIM`: Joint Mutual Information Maximization
  - `MRMR`: Minimum Redundancy Maximum Relevance

### 2. Integrated Feature Selection in Stage 1 Full Search
- **GCPC Module**: `google_cloud_pipeline_components.v1.automl.tabular.utils.get_automl_tabular_pipeline_and_parameters`
- **Config Parameters**:
  - `num_selected_features`: Target integer number of features to select for model training.
  - `apply_feature_selection_tuning`: Boolean flag indicating whether to tune the feature selection cutoff hyperparameter.
- **CLI Options**: `--num-selected-features INT` and `--apply-feature-selection-tuning` across `run-automl`, `run-pipeline`, and `compile-pipeline`.
