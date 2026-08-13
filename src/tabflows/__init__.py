"""Tabflows: AutoML Tabular classification pipeline repository for Vertex AI."""

from tabflows.config import TabularPipelineConfig
from tabflows.experiments import (
    get_model_evaluation_metrics,
    get_model_feature_attributions,
    list_experiment_runs,
)
from tabflows.inference import (
    cleanup_endpoint,
    deploy_model_to_endpoint,
    get_model_from_pipeline_job,
    list_models,
    predict_online,
    run_batch_prediction,
)
from tabflows.pipeline import (
    build_automl_tabular_pipeline,
    build_skip_architecture_search_pipeline,
    generate_auto_transformation,
    generate_fte_transformations,
    get_bucket_name_and_path,
    get_task_detail,
    setup_gcp_resources,
    write_fte_transformations,
)

__version__ = "0.1.0"
__all__ = [
    "TabularPipelineConfig",
    "build_automl_tabular_pipeline",
    "build_skip_architecture_search_pipeline",
    "cleanup_endpoint",
    "deploy_model_to_endpoint",
    "generate_auto_transformation",
    "generate_fte_transformations",
    "get_bucket_name_and_path",
    "get_model_evaluation_metrics",
    "get_model_feature_attributions",
    "get_model_from_pipeline_job",
    "get_task_detail",
    "list_experiment_runs",
    "list_models",
    "predict_online",
    "run_batch_prediction",
    "setup_gcp_resources",
    "write_fte_transformations",
]
