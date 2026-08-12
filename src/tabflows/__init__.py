"""Tabflows: AutoML Tabular classification pipeline repository for Vertex AI."""

from tabflows.config import TabularPipelineConfig
from tabflows.pipeline import (
    build_automl_tabular_pipeline,
    build_skip_architecture_search_pipeline,
    generate_auto_transformation,
    get_bucket_name_and_path,
    get_task_detail,
    setup_gcp_resources,
)

__version__ = "0.1.0"
__all__ = [
    "TabularPipelineConfig",
    "build_automl_tabular_pipeline",
    "build_skip_architecture_search_pipeline",
    "generate_auto_transformation",
    "get_bucket_name_and_path",
    "get_task_detail",
    "setup_gcp_resources",
]
