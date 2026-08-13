"""Configuration schema for Vertex AI AutoML Tabular pipelines."""

from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TabularPipelineConfig(BaseSettings):
    """Configuration options for building AutoML Tabular pipelines.

    Loads environment variables from `.env` file automatically if present.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    project_id: str = Field(
        default="",
        description="Google Cloud Project ID",
        validation_alias="GCP_PROJECT",
    )
    location: str = Field(
        default="us-central1",
        description="GCP Region",
        validation_alias="GCP_LOCATION",
    )
    bucket_uri: str = Field(
        default="",
        description="Cloud Storage bucket URI (gs://...)",
        validation_alias="GCP_BUCKET_URI",
    )
    root_dir_name: str = Field(
        default="automl_tabular_pipeline",
        description="Root pipeline output folder",
        validation_alias="PIPELINE_ROOT_DIR_NAME",
    )
    prediction_type: str = Field(
        default="classification",
        description="Task prediction type",
        validation_alias="PREDICTION_TYPE",
    )
    optimization_objective: str = Field(
        default="minimize-log-loss",
        description="Optimization goal",
        validation_alias="OPTIMIZATION_OBJECTIVE",
    )
    target_column: str = Field(
        default="deposit",
        description="Target column name",
        validation_alias="TARGET_COLUMN",
    )
    data_source_csv_filenames: str = Field(
        default=(
            "gs://cloud-samples-data/vertex-ai/tabular-workflows/datasets/bank-marketing/train.csv"
        ),
        description="CSV dataset path in GCS",
    )
    data_source_bigquery_table_path: str | None = Field(
        default=None, description="BigQuery table URI format bq://project.dataset.table"
    )
    training_fraction: float | None = Field(default=0.8, description="Train split fraction")
    validation_fraction: float | None = Field(default=0.1, description="Validation split fraction")
    test_fraction: float | None = Field(default=0.1, description="Test split fraction")
    predefined_split_key: str | None = Field(
        default=None, description="Predefined split column key"
    )
    timestamp_split_key: str | None = Field(default=None, description="Timestamp split column key")
    stratified_split_key: str | None = Field(
        default=None, description="Stratified split column key"
    )
    weight_column: str | None = Field(default=None, description="Sample weight column name")
    train_budget_milli_node_hours: int = Field(
        default=1000, description="Training budget in milli node hours"
    )
    run_evaluation: bool = Field(default=True, description="Whether to run evaluation component")
    run_distillation: bool = Field(default=False, description="Whether to run model distillation")
    export_additional_model_without_custom_ops: bool = Field(
        default=False, description="Export model without custom TF ops"
    )
    features: list[str] = Field(
        default_factory=lambda: [
            "age",
            "job",
            "marital",
            "education",
            "default",
            "balance",
            "housing",
            "loan",
            "contact",
            "day",
            "month",
            "duration",
            "campaign",
            "pdays",
            "previous",
            "poutcome",
        ],
        description="List of feature column names",
    )
    study_spec_parameters_override: list[dict[str, Any]] | None = Field(
        default_factory=lambda: [
            {
                "parameter_id": "model_type",
                "categorical_value_spec": {"values": ["nn"]},
            }
        ],
        description="Study spec parameter overrides",
    )
    worker_pool_specs_override: Any | None = Field(
        default_factory=lambda: [
            {"machine_spec": {"machine_type": "n1-standard-8"}},
            {},
            {},
            {"machine_spec": {"machine_type": "n1-standard-4"}},
        ],
        description="Worker pool spec overrides",
    )
    dataflow_subnetwork: str | None = Field(default=None, description="Dataflow subnetwork URI")
    dataflow_use_public_ips: bool = Field(
        default=True, description="Whether Dataflow uses public IPs"
    )
    serving_machine_type: str = Field(
        default="n1-standard-4",
        description="Serving machine type for online endpoint deployment",
        validation_alias="SERVING_MACHINE_TYPE",
    )
    min_replica_count: int = Field(
        default=1,
        description="Minimum replica count for endpoint deployment",
        validation_alias="MIN_REPLICA_COUNT",
    )
    max_replica_count: int = Field(
        default=1,
        description="Maximum replica count for endpoint deployment",
        validation_alias="MAX_REPLICA_COUNT",
    )
    batch_predict_instances_format: str = Field(
        default="csv",
        description="Input instance format for batch prediction (csv, jsonl, bigquery)",
        validation_alias="BATCH_PREDICT_INSTANCES_FORMAT",
    )
    batch_predict_predictions_format: str = Field(
        default="jsonl",
        description="Output predictions format for batch prediction (jsonl, csv, bigquery)",
        validation_alias="BATCH_PREDICT_PREDICTIONS_FORMAT",
    )
    experiment_name: str = Field(
        default="automl-tabular-classification-experiments",
        description="Vertex AI Experiment name for tracking pipeline runs",
        validation_alias="EXPERIMENT_NAME",
    )
    fte_transformations_path: str | None = Field(
        default=None,
        description="GCS URI or path for custom Feature Transform Engine JSON spec",
        validation_alias="FTE_TRANSFORMATIONS_PATH",
    )

    @property
    def root_dir(self) -> str:
        """Return full Cloud Storage URI for pipeline outputs."""
        return f"{self.bucket_uri.rstrip('/')}/{self.root_dir_name}"

    @property
    def transform_config_path(self) -> str:
        """Return full Cloud Storage URI for transformation configuration JSON."""
        return f"{self.root_dir}/transform_config_unique.json"
