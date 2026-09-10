"""Configuration schema for Vertex AI AutoML Tabular pipelines."""

from typing import Any

from pydantic import AliasChoices, Field, field_validator
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
    optimization_objective_recall_value: float | None = Field(
        default=None,
        description="Target recall value for maximize-precision-at-recall objective",
        validation_alias="OPTIMIZATION_OBJECTIVE_RECALL_VALUE",
    )
    optimization_objective_precision_value: float | None = Field(
        default=None,
        description="Target precision value for maximize-recall-at-precision objective",
        validation_alias="OPTIMIZATION_OBJECTIVE_PRECISION_VALUE",
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
        validation_alias=AliasChoices(
            "DATA_SOURCE_CSV_FILENAMES", "csv_file_path", "data_source_csv_filenames"
        ),
    )
    data_source_bigquery_table_path: str | None = Field(
        default=None, description="BigQuery table URI format bq://project.dataset.table"
    )
    bigquery_table_path: str | None = Field(
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
    skip_evaluation: bool = Field(
        default=False, description="Whether to bypass evaluation component"
    )
    run_distillation: bool = Field(default=False, description="Whether to run model distillation")
    run_architecture_search: bool = Field(
        default=True,
        description="Whether to run full architecture search and hyperparameter tuning",
        validation_alias="RUN_ARCHITECTURE_SEARCH",
    )
    tuning_result_output: str | None = Field(
        default=None,
        description="GCS URI or artifact path of Stage 1 tuning_result_output",
        validation_alias="TUNING_RESULT_OUTPUT",
    )
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
    dataflow_machine_type: str | None = Field(
        default=None,
        description="Machine type for Dataflow workers",
        validation_alias="DATAFLOW_MACHINE_TYPE",
    )
    dataflow_max_num_workers: int | None = Field(
        default=None,
        description="Maximum number of Dataflow worker instances",
        validation_alias="DATAFLOW_MAX_NUM_WORKERS",
    )
    dataflow_disk_size_gb: int | None = Field(
        default=None,
        description="Disk size in GB for Dataflow worker instances",
        validation_alias="DATAFLOW_DISK_SIZE_GB",
    )
    encryption_spec_key_name: str | None = Field(
        default=None,
        description="Cloud KMS encryption key name for CMEK",
        validation_alias="ENCRYPTION_SPEC_KEY_NAME",
    )
    dataflow_subnetwork: str | None = Field(default=None, description="Dataflow subnetwork URI")
    dataflow_use_public_ips: bool = Field(
        default=True, description="Whether Dataflow uses public IPs"
    )
    dataflow_service_account: str | None = Field(
        default=None,
        description="Custom service account email for Dataflow jobs",
        validation_alias="DATAFLOW_SERVICE_ACCOUNT",
    )
    enable_probabilistic_inference: bool = Field(
        default=False,
        description="Whether to enable probabilistic quantile inference",
        validation_alias="ENABLE_PROBABILISTIC_INFERENCE",
    )
    quantiles: list[float] | None = Field(
        default=None,
        description="Quantiles for probabilistic inference (up to 5 values between 0.0 and 1.0)",
        validation_alias="QUANTILES",
    )
    model_display_name: str | None = Field(
        default=None,
        description="Model display name registered in Vertex AI Model Registry",
        validation_alias="MODEL_DISPLAY_NAME",
    )
    model_description: str | None = Field(
        default=None,
        description="Model description registered in Vertex AI Model Registry",
        validation_alias="MODEL_DESCRIPTION",
    )
    evaluation_batch_predict_machine_type: str | None = Field(
        default=None,
        description="Machine type for evaluation batch predict",
        validation_alias="EVALUATION_BATCH_PREDICT_MACHINE_TYPE",
    )
    evaluation_batch_predict_max_replica_count: int | None = Field(
        default=None,
        description="Max replica count for evaluation batch predict",
        validation_alias="EVALUATION_BATCH_PREDICT_MAX_REPLICA_COUNT",
    )
    evaluation_batch_explain_machine_type: str | None = Field(
        default=None,
        description="Machine type for evaluation batch explain",
        validation_alias="EVALUATION_BATCH_EXPLAIN_MACHINE_TYPE",
    )
    evaluation_batch_explain_max_replica_count: int | None = Field(
        default=None,
        description="Max replica count for evaluation batch explain",
        validation_alias="EVALUATION_BATCH_EXPLAIN_MAX_REPLICA_COUNT",
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
        validation_alias=AliasChoices(
            "FTE_TRANSFORMATIONS_PATH", "transform_config_path", "fte_transformations_path"
        ),
    )
    feature_selection_algorithm: str | None = Field(
        default=None,
        description="Feature selection algorithm: AMI, CMIM, JMIM, MRMR",
        validation_alias="FEATURE_SELECTION_ALGORITHM",
    )
    feature_selection_engine: str = Field(
        default="bigquery",
        description="Execution engine for feature selection: 'bigquery' or 'dataflow'",
        validation_alias="FEATURE_SELECTION_ENGINE",
    )
    max_selected_features: int | None = Field(
        default=None,
        description="Maximum number of features to select in standalone pipeline",
        validation_alias="MAX_SELECTED_FEATURES",
    )
    num_selected_features: int | None = Field(
        default=None,
        description="Number of features to filter down to during Stage 1 AutoML training",
        validation_alias="NUM_SELECTED_FEATURES",
    )
    apply_feature_selection_tuning: bool | None = Field(
        default=None,
        description="Whether to tune feature selection hyperparameters in Stage 1",
        validation_alias="APPLY_FEATURE_SELECTION_TUNING",
    )
    bigquery_staging_full_dataset_id: str | None = Field(
        default=None,
        description="BigQuery dataset for intermediate staging tables",
        validation_alias="BIGQUERY_STAGING_FULL_DATASET_ID",
    )

    @field_validator("feature_selection_algorithm")
    @classmethod
    def validate_feature_selection_algorithm(cls, v: str | None) -> str | None:
        if v is not None:
            v_upper = v.upper()
            allowed = {"AMI", "CMIM", "JMIM", "MRMR"}
            if v_upper not in allowed:
                raise ValueError(
                    f"Invalid feature_selection_algorithm: {v}. Must be one of {sorted(allowed)}"
                )
            return v_upper
        return v

    @field_validator("feature_selection_engine")
    @classmethod
    def validate_feature_selection_engine(cls, v: str) -> str:
        v_lower = v.lower()
        if v_lower not in {"bigquery", "dataflow"}:
            raise ValueError(
                f"Invalid feature_selection_engine: {v}. Must be 'bigquery' or 'dataflow'"
            )
        return v_lower

    @property
    def root_dir(self) -> str:
        """Return full Cloud Storage URI for pipeline outputs."""
        return f"{self.bucket_uri.rstrip('/')}/{self.root_dir_name}"

    @property
    def transform_config_path(self) -> str:
        """Return full Cloud Storage URI for transformation configuration JSON."""
        if self.fte_transformations_path:
            return self.fte_transformations_path
        return f"{self.root_dir}/transform_config_unique.json"
