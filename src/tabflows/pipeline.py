"""Pipeline construction and GCS helper functions for AutoML Tabular workflows."""

import json
import logging
from typing import Any, cast

import google.auth
from google.auth.credentials import AnonymousCredentials
from google.cloud import aiplatform, storage
from google_cloud_pipeline_components.v1.automl.tabular import utils as automl_tabular_utils

from tabflows.config import TabularPipelineConfig
from tabflows.experiments import log_experiment_run

logger = logging.getLogger(__name__)


def _get_credentials_or_anonymous(credentials: Any | None = None) -> Any:
    if credentials is not None:
        return credentials
    try:
        creds, _ = google.auth.default()
        return creds
    except Exception:
        return AnonymousCredentials()



def generate_auto_transformation(column_names: list[str]) -> list[dict[str, Any]]:
    """Generate auto transformation configurations for a list of feature columns."""
    return [{"auto": {"column_name": col}} for col in column_names]


def get_bucket_name_and_path(uri: str) -> tuple[str, str]:
    """Parse GCS URI (gs://bucket/path) into bucket name and relative object path."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {uri}. Must start with gs://")
    no_prefix = uri[5:]
    splits = no_prefix.split("/", 1)
    bucket_name = splits[0]
    object_path = splits[1] if len(splits) > 1 else ""
    return bucket_name, object_path


def download_from_gcs(storage_client: storage.Client, uri: str) -> str:
    """Download string content from a Cloud Storage URI."""
    bucket_name, object_path = get_bucket_name_and_path(uri)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(object_path)
    return blob.download_as_string().decode("utf-8")


def write_to_gcs(storage_client: storage.Client, uri: str, content: str) -> None:
    """Upload string content to a Cloud Storage URI."""
    bucket_name, object_path = get_bucket_name_and_path(uri)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(object_path)
    blob.upload_from_string(content)


def write_auto_transformations(
    storage_client: storage.Client, uri: str, column_names: list[str]
) -> None:
    """Generate and write auto transformation configuration JSON to Cloud Storage."""
    transformations = generate_auto_transformation(column_names)
    write_to_gcs(storage_client, uri, json.dumps(transformations))


FTE_TYPE_MAP = {
    "categorical": "categorical",
    "numeric": "numeric",
    "timestamp": "timestamp",
    "text": "text_embedding",
    "auto": "auto",
}


def generate_fte_transformations(column_types: dict[str, str]) -> list[dict[str, Any]]:
    """Generate Feature Transform Engine (FTE) transformations configuration list.

    Args:
        column_types: Mapping of column names to transform type strings
                     ("categorical", "numeric", "timestamp", "text", "auto").

    Returns:
        List of FTE transformation dictionaries.
    """
    transformations: list[dict[str, Any]] = []
    for col_name, col_type in column_types.items():
        if col_type not in FTE_TYPE_MAP:
            raise ValueError(
                f"Unsupported FTE column type: '{col_type}'. "
                f"Supported types are: {list(FTE_TYPE_MAP.keys())}"
            )
        transform_key = FTE_TYPE_MAP[col_type]
        transformations.append({transform_key: {"column_name": col_name}})
    return transformations


def write_fte_transformations(
    storage_client: storage.Client, uri: str, column_types: dict[str, str]
) -> None:
    """Generate and write FTE transformation configuration JSON to Cloud Storage."""
    transformations = generate_fte_transformations(column_types)
    write_to_gcs(storage_client, uri, json.dumps(transformations))


SAMPLE_TEST_CSV = (
    "age,job,marital,education,default,balance,housing,loan,contact,day,month,duration,campaign,pdays,previous,poutcome\n"
    "35,technician,married,tertiary,no,1350,yes,no,cellular,15,may,220,1,-1,0,unknown\n"
    "42,admin.,single,secondary,no,450,no,no,cellular,18,jul,180,2,-1,0,unknown\n"
)


def setup_gcp_resources(
    config: TabularPipelineConfig,
    storage_client: storage.Client | None = None,
) -> dict[str, str]:
    """Create Cloud Storage bucket if missing and upload initial assets.

    Returns summary dictionary of created and configured assets.
    """
    if storage_client is None:
        storage_client = storage.Client(project=config.project_id)

    bucket_name, _ = get_bucket_name_and_path(config.bucket_uri)

    bucket = storage_client.bucket(bucket_name)
    if not bucket.exists():
        bucket = storage_client.create_bucket(
            bucket_or_name=bucket_name,
            project=config.project_id,
            location=config.location,
        )
        bucket_created = True
    else:
        bucket_created = False

    write_auto_transformations(storage_client, config.transform_config_path, config.features)

    test_instances_uri = f"{config.bucket_uri}/test_instances.csv"
    write_to_gcs(storage_client, test_instances_uri, SAMPLE_TEST_CSV)

    return {
        "bucket_name": bucket_name,
        "bucket_uri": config.bucket_uri,
        "bucket_created": str(bucket_created),
        "transform_config_path": config.transform_config_path,
        "test_instances_uri": test_instances_uri,
    }


def get_task_detail(task_details: list[Any], task_name: str) -> Any | None:
    """Retrieve task detail object by task name from Vertex AI Pipeline job output."""
    for task in task_details:
        t_name = getattr(task, "task_name", None) or getattr(task, "name", None)
        if t_name == task_name:
            return task
    return None


def _build_data_source_spec(config: TabularPipelineConfig) -> dict[str, Any]:
    bq_table = config.bigquery_table_path or config.data_source_bigquery_table_path
    if bq_table:
        return {"bigquery_table_path": bq_table}
    if isinstance(config.data_source_csv_filenames, list):
        return {"csv_filenames": config.data_source_csv_filenames}
    return {"csv_filenames": [config.data_source_csv_filenames]}


def _build_split_spec(config: TabularPipelineConfig) -> dict[str, Any]:
    if config.predefined_split_key:
        return {"predefined_split_key": config.predefined_split_key}
    if config.timestamp_split_key:
        return {"timestamp_split_key": config.timestamp_split_key}
    if config.stratified_split_key:
        return {"stratified_split_key": config.stratified_split_key}
    return {
        "fraction_split": {
            "training_fraction": config.training_fraction,
            "validation_fraction": config.validation_fraction,
            "test_fraction": config.test_fraction,
        }
    }


def build_automl_tabular_pipeline(
    config: TabularPipelineConfig,
) -> tuple[str, dict[str, Any]]:
    """Build AutoML Tabular pipeline template path and parameter dictionary.

    Uses google_cloud_pipeline_components.v1.automl.tabular.utils.
    """
    return automl_tabular_utils.get_automl_tabular_pipeline_and_parameters(
        project=config.project_id,
        location=config.location,
        root_dir=config.root_dir,
        target_column=config.target_column,
        prediction_type=config.prediction_type,
        optimization_objective=config.optimization_objective,
        optimization_objective_recall_value=config.optimization_objective_recall_value,
        optimization_objective_precision_value=config.optimization_objective_precision_value,
        transformations=config.transform_config_path,
        train_budget_milli_node_hours=config.train_budget_milli_node_hours,
        data_source_csv_filenames=config.data_source_csv_filenames,
        data_source_bigquery_table_path=(
            config.bigquery_table_path or config.data_source_bigquery_table_path
        ),
        weight_column=config.weight_column,
        predefined_split_key=config.predefined_split_key,
        timestamp_split_key=config.timestamp_split_key,
        stratified_split_key=config.stratified_split_key,
        training_fraction=config.training_fraction,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
        study_spec_parameters_override=config.study_spec_parameters_override,
        stage_1_tuner_worker_pool_specs_override=config.worker_pool_specs_override,
        cv_trainer_worker_pool_specs_override=config.worker_pool_specs_override,
        run_evaluation=False if config.skip_evaluation else config.run_evaluation,
        run_distillation=config.run_distillation,
        dataflow_subnetwork=config.dataflow_subnetwork,
        dataflow_use_public_ips=config.dataflow_use_public_ips,
        export_additional_model_without_custom_ops=config.export_additional_model_without_custom_ops,
    )


def build_skip_architecture_search_pipeline(
    config: TabularPipelineConfig,
    stage_1_tuning_result_artifact_uri: str,
) -> tuple[str, dict[str, Any]]:
    """Build Skip Architecture Search AutoML Tabular pipeline template path and parameter dict.

    Uses google_cloud_pipeline_components.v1.automl.tabular.utils.
    """
    return automl_tabular_utils.get_skip_architecture_search_pipeline_and_parameters(
        project=config.project_id,
        location=config.location,
        root_dir=config.root_dir,
        target_column=config.target_column,
        prediction_type=config.prediction_type,
        optimization_objective=config.optimization_objective,
        optimization_objective_recall_value=config.optimization_objective_recall_value,
        optimization_objective_precision_value=config.optimization_objective_precision_value,
        transformations=config.transform_config_path,
        train_budget_milli_node_hours=config.train_budget_milli_node_hours,
        data_source_csv_filenames=config.data_source_csv_filenames,
        data_source_bigquery_table_path=(
            config.bigquery_table_path or config.data_source_bigquery_table_path
        ),
        weight_column=config.weight_column,
        predefined_split_key=config.predefined_split_key,
        timestamp_split_key=config.timestamp_split_key,
        stratified_split_key=config.stratified_split_key,
        training_fraction=config.training_fraction,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
        stage_1_tuning_result_artifact_uri=stage_1_tuning_result_artifact_uri,
        run_evaluation=False if config.skip_evaluation else config.run_evaluation,
        dataflow_subnetwork=config.dataflow_subnetwork,
        dataflow_use_public_ips=config.dataflow_use_public_ips,
        export_additional_model_without_custom_ops=config.export_additional_model_without_custom_ops,
    )


def build_skip_evaluation_pipeline(
    config: TabularPipelineConfig,
) -> tuple[str, dict[str, Any]]:
    """Build Skip Evaluation AutoML Tabular pipeline template path and parameter dict.

    Uses google_cloud_pipeline_components.v1.automl.tabular.utils.
    """
    data_source = _build_data_source_spec(config)
    split_spec = _build_split_spec(config)
    transformations = generate_auto_transformation(config.features)

    return automl_tabular_utils.get_skip_evaluation_pipeline_and_parameters(
        project=config.project_id,
        location=config.location,
        root_dir=config.root_dir,
        target_column_name=config.target_column,
        prediction_type=config.prediction_type,
        optimization_objective=config.optimization_objective,
        optimization_objective_recall_value=(
            config.optimization_objective_recall_value
            if config.optimization_objective_recall_value is not None
            else -1.0
        ),
        optimization_objective_precision_value=(
            config.optimization_objective_precision_value
            if config.optimization_objective_precision_value is not None
            else -1.0
        ),
        transformations=cast(Any, transformations),
        split_spec=split_spec,
        data_source=data_source,
        train_budget_milli_node_hours=float(config.train_budget_milli_node_hours),
        weight_column_name=config.weight_column or "",
        study_spec_override=cast(Any, config.study_spec_parameters_override),
        stage_1_tuner_worker_pool_specs_override=cast(Any, config.worker_pool_specs_override),
        cv_trainer_worker_pool_specs_override=cast(Any, config.worker_pool_specs_override),
        export_additional_model_without_custom_ops=config.export_additional_model_without_custom_ops,
        dataflow_subnetwork=config.dataflow_subnetwork or "",
        dataflow_use_public_ips=config.dataflow_use_public_ips,
    )


def build_distill_skip_evaluation_pipeline(
    config: TabularPipelineConfig,
) -> tuple[str, dict[str, Any]]:
    """Build Distill Skip Evaluation AutoML Tabular pipeline template path and parameter dict.

    Uses google_cloud_pipeline_components.v1.automl.tabular.utils.
    """
    data_source = _build_data_source_spec(config)
    split_spec = _build_split_spec(config)
    transformations = generate_auto_transformation(config.features)

    return automl_tabular_utils.get_distill_skip_evaluation_pipeline_and_parameters(
        project=config.project_id,
        location=config.location,
        root_dir=config.root_dir,
        target_column_name=config.target_column,
        prediction_type=config.prediction_type,
        optimization_objective=config.optimization_objective,
        optimization_objective_recall_value=(
            config.optimization_objective_recall_value
            if config.optimization_objective_recall_value is not None
            else -1.0
        ),
        optimization_objective_precision_value=(
            config.optimization_objective_precision_value
            if config.optimization_objective_precision_value is not None
            else -1.0
        ),
        transformations=cast(Any, transformations),
        split_spec=split_spec,
        data_source=data_source,
        train_budget_milli_node_hours=float(config.train_budget_milli_node_hours),
        weight_column_name=config.weight_column or "",
        study_spec_override=cast(Any, config.study_spec_parameters_override),
        stage_1_tuner_worker_pool_specs_override=cast(Any, config.worker_pool_specs_override),
        cv_trainer_worker_pool_specs_override=cast(Any, config.worker_pool_specs_override),
        export_additional_model_without_custom_ops=config.export_additional_model_without_custom_ops,
        dataflow_subnetwork=config.dataflow_subnetwork or "",
        dataflow_use_public_ips=config.dataflow_use_public_ips,
    )


def get_no_custom_ops_model_uri(storage_client: storage.Client, task_details: list[Any]) -> str:
    """Extract model without custom ops GCS URI from task details."""
    ensemble_task = get_task_detail(task_details, "automl-tabular-ensemble")
    if not ensemble_task:
        raise ValueError("Task 'automl-tabular-ensemble' not found in task details.")
    uri = ensemble_task.outputs["model_without_custom_ops"].artifacts[0].uri
    return download_from_gcs(storage_client, uri)


def get_feature_attributions(storage_client: storage.Client, task_details: list[Any]) -> str:
    """Download feature attributions JSON from pipeline task details."""
    ensemble_task = get_task_detail(task_details, "feature-attribution-2")
    if not ensemble_task:
        raise ValueError("Task 'feature-attribution-2' not found in task details.")
    uri = ensemble_task.outputs["feature_attributions"].artifacts[0].uri
    return download_from_gcs(storage_client, uri)


def get_evaluation_metrics(storage_client: storage.Client, task_details: list[Any]) -> str:
    """Download evaluation metrics JSON from pipeline task details."""
    ensemble_task = get_task_detail(task_details, "model-evaluation-2")
    if not ensemble_task:
        raise ValueError("Task 'model-evaluation-2' not found in task details.")
    uri = ensemble_task.outputs["evaluation_metrics"].artifacts[0].uri
    return download_from_gcs(storage_client, uri)


def create_tabular_pipeline_job(
    config: TabularPipelineConfig,
    job_id: str = "automl-tabular-full-search",
    log_experiment: bool = True,
    credentials: Any | None = None,
) -> aiplatform.PipelineJob:
    """Create a Vertex AI PipelineJob for Stage 1 AutoML Tabular (Full Search)."""
    template_path, parameter_values = build_automl_tabular_pipeline(config)
    job = aiplatform.PipelineJob(
        display_name=job_id,
        project=config.project_id,
        location=config.location,
        template_path=template_path,
        job_id=job_id,
        pipeline_root=config.root_dir,
        parameter_values=parameter_values,
        enable_caching=False,
        credentials=_get_credentials_or_anonymous(credentials),
    )
    if log_experiment:
        try:
            log_experiment_run(
                run_name=job_id,
                pipeline_job=job,
                config=config,
            )
        except Exception as e:
            logger.warning("Failed to log experiment run for pipeline job '%s': %s", job_id, e)
    return job


def run_skip_architecture_search_pipeline(
    config: TabularPipelineConfig,
    tuning_result_artifact_uri: str | None = None,
    job_id: str = "automl-tabular-skip-search",
    log_experiment: bool = True,
    credentials: Any | None = None,
) -> aiplatform.PipelineJob:
    """Create a Vertex AI PipelineJob for Stage 2 AutoML Tabular (Skip Architecture Search)."""
    tuning_uri = tuning_result_artifact_uri or config.tuning_result_output
    if not tuning_uri:
        raise ValueError(
            "tuning_result_output must be provided in config "
            "or as argument tuning_result_artifact_uri"
        )

    template_path, parameter_values = build_skip_architecture_search_pipeline(
        config=config,
        stage_1_tuning_result_artifact_uri=tuning_uri,
    )
    job = aiplatform.PipelineJob(
        display_name=job_id,
        project=config.project_id,
        location=config.location,
        template_path=template_path,
        job_id=job_id,
        pipeline_root=config.root_dir,
        parameter_values=parameter_values,
        enable_caching=False,
        credentials=_get_credentials_or_anonymous(credentials),
    )
    if log_experiment:
        try:
            log_experiment_run(
                run_name=job_id,
                pipeline_job=job,
                config=config,
            )
        except Exception as e:
            logger.warning("Failed to log experiment run for pipeline job '%s': %s", job_id, e)
    return job


def create_skip_evaluation_pipeline_job(
    config: TabularPipelineConfig,
    job_id: str = "automl-tabular-skip-evaluation",
    log_experiment: bool = True,
    credentials: Any | None = None,
) -> aiplatform.PipelineJob:
    """Create a Vertex AI PipelineJob for AutoML Tabular skipping evaluation."""
    template_path, parameter_values = build_skip_evaluation_pipeline(config)
    job = aiplatform.PipelineJob(
        display_name=job_id,
        project=config.project_id,
        location=config.location,
        template_path=template_path,
        job_id=job_id,
        pipeline_root=config.root_dir,
        parameter_values=parameter_values,
        enable_caching=False,
        credentials=_get_credentials_or_anonymous(credentials),
    )
    if log_experiment:
        try:
            log_experiment_run(
                run_name=job_id,
                pipeline_job=job,
                config=config,
            )
        except Exception as e:
            logger.warning("Failed to log experiment run for pipeline job '%s': %s", job_id, e)
    return job


def create_distill_skip_evaluation_pipeline_job(
    config: TabularPipelineConfig,
    job_id: str = "automl-tabular-distill-skip-evaluation",
    log_experiment: bool = True,
    credentials: Any | None = None,
) -> aiplatform.PipelineJob:
    """Create a Vertex AI PipelineJob for AutoML Tabular with distillation skipping evaluation."""
    template_path, parameter_values = build_distill_skip_evaluation_pipeline(config)
    job = aiplatform.PipelineJob(
        display_name=job_id,
        project=config.project_id,
        location=config.location,
        template_path=template_path,
        job_id=job_id,
        pipeline_root=config.root_dir,
        parameter_values=parameter_values,
        enable_caching=False,
        credentials=_get_credentials_or_anonymous(credentials),
    )
    if log_experiment:
        try:
            log_experiment_run(
                run_name=job_id,
                pipeline_job=job,
                config=config,
            )
        except Exception as e:
            logger.warning("Failed to log experiment run for pipeline job '%s': %s", job_id, e)
    return job

