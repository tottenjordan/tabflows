"""Inference utilities for Vertex AI AutoML Tabular models (Online & Batch)."""

import datetime
from typing import Any

from google.cloud import aiplatform

from tabflows.config import TabularPipelineConfig
from tabflows.pipeline import get_task_detail


def list_models(
    config: TabularPipelineConfig | None = None,
    limit: int = 5,
) -> list[aiplatform.Model]:
    """List recent Vertex AI Models in the project."""
    if config is None:
        config = TabularPipelineConfig()

    aiplatform.init(project=config.project_id, location=config.location)
    models = aiplatform.Model.list(order_by="create_time desc")
    return models[:limit]


def get_model_from_pipeline_job(
    pipeline_job: str | aiplatform.PipelineJob,
) -> aiplatform.Model:
    """Retrieve the trained aiplatform.Model from a completed PipelineJob or resource name."""
    if isinstance(pipeline_job, str):
        pipeline_job = aiplatform.PipelineJob.get(pipeline_job)

    task_details = pipeline_job.gca_resource.job_detail.task_details
    model_upload_task = get_task_detail(task_details, "model-upload")

    if (
        not model_upload_task
        or not model_upload_task.outputs
        or "model" not in model_upload_task.outputs
    ):
        raise ValueError(
            "Task 'model-upload' not found or does not contain 'model' output artifact."
        )

    model_resource_name = model_upload_task.outputs["model"].artifacts[0].metadata["resourceName"]
    return aiplatform.Model(model_name=model_resource_name)


def deploy_model_to_endpoint(
    model: str | aiplatform.Model,
    config: TabularPipelineConfig | None = None,
    endpoint_display_name: str | None = None,
    sync: bool = True,
) -> aiplatform.Endpoint:
    """Deploy an AutoML Tabular model to a real-time Vertex AI Endpoint."""
    if config is None:
        config = TabularPipelineConfig()

    if isinstance(model, str):
        model = aiplatform.Model(model_name=model)

    if not endpoint_display_name:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        endpoint_display_name = f"automl-tabular-endpoint-{timestamp}"

    aiplatform.init(project=config.project_id, location=config.location)

    endpoint = aiplatform.Endpoint.create(display_name=endpoint_display_name)
    model.deploy(
        endpoint=endpoint,
        machine_type=config.serving_machine_type,
        min_replica_count=config.min_replica_count,
        max_replica_count=config.max_replica_count,
        sync=sync,
    )
    return endpoint


def predict_online(
    endpoint: str | aiplatform.Endpoint,
    instances: list[dict[str, Any]],
) -> list[Any]:
    """Execute real-time online predictions against a deployed Vertex AI Endpoint."""
    if isinstance(endpoint, str):
        endpoint = aiplatform.Endpoint(endpoint_name=endpoint)

    prediction_response = endpoint.predict(instances=instances)
    return list(prediction_response.predictions)


def run_batch_prediction(
    model: str | aiplatform.Model,
    config: TabularPipelineConfig | None = None,
    gcs_source: str | list[str] = "",
    gcs_destination_prefix: str | None = None,
    job_display_name: str | None = None,
) -> aiplatform.BatchPredictionJob:
    """Submit a Batch Prediction job for an AutoML Tabular model against GCS data sources."""
    if config is None:
        config = TabularPipelineConfig()

    if isinstance(model, str):
        model = aiplatform.Model(model_name=model)

    if not gcs_destination_prefix:
        gcs_destination_prefix = f"{config.root_dir}/batch_predictions"

    if isinstance(gcs_source, str):
        gcs_source = [gcs_source]

    if not job_display_name:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        job_display_name = f"automl-tabular-batch-{timestamp}"

    aiplatform.init(project=config.project_id, location=config.location)

    batch_job = model.batch_predict(
        job_display_name=job_display_name,
        gcs_source=gcs_source,
        gcs_destination_prefix=gcs_destination_prefix,
        instances_format=config.batch_predict_instances_format,
        predictions_format=config.batch_predict_predictions_format,
        machine_type=config.serving_machine_type,
        sync=False,
    )
    return batch_job


def cleanup_endpoint(
    endpoint: str | aiplatform.Endpoint,
    delete_endpoint: bool = True,
) -> None:
    """Undeploy all models from an endpoint and optionally delete the endpoint resource."""
    if isinstance(endpoint, str):
        endpoint = aiplatform.Endpoint(endpoint_name=endpoint)

    endpoint.undeploy_all()
    if delete_endpoint:
        endpoint.delete()
