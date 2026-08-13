"""Unit tests for tabflows.inference module."""

from unittest.mock import MagicMock, patch

import pytest

from tabflows.config import TabularPipelineConfig
from tabflows.inference import (
    cleanup_endpoint,
    deploy_model_to_endpoint,
    get_model_from_pipeline_job,
    list_models,
    predict_online,
    run_batch_prediction,
)


def test_list_models_mocked() -> None:
    """Test listing recent models from Vertex AI Model Registry."""
    config = TabularPipelineConfig(project_id="test-project", location="us-central1")
    mock_models = [MagicMock(), MagicMock()]

    with (
        patch("tabflows.inference.aiplatform.init") as mock_init,
        patch("tabflows.inference.aiplatform.Model.list") as mock_list,
    ):
        mock_list.return_value = mock_models
        models = list_models(config=config, limit=2)

        mock_init.assert_called_once_with(project="test-project", location="us-central1")
        mock_list.assert_called_once_with(order_by="create_time desc")
        assert models == mock_models


def test_get_model_from_pipeline_job_mocked() -> None:
    """Test retrieving model resource name from pipeline job task details."""
    mock_job = MagicMock()

    mock_artifact = MagicMock()
    mock_artifact.metadata = {
        "resourceName": "projects/123/locations/us-central1/models/test-model-456"
    }

    mock_task = MagicMock()
    mock_task.task_name = "model-upload"
    mock_task.outputs = {"model": MagicMock(artifacts=[mock_artifact])}

    mock_job.gca_resource.job_detail.task_details = [mock_task]

    with patch("tabflows.inference.aiplatform.Model") as mock_model_cls:
        mock_model_cls.return_value = MagicMock()
        model = get_model_from_pipeline_job(mock_job)
        mock_model_cls.assert_called_once_with(
            model_name="projects/123/locations/us-central1/models/test-model-456"
        )
        assert model == mock_model_cls.return_value


def test_get_model_from_pipeline_job_missing_task_raises() -> None:
    """Test that missing model-upload task raises ValueError."""
    mock_job = MagicMock()
    mock_job.gca_resource.job_detail.task_details = []

    with pytest.raises(ValueError, match="Task 'model-upload' not found or does not contain"):
        get_model_from_pipeline_job(mock_job)


def test_deploy_model_to_endpoint_mocked() -> None:
    """Test deploying model to online endpoint with config settings."""
    config = TabularPipelineConfig(
        project_id="test-project",
        location="us-central1",
        serving_machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=2,
    )

    mock_model = MagicMock()
    mock_endpoint = MagicMock()

    with (
        patch("tabflows.inference.aiplatform.init") as mock_init,
        patch("tabflows.inference.aiplatform.Endpoint.create") as mock_create_endpoint,
    ):
        mock_create_endpoint.return_value = mock_endpoint

        endpoint = deploy_model_to_endpoint(
            model=mock_model,
            config=config,
            endpoint_display_name="custom-endpoint-name",
        )

        mock_init.assert_called_once_with(project="test-project", location="us-central1")
        mock_create_endpoint.assert_called_once_with(display_name="custom-endpoint-name")
        mock_model.deploy.assert_called_once_with(
            endpoint=mock_endpoint,
            machine_type="n1-standard-4",
            min_replica_count=1,
            max_replica_count=2,
            sync=True,
        )
        assert endpoint == mock_endpoint


def test_predict_online_mocked() -> None:
    """Test online prediction response parsing."""
    mock_endpoint = MagicMock()
    mock_response = MagicMock()
    mock_response.predictions = [
        {"classes": ["0", "1"], "scores": [0.85, 0.15]},
    ]
    mock_endpoint.predict.return_value = mock_response

    instances = [{"age": "30", "job": "blue-collar", "marital": "married"}]
    results = predict_online(endpoint=mock_endpoint, instances=instances)

    mock_endpoint.predict.assert_called_once_with(instances=instances)
    assert results == [{"classes": ["0", "1"], "scores": [0.85, 0.15]}]


def test_run_batch_prediction_mocked() -> None:
    """Test submitting batch prediction job."""
    config = TabularPipelineConfig(
        project_id="test-project",
        location="us-central1",
        bucket_uri="gs://test-bucket",
        batch_predict_instances_format="csv",
        batch_predict_predictions_format="jsonl",
    )

    mock_model = MagicMock()
    mock_batch_job = MagicMock()
    mock_model.batch_predict.return_value = mock_batch_job

    with patch("tabflows.inference.aiplatform.init") as mock_init:
        batch_job = run_batch_prediction(
            model=mock_model,
            config=config,
            gcs_source="gs://test-bucket/test_data.csv",
            gcs_destination_prefix="gs://test-bucket/batch_output",
            job_display_name="custom-batch-job",
        )

        mock_init.assert_called_once_with(project="test-project", location="us-central1")
        mock_model.batch_predict.assert_called_once_with(
            job_display_name="custom-batch-job",
            gcs_source=["gs://test-bucket/test_data.csv"],
            gcs_destination_prefix="gs://test-bucket/batch_output",
            instances_format="csv",
            predictions_format="jsonl",
            machine_type="n1-standard-4",
            sync=False,
        )
        assert batch_job == mock_batch_job


def test_cleanup_endpoint_mocked() -> None:
    """Test cleaning up endpoint by undeploying models and deleting resource."""
    mock_endpoint = MagicMock()

    cleanup_endpoint(endpoint=mock_endpoint, delete_endpoint=True)

    mock_endpoint.undeploy_all.assert_called_once()
    mock_endpoint.delete.assert_called_once()
