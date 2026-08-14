"""Unit tests for pipeline helper functions and component builders."""

from unittest.mock import MagicMock, patch

import pytest

from tabflows.config import TabularPipelineConfig
from tabflows.pipeline import (
    build_automl_tabular_pipeline,
    build_skip_architecture_search_pipeline,
    create_tabular_pipeline_job,
    generate_auto_transformation,
    generate_fte_transformations,
    get_bucket_name_and_path,
    get_task_detail,
    run_skip_architecture_search_pipeline,
    setup_gcp_resources,
    write_fte_transformations,
)


def test_generate_auto_transformation():
    columns = ["age", "job", "balance"]
    transformations = generate_auto_transformation(columns)

    assert len(transformations) == 3
    assert transformations[0] == {"auto": {"column_name": "age"}}
    assert transformations[1] == {"auto": {"column_name": "job"}}
    assert transformations[2] == {"auto": {"column_name": "balance"}}


def test_generate_fte_transformations_mixed_types():
    column_types = {
        "age": "numeric",
        "job": "categorical",
        "signup_date": "timestamp",
        "description": "text",
        "unknown_col": "auto",
    }
    transformations = generate_fte_transformations(column_types)

    assert len(transformations) == 5
    assert transformations[0] == {"numeric": {"column_name": "age"}}
    assert transformations[1] == {"categorical": {"column_name": "job"}}
    assert transformations[2] == {"timestamp": {"column_name": "signup_date"}}
    assert transformations[3] == {"text_embedding": {"column_name": "description"}}
    assert transformations[4] == {"auto": {"column_name": "unknown_col"}}


def test_generate_fte_transformations_invalid_type():
    with pytest.raises(ValueError, match="Unsupported FTE column type"):
        generate_fte_transformations({"col1": "invalid_type"})


def test_write_fte_transformations():
    mock_storage_client = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_storage_client.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob

    column_types = {"age": "numeric", "job": "categorical"}
    write_fte_transformations(mock_storage_client, "gs://test-bucket/fte_spec.json", column_types)

    mock_storage_client.bucket.assert_called_once_with("test-bucket")
    mock_bucket.blob.assert_called_once_with("fte_spec.json")
    mock_blob.upload_from_string.assert_called_once()
    uploaded_content = mock_blob.upload_from_string.call_args[0][0]
    assert '"numeric"' in uploaded_content
    assert '"categorical"' in uploaded_content


def test_get_bucket_name_and_path_valid():
    bucket_name, path = get_bucket_name_and_path("gs://my-bucket/path/to/object.json")
    assert bucket_name == "my-bucket"
    assert path == "path/to/object.json"

    bucket_name, path = get_bucket_name_and_path("gs://my-bucket")
    assert bucket_name == "my-bucket"
    assert path == ""


def test_get_bucket_name_and_path_invalid():
    with pytest.raises(ValueError, match="Invalid GCS URI"):
        get_bucket_name_and_path("http://example.com/file.json")


def test_setup_gcp_resources_mocked():
    mock_storage_client = MagicMock()
    mock_bucket = MagicMock()
    mock_bucket.exists.return_value = False
    mock_storage_client.bucket.return_value = mock_bucket
    mock_storage_client.create_bucket.return_value = mock_bucket

    config = TabularPipelineConfig(
        project_id="test-proj",
        bucket_uri="gs://test-bucket",
    )

    summary = setup_gcp_resources(config, storage_client=mock_storage_client)

    assert summary["bucket_name"] == "test-bucket"
    assert summary["bucket_created"] == "True"
    assert (
        summary["transform_config_path"]
        == "gs://test-bucket/automl_tabular_pipeline/transform_config_unique.json"
    )
    mock_storage_client.create_bucket.assert_called_once()


def test_get_task_detail():
    task1 = MagicMock()
    task1.task_name = "automl-tabular-stage-1-tuner"

    task2 = MagicMock()
    task2.task_name = "model-upload"

    task_details = [task1, task2]

    found = get_task_detail(task_details, "automl-tabular-stage-1-tuner")
    assert found == task1

    not_found = get_task_detail(task_details, "non-existent-task")
    assert not_found is None


def test_build_automl_tabular_pipeline():
    config = TabularPipelineConfig(
        project_id="test-project",
        bucket_uri="gs://test-bucket",
    )

    template_path, parameter_values = build_automl_tabular_pipeline(config)
    assert isinstance(template_path, str)
    assert isinstance(parameter_values, dict)
    assert parameter_values["project"] == "test-project"
    assert parameter_values["target_column"] == "deposit"


def test_build_skip_architecture_search_pipeline():
    config = TabularPipelineConfig(
        project_id="test-project",
        bucket_uri="gs://test-bucket",
    )
    tuning_uri = "gs://test-bucket/automl_tabular_pipeline/tuning_result"

    template_path, parameter_values = build_skip_architecture_search_pipeline(config, tuning_uri)
    assert isinstance(template_path, str)
    assert isinstance(parameter_values, dict)
    assert parameter_values["project"] == "test-project"
    assert parameter_values["stage_1_tuning_result_artifact_uri"] == tuning_uri


def test_create_tabular_pipeline_job_with_experiment():
    from google.cloud import aiplatform

    config = TabularPipelineConfig(
        project_id="test-project",
        bucket_uri="gs://test-bucket",
        experiment_name="test-exp",
    )

    with (
        patch("tabflows.pipeline.log_experiment_run") as mock_log_exp,
    ):
        job = create_tabular_pipeline_job(config, job_id="test-job-id", log_experiment=True)
        assert isinstance(job, aiplatform.PipelineJob)
        mock_log_exp.assert_called_once_with(
            run_name="test-job-id",
            pipeline_job=job,
            config=config,
        )


def test_create_tabular_pipeline_job_without_experiment():
    from google.cloud import aiplatform

    config = TabularPipelineConfig(
        project_id="test-project",
        bucket_uri="gs://test-bucket",
    )

    with (
        patch("tabflows.pipeline.log_experiment_run") as mock_log_exp,
    ):
        job = create_tabular_pipeline_job(config, job_id="test-job-id", log_experiment=False)
        assert isinstance(job, aiplatform.PipelineJob)
        mock_log_exp.assert_not_called()


def test_run_skip_architecture_search_pipeline_with_experiment():
    from google.cloud import aiplatform

    config = TabularPipelineConfig(
        project_id="test-project",
        bucket_uri="gs://test-bucket",
        tuning_result_output="gs://test-bucket/tuning_result",
        experiment_name="test-exp",
    )

    with (
        patch("tabflows.pipeline.log_experiment_run") as mock_log_exp,
    ):
        job = run_skip_architecture_search_pipeline(
            config, job_id="test-skip-job", log_experiment=True
        )
        assert isinstance(job, aiplatform.PipelineJob)
        mock_log_exp.assert_called_once_with(
            run_name="test-skip-job",
            pipeline_job=job,
            config=config,
        )


def test_run_skip_architecture_search_pipeline_without_experiment():
    from google.cloud import aiplatform

    config = TabularPipelineConfig(
        project_id="test-project",
        bucket_uri="gs://test-bucket",
        tuning_result_output="gs://test-bucket/tuning_result",
    )

    with (
        patch("tabflows.pipeline.aiplatform.init") as mock_init,
        patch("tabflows.pipeline.log_experiment_run") as mock_log_exp,
    ):
        job = run_skip_architecture_search_pipeline(
            config, job_id="test-skip-job", log_experiment=False
        )
        assert isinstance(job, aiplatform.PipelineJob)
        mock_init.assert_not_called()
        mock_log_exp.assert_not_called()

    # Test error when tuning_result_output missing
    config_no_tuning = TabularPipelineConfig(
        project_id="test-project",
        bucket_uri="gs://test-bucket",
    )
    with pytest.raises(ValueError, match="tuning_result_output must be provided"):
        run_skip_architecture_search_pipeline(config_no_tuning, log_experiment=False)


def test_build_automl_tabular_pipeline_bigquery_and_predefined_split():
    config = TabularPipelineConfig(
        project_id="test-project",
        bucket_uri="gs://test-bucket",
        bigquery_table_path="bq://test-project.dataset.table",
        predefined_split_key="split_col",
    )

    template_path, parameter_values = build_automl_tabular_pipeline(config)
    assert isinstance(template_path, str)
    assert isinstance(parameter_values, dict)
    assert parameter_values["data_source_bigquery_table_path"] == "bq://test-project.dataset.table"
    assert parameter_values["predefined_split_key"] == "split_col"


def test_build_skip_architecture_search_pipeline_bigquery_and_predefined_split():
    config = TabularPipelineConfig(
        project_id="test-project",
        bucket_uri="gs://test-bucket",
        bigquery_table_path="bq://test-project.dataset.table",
        predefined_split_key="split_col",
    )
    tuning_uri = "gs://test-bucket/automl_tabular_pipeline/tuning_result"

    template_path, parameter_values = build_skip_architecture_search_pipeline(config, tuning_uri)
    assert isinstance(template_path, str)
    assert isinstance(parameter_values, dict)
    assert parameter_values["data_source_bigquery_table_path"] == "bq://test-project.dataset.table"
    assert parameter_values["predefined_split_key"] == "split_col"


def test_build_automl_tabular_pipeline_specialized_objectives():
    config = TabularPipelineConfig(
        project_id="test-project",
        bucket_uri="gs://test-bucket",
        optimization_objective="maximize-precision-at-recall",
        optimization_objective_recall_value=0.95,
    )

    template_path, parameter_values = build_automl_tabular_pipeline(config)
    assert isinstance(template_path, str)
    assert isinstance(parameter_values, dict)
    assert parameter_values["optimization_objective"] == "maximize-precision-at-recall"
    assert parameter_values["optimization_objective_recall_value"] == 0.95


def test_create_pipeline_job_explicit_project_param(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("GCP_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)

    config = TabularPipelineConfig(
        project_id="test-project",
        bucket_uri="gs://test-bucket",
    )

    job = create_tabular_pipeline_job(config, job_id="test-job-explicit", log_experiment=False)
    assert job.project == config.project_id


def test_create_pipeline_job_unauthenticated_anonymous_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from google.auth.credentials import AnonymousCredentials
    from google.auth.exceptions import DefaultCredentialsError

    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("GCP_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)

    config = TabularPipelineConfig(
        project_id="test-project",
        bucket_uri="gs://test-bucket",
    )

    with patch("google.auth.default", side_effect=DefaultCredentialsError("No credentials")):
        job = create_tabular_pipeline_job(config, log_experiment=False)
        assert job.project == config.project_id
        assert isinstance(job.credentials, AnonymousCredentials)




