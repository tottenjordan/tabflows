"""Unit tests for TabularPipelineConfig schema and environment variable loading."""

import pytest

from tabflows.config import TabularPipelineConfig


def test_tabular_pipeline_config_defaults():
    config = TabularPipelineConfig(
        project_id="test-project",
        bucket_uri="gs://my-test-bucket",
    )

    assert config.project_id == "test-project"
    assert config.location == "us-central1"
    assert config.prediction_type == "classification"
    assert config.optimization_objective == "minimize-log-loss"
    assert config.target_column == "deposit"
    assert config.fte_transformations_path is None
    assert config.root_dir == "gs://my-test-bucket/automl_tabular_pipeline"
    assert (
        config.transform_config_path
        == "gs://my-test-bucket/automl_tabular_pipeline/transform_config_unique.json"
    )
    assert len(config.features) == 16
    assert config.stratified_split_key is None
    assert config.skip_evaluation is False
    assert config.export_additional_model_without_custom_ops is False


def test_tabular_pipeline_config_env_vars(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GCP_PROJECT", "env-project-123")
    monkeypatch.setenv("GCP_LOCATION", "us-east1")
    monkeypatch.setenv("GCP_BUCKET_URI", "gs://env-bucket-456")
    monkeypatch.setenv("FTE_TRANSFORMATIONS_PATH", "gs://env-bucket-456/fte_spec.json")

    config = TabularPipelineConfig()

    assert config.project_id == "env-project-123"
    assert config.location == "us-east1"
    assert config.bucket_uri == "gs://env-bucket-456"
    assert config.fte_transformations_path == "gs://env-bucket-456/fte_spec.json"
    assert config.root_dir == "gs://env-bucket-456/automl_tabular_pipeline"


def test_tabular_pipeline_config_custom_values():
    config = TabularPipelineConfig(
        project_id="custom-project",
        location="europe-west4",
        bucket_uri="gs://custom-bucket/",
        root_dir_name="custom_pipeline",
        target_column="label",
        prediction_type="regression",
        optimization_objective="minimize-rmse",
    )

    assert config.project_id == "custom-project"
    assert config.location == "europe-west4"
    assert config.root_dir == "gs://custom-bucket/custom_pipeline"
    assert (
        config.transform_config_path
        == "gs://custom-bucket/custom_pipeline/transform_config_unique.json"
    )
    assert config.prediction_type == "regression"
    assert config.bigquery_table_path is None
    assert config.predefined_split_key is None


def test_tabular_pipeline_config_bigquery_and_split_key():
    config = TabularPipelineConfig(
        project_id="custom-project",
        bucket_uri="gs://custom-bucket/",
        bigquery_table_path="bq://proj.ds.tbl",
        predefined_split_key="split_flag",
    )

    assert config.bigquery_table_path == "bq://proj.ds.tbl"
    assert config.predefined_split_key == "split_flag"


def test_tabular_pipeline_config_specialized_objectives():
    config = TabularPipelineConfig(
        project_id="test-project",
        bucket_uri="gs://test-bucket",
        optimization_objective="maximize-precision-at-recall",
        optimization_objective_recall_value=0.95,
    )

    assert config.optimization_objective == "maximize-precision-at-recall"
    assert config.optimization_objective_recall_value == 0.95
    assert config.optimization_objective_precision_value is None


def test_tabular_pipeline_config_stratified_split_and_skip_eval():
    config = TabularPipelineConfig(
        project_id="test-project",
        bucket_uri="gs://test-bucket",
        stratified_split_key="strata_col",
        skip_evaluation=True,
        export_additional_model_without_custom_ops=True,
        weight_column="sample_w",
    )

    assert config.stratified_split_key == "strata_col"
    assert config.skip_evaluation is True
    assert config.export_additional_model_without_custom_ops is True
    assert config.weight_column == "sample_w"


def test_tabular_pipeline_config_probabilistic_inference_and_metadata():
    config = TabularPipelineConfig(
        project_id="test-project",
        bucket_uri="gs://test-bucket",
        enable_probabilistic_inference=True,
        quantiles=[0.1, 0.5, 0.9],
        model_display_name="churn-champion-v1",
        model_description="Automated production model",
        dataflow_service_account="sa@project.iam.gserviceaccount.com",
        evaluation_batch_predict_machine_type="n1-highmem-8",
        evaluation_batch_predict_max_replica_count=10,
        evaluation_batch_explain_machine_type="n1-highmem-8",
        evaluation_batch_explain_max_replica_count=5,
    )

    assert config.enable_probabilistic_inference is True
    assert config.quantiles == [0.1, 0.5, 0.9]
    assert config.model_display_name == "churn-champion-v1"
    assert config.model_description == "Automated production model"
    assert config.dataflow_service_account == "sa@project.iam.gserviceaccount.com"
    assert config.evaluation_batch_predict_machine_type == "n1-highmem-8"
    assert config.evaluation_batch_predict_max_replica_count == 10
    assert config.evaluation_batch_explain_machine_type == "n1-highmem-8"
    assert config.evaluation_batch_explain_max_replica_count == 5


def test_tabular_pipeline_config_feature_selection():
    config = TabularPipelineConfig(
        project_id="test-project",
        bucket_uri="gs://test-bucket",
        feature_selection_algorithm="mrmr",
        feature_selection_engine="BigQuery",
        max_selected_features=20,
        num_selected_features=15,
        apply_feature_selection_tuning=True,
        bigquery_staging_full_dataset_id="test-project.staging_ds",
        dataflow_machine_type="n1-standard-4",
        dataflow_max_num_workers=25,
        dataflow_disk_size_gb=100,
        encryption_spec_key_name="projects/p/locations/l/keyRings/r/cryptoKeys/k",
    )

    assert config.feature_selection_algorithm == "MRMR"
    assert config.feature_selection_engine == "bigquery"
    assert config.max_selected_features == 20
    assert config.num_selected_features == 15
    assert config.apply_feature_selection_tuning is True
    assert config.bigquery_staging_full_dataset_id == "test-project.staging_ds"
    assert config.dataflow_machine_type == "n1-standard-4"
    assert config.dataflow_max_num_workers == 25
    assert config.dataflow_disk_size_gb == 100
    assert config.encryption_spec_key_name == "projects/p/locations/l/keyRings/r/cryptoKeys/k"


def test_tabular_pipeline_config_feature_selection_validation_errors():
    import pytest

    with pytest.raises(ValueError, match="Invalid feature_selection_algorithm"):
        TabularPipelineConfig(
            project_id="test-project",
            bucket_uri="gs://test-bucket",
            feature_selection_algorithm="INVALID_ALGO",
        )

    with pytest.raises(ValueError, match="Invalid feature_selection_engine"):
        TabularPipelineConfig(
            project_id="test-project",
            bucket_uri="gs://test-bucket",
            feature_selection_engine="spark",
        )

