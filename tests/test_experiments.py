"""Unit tests for tabflows.experiments module."""

from unittest.mock import MagicMock, patch

from tabflows.config import TabularPipelineConfig
from tabflows.experiments import (
    get_model_evaluation_metrics,
    get_model_feature_attributions,
    list_experiment_runs,
    log_experiment_run,
    run_doe_campaign,
)


def test_get_model_evaluation_metrics_mocked() -> None:
    """Test retrieving evaluation metrics from model evaluation."""
    mock_model = MagicMock()
    mock_eval = MagicMock()
    mock_eval.metrics = {
        "logLoss": 0.25,
        "auPrc": 0.91,
        "auRoc": 0.94,
    }
    mock_model.list_model_evaluations.return_value = [mock_eval]

    metrics = get_model_evaluation_metrics(mock_model)
    assert metrics == {"logLoss": 0.25, "auPrc": 0.91, "auRoc": 0.94}


def test_get_model_feature_attributions_mocked() -> None:
    """Test retrieving feature attributions from model evaluation."""
    mock_model = MagicMock()
    mock_eval = MagicMock()
    mock_eval.metrics = {
        "modelExplanation": {
            "meanAttributions": {
                "featureAttributions": {
                    "age": 0.35,
                    "duration": 0.45,
                    "balance": 0.20,
                }
            }
        }
    }
    mock_model.list_model_evaluations.return_value = [mock_eval]

    attributions = get_model_feature_attributions(mock_model)
    assert attributions == {"age": 0.35, "duration": 0.45, "balance": 0.20}


def test_get_model_evaluation_empty_returns_empty_dict() -> None:
    """Test that model with no evaluations returns empty dictionary."""
    mock_model = MagicMock()
    mock_model.list_model_evaluations.return_value = []

    metrics = get_model_evaluation_metrics(mock_model)
    attributions = get_model_feature_attributions(mock_model)

    assert metrics == {}
    assert attributions == {}


def test_list_experiment_runs_mocked() -> None:
    """Test listing experiment runs dataframe."""
    config = TabularPipelineConfig(
        project_id="test-project",
        location="us-central1",
        experiment_name="custom-experiment",
    )
    mock_df = MagicMock()

    with (
        patch("tabflows.experiments.aiplatform.init") as mock_init,
        patch("tabflows.experiments.aiplatform.get_experiment_df") as mock_get_df,
    ):
        mock_get_df.return_value = mock_df
        df = list_experiment_runs(config=config)

        mock_init.assert_called_once_with(project="test-project", location="us-central1")
        mock_get_df.assert_called_once_with(experiment="custom-experiment")
        assert df == mock_df


def test_log_experiment_run_basic() -> None:
    """Test logging an experiment run with params and metrics."""
    config = TabularPipelineConfig(
        project_id="test-proj",
        location="us-central1",
        experiment_name="test-exp",
    )
    mock_run = MagicMock()

    with (
        patch("tabflows.experiments.aiplatform.init") as mock_init,
        patch("tabflows.experiments.aiplatform.start_run", return_value=mock_run) as mock_start,
        patch("tabflows.experiments.aiplatform.log_params") as mock_log_params,
        patch("tabflows.experiments.aiplatform.log_metrics") as mock_log_metrics,
    ):
        run = log_experiment_run(
            run_name="run-1",
            params={"lr": 0.01},
            metrics={"acc": 0.92},
            config=config,
        )

        mock_init.assert_called_once_with(
            project="test-proj", location="us-central1", experiment="test-exp"
        )
        mock_start.assert_called_once_with(run_name="run-1")
        mock_log_params.assert_called_once_with({"lr": 0.01})
        mock_log_metrics.assert_called_once_with({"acc": 0.92})
        assert run == mock_run


def test_log_experiment_run_with_model() -> None:
    """Test logging experiment run including model evaluation metrics."""
    mock_model = MagicMock()
    mock_run = MagicMock()

    with (
        patch("tabflows.experiments.aiplatform.init"),
        patch("tabflows.experiments.aiplatform.start_run", return_value=mock_run),
        patch("tabflows.experiments.aiplatform.log_metrics") as mock_log_metrics,
        patch(
            "tabflows.experiments.get_model_evaluation_metrics",
            return_value={"logLoss": 0.15},
        ) as mock_get_eval,
    ):
        run = log_experiment_run(run_name="run-model", model=mock_model)

        mock_get_eval.assert_called_once()
        mock_log_metrics.assert_called_once_with({"logLoss": 0.15})
        assert run == mock_run


def test_log_experiment_run_with_pipeline_job_str() -> None:
    """Test logging experiment run with string pipeline job."""
    mock_run = MagicMock()
    job_str = "projects/123/locations/us-central1/pipelineJobs/456"

    with (
        patch("tabflows.experiments.aiplatform.init"),
        patch("tabflows.experiments.aiplatform.start_run", return_value=mock_run),
        patch("tabflows.experiments.aiplatform.log_params") as mock_log_params,
    ):
        run = log_experiment_run(run_name="run-pj-str", pipeline_job=job_str)

        mock_log_params.assert_called_once_with({"pipeline_job_resource_name": job_str})
        assert run == mock_run


def test_log_experiment_run_with_pipeline_job_obj() -> None:
    """Test logging experiment run with PipelineJob object."""
    mock_run = MagicMock()
    mock_job = MagicMock()
    mock_job.resource_name = "projects/123/locations/us-central1/pipelineJobs/789"

    with (
        patch("tabflows.experiments.aiplatform.init"),
        patch("tabflows.experiments.aiplatform.start_run", return_value=mock_run),
        patch("tabflows.experiments.aiplatform.log_params") as mock_log_params,
    ):
        run = log_experiment_run(run_name="run-pj-obj", pipeline_job=mock_job)

        mock_log_params.assert_called_once_with(
            {"pipeline_job_resource_name": "projects/123/locations/us-central1/pipelineJobs/789"}
        )
        mock_run._log_pipeline_job.assert_called_once_with(mock_job)
        assert run == mock_run


def test_run_doe_campaign() -> None:
    """Test running a Design of Experiments campaign across variants."""
    variants = [
        {"name": "baseline", "distill": False},
        {"name": "distilled", "distill": True},
    ]
    mock_run1 = MagicMock()
    mock_run2 = MagicMock()

    with patch(
        "tabflows.experiments.log_experiment_run", side_effect=[mock_run1, mock_run2]
    ) as mock_log_exp:
        summary = run_doe_campaign("exp-camp", variants)

        assert len(summary) == 2
        assert mock_log_exp.call_count == 2

        assert summary[0]["campaign"] == "exp-camp"
        assert summary[0]["variant_name"] == "baseline"
        assert summary[0]["run_name"] == "exp-camp-baseline"
        assert summary[0]["params"] == variants[0]
        assert summary[0]["run"] == mock_run1

        assert summary[1]["variant_name"] == "distilled"
        assert summary[1]["run_name"] == "exp-camp-distilled"
        assert summary[1]["run"] == mock_run2

