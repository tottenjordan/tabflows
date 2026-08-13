"""Unit tests for tabflows.experiments module."""

from unittest.mock import MagicMock, patch

from tabflows.config import TabularPipelineConfig
from tabflows.experiments import (
    get_model_evaluation_metrics,
    get_model_feature_attributions,
    list_experiment_runs,
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
