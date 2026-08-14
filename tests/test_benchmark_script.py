"""Unit tests for scripts/benchmark_bakeoff_deployment.py script."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.benchmark_bakeoff_deployment import (  # noqa: E402
    benchmark_deployment,
    calculate_latency_stats,
    get_bakeoff_model,
)
from tabflows.config import TabularPipelineConfig  # noqa: E402


def test_calculate_latency_stats_known_array() -> None:
    """Test calculate_latency_stats with a known latency array."""
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0, 100.0]
    stats = calculate_latency_stats(latencies)

    assert stats["p50"] == 35.0
    assert stats["p90"] == 75.0
    assert stats["p95"] == 87.5
    assert stats["qps"] == 24.0


def test_calculate_latency_stats_empty_list() -> None:
    """Test calculate_latency_stats with an empty list."""
    stats = calculate_latency_stats([])

    assert stats == {"p50": 0.0, "p90": 0.0, "p95": 0.0, "qps": 0.0}


def test_get_bakeoff_model_by_resource_name() -> None:
    """Test get_bakeoff_model with resource name starting with projects/."""
    config = TabularPipelineConfig(project_id="test-proj", location="us-central1")
    resource_name = "projects/123/locations/us-central1/models/m123"

    with (
        patch("scripts.benchmark_bakeoff_deployment.aiplatform.init") as mock_init,
        patch("scripts.benchmark_bakeoff_deployment.aiplatform.Model") as mock_model_cls,
    ):
        mock_model_cls.return_value = MagicMock()
        model = get_bakeoff_model(resource_name, config)

        mock_init.assert_called_once_with(project="test-proj", location="us-central1")
        mock_model_cls.assert_called_once_with(model_name=resource_name)
        assert model == mock_model_cls.return_value


def test_get_bakeoff_model_by_display_name_found() -> None:
    """Test get_bakeoff_model searching by display_name."""
    config = TabularPipelineConfig(project_id="test-proj", location="us-central1")
    mock_model = MagicMock()

    with (
        patch("scripts.benchmark_bakeoff_deployment.aiplatform.init"),
        patch(
            "scripts.benchmark_bakeoff_deployment.aiplatform.Model.list", return_value=[mock_model]
        ) as mock_list,
    ):
        model = get_bakeoff_model("bakeoff-baseline-full-search", config)

        mock_list.assert_called_once_with(
            filter='display_name="bakeoff-baseline-full-search"',
            order_by="create_time desc",
        )
        assert model == mock_model


def test_get_bakeoff_model_not_found_raises() -> None:
    """Test get_bakeoff_model raising ValueError when model is not found."""
    config = TabularPipelineConfig(project_id="test-proj", location="us-central1")

    with (
        patch("scripts.benchmark_bakeoff_deployment.aiplatform.init"),
        patch(
            "scripts.benchmark_bakeoff_deployment.aiplatform.Model.list", return_value=[]
        ) as mock_list,
    ):
        with pytest.raises(ValueError, match="Bake-off model 'missing-model' not found"):
            get_bakeoff_model("missing-model", config)
        assert mock_list.call_count == 2


def test_benchmark_deployment_mocked() -> None:
    """Test end-to-end benchmark_deployment execution with mocked dependencies."""
    config = TabularPipelineConfig(project_id="test-proj", location="us-central1")
    mock_champion = MagicMock()
    mock_challenger = MagicMock()
    mock_endpoint = MagicMock()

    with (
        patch("scripts.benchmark_bakeoff_deployment.get_bakeoff_model") as mock_get_model,
        patch(
            "scripts.benchmark_bakeoff_deployment.deploy_model_to_endpoint",
            return_value=mock_endpoint,
        ) as mock_deploy,
        patch(
            "scripts.benchmark_bakeoff_deployment.predict_online",
            return_value=[{"classes": ["0", "1"], "scores": [0.9, 0.1]}],
        ) as mock_predict,
        patch("scripts.benchmark_bakeoff_deployment.cleanup_endpoint") as mock_cleanup,
    ):
        mock_get_model.side_effect = [mock_champion, mock_challenger]

        stats = benchmark_deployment(config=config, num_requests=5)

        assert mock_get_model.call_count == 2
        assert mock_deploy.call_count == 2
        assert mock_predict.call_count == 5
        mock_cleanup.assert_called_once_with(endpoint=mock_endpoint, delete_endpoint=True)

        assert "p50" in stats
        assert "p90" in stats
        assert "p95" in stats
        assert "qps" in stats
        assert stats["qps"] > 0
