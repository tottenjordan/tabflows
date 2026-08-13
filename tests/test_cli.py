"""Unit tests for tabflows CLI commands."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from tabflows.cli import main


def test_run_automl_distill_flag():
    runner = CliRunner()
    with patch("tabflows.cli.build_automl_tabular_pipeline") as mock_build:
        mock_build.return_value = ("gs://test-bucket/template.yaml", {"project": "test-proj"})

        result = runner.invoke(
            main,
            [
                "run-automl",
                "--project",
                "test-proj",
                "--bucket",
                "gs://test-bucket",
                "--compile-only",
                "--distill",
            ],
        )

        assert result.exit_code == 0
        assert "Template compiled at:" in result.output
        mock_build.assert_called_once()
        config_passed = mock_build.call_args[0][0]
        assert config_passed.run_distillation is True


def test_run_automl_no_distill_flag():
    runner = CliRunner()
    with patch("tabflows.cli.build_automl_tabular_pipeline") as mock_build:
        mock_build.return_value = ("gs://test-bucket/template.yaml", {"project": "test-proj"})

        result = runner.invoke(
            main,
            [
                "run-automl",
                "--project",
                "test-proj",
                "--bucket",
                "gs://test-bucket",
                "--compile-only",
                "--no-distill",
            ],
        )

        assert result.exit_code == 0
        mock_build.assert_called_once()
        config_passed = mock_build.call_args[0][0]
        assert config_passed.run_distillation is False


def test_generate_fte_config_local_path(tmp_path: Path):
    runner = CliRunner()
    output_file = tmp_path / "subdir" / "fte_config.json"
    columns_json = '{"age": "numeric", "job": "categorical", "notes": "text"}'

    result = runner.invoke(
        main,
        [
            "generate-fte-config",
            "--output-path",
            str(output_file),
            "--columns-json",
            columns_json,
        ],
    )

    assert result.exit_code == 0
    assert f"FTE configuration written to: {output_file}" in result.output

    assert output_file.exists()
    content = json.loads(output_file.read_text(encoding="utf-8"))
    assert content == [
        {"numeric": {"column_name": "age"}},
        {"categorical": {"column_name": "job"}},
        {"text_embedding": {"column_name": "notes"}},
    ]


def test_generate_fte_config_gcs_path():
    runner = CliRunner()
    gcs_uri = "gs://my-bucket/configs/fte.json"
    columns_json = '{"age": "numeric", "job": "categorical"}'

    with (
        patch("tabflows.cli.storage.Client") as mock_storage_cls,
        patch("tabflows.cli.write_to_gcs") as mock_write_gcs,
    ):
        mock_storage_client = MagicMock()
        mock_storage_cls.return_value = mock_storage_client

        result = runner.invoke(
            main,
            [
                "generate-fte-config",
                "--output-path",
                gcs_uri,
                "--columns-json",
                columns_json,
                "--bucket",
                "gs://my-bucket",
            ],
        )

        assert result.exit_code == 0
        assert f"FTE configuration uploaded to GCS: {gcs_uri}" in result.output
        mock_write_gcs.assert_called_once()
        call_args = mock_write_gcs.call_args[0]
        assert call_args[1] == gcs_uri
        parsed_written_json = json.loads(call_args[2])
        assert parsed_written_json == [
            {"numeric": {"column_name": "age"}},
            {"categorical": {"column_name": "job"}},
        ]


def test_generate_fte_config_invalid_json():
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "generate-fte-config",
            "--output-path",
            "fte.json",
            "--columns-json",
            "not valid json",
        ],
    )

    assert result.exit_code != 0
    assert "Error parsing --columns-json JSON:" in result.output


def test_generate_fte_config_non_dict_json():
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "generate-fte-config",
            "--output-path",
            "fte.json",
            "--columns-json",
            '["age", "job"]',
        ],
    )

    assert result.exit_code != 0
    assert "Error: --columns-json must be a JSON object" in result.output


def test_generate_fte_config_invalid_column_type():
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "generate-fte-config",
            "--output-path",
            "fte.json",
            "--columns-json",
            '{"age": "unsupported_type"}',
        ],
    )

    assert result.exit_code != 0
    assert "Error generating FTE transformations:" in result.output


def test_log_experiment_success():
    runner = CliRunner()
    with patch("tabflows.cli.log_experiment_run") as mock_log:
        result = runner.invoke(
            main,
            [
                "log-experiment",
                "--run-name",
                "test-run-1",
                "--model-resource-name",
                "projects/123/locations/us-central1/models/456",
                "--pipeline-job-id",
                "job-789",
                "--params-json",
                '{"lr": 0.01, "optimizer": "adam"}',
                "--metrics-json",
                '{"accuracy": 0.95}',
                "--project",
                "my-project",
                "--location",
                "us-central1",
            ],
        )

        assert result.exit_code == 0
        assert "Successfully logged experiment run 'test-run-1'" in result.output
        mock_log.assert_called_once()
        _, kwargs = mock_log.call_args
        assert kwargs["run_name"] == "test-run-1"
        assert kwargs["model"] == "projects/123/locations/us-central1/models/456"
        assert kwargs["pipeline_job"] == "job-789"
        assert kwargs["params"] == {"lr": 0.01, "optimizer": "adam"}
        assert kwargs["metrics"] == {"accuracy": 0.95}
        assert kwargs["config"].project_id == "my-project"
        assert kwargs["config"].location == "us-central1"


def test_log_experiment_invalid_params_json():
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "log-experiment",
            "--run-name",
            "test-run-1",
            "--params-json",
            "invalid json",
        ],
    )

    assert result.exit_code != 0
    assert "Error parsing --params-json JSON:" in result.output


def test_log_experiment_non_dict_params_json():
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "log-experiment",
            "--run-name",
            "test-run-1",
            "--params-json",
            '["lr", 0.01]',
        ],
    )

    assert result.exit_code != 0
    assert "Error: --params-json must be a JSON object." in result.output


def test_log_experiment_invalid_metrics_json():
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "log-experiment",
            "--run-name",
            "test-run-1",
            "--metrics-json",
            "invalid json",
        ],
    )

    assert result.exit_code != 0
    assert "Error parsing --metrics-json JSON:" in result.output


def test_log_experiment_non_dict_metrics_json():
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "log-experiment",
            "--run-name",
            "test-run-1",
            "--metrics-json",
            "[1, 2, 3]",
        ],
    )

    assert result.exit_code != 0
    assert "Error: --metrics-json must be a JSON object." in result.output


def test_deploy_endpoint_with_traffic_split_json():
    runner = CliRunner()
    mock_endpoint = MagicMock()
    mock_endpoint.resource_name = "projects/123/locations/us-central1/endpoints/ep123"

    with patch("tabflows.cli.deploy_model_to_endpoint") as mock_deploy:
        mock_deploy.return_value = mock_endpoint

        result = runner.invoke(
            main,
            [
                "deploy-endpoint",
                "--model",
                "projects/123/locations/us-central1/models/456",
                "--project",
                "my-project",
                "--location",
                "us-central1",
                "--traffic-split-json",
                '{"0": 90, "1": 10}',
            ],
        )

        assert result.exit_code == 0
        assert "Endpoint deployment initiated:" in result.output
        mock_deploy.assert_called_once()
        _, kwargs = mock_deploy.call_args
        assert kwargs["model"] == "projects/123/locations/us-central1/models/456"
        assert kwargs["traffic_split"] == {"0": 90, "1": 10}


def test_deploy_endpoint_invalid_traffic_split_json():
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "deploy-endpoint",
            "--model",
            "m123",
            "--traffic-split-json",
            "invalid json",
        ],
    )

    assert result.exit_code != 0
    assert "Error parsing --traffic-split-json JSON:" in result.output


def test_deploy_endpoint_non_dict_traffic_split_json():
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "deploy-endpoint",
            "--model",
            "m123",
            "--traffic-split-json",
            "[90, 10]",
        ],
    )

    assert result.exit_code != 0
    assert "Error: --traffic-split-json must be a JSON object." in result.output

