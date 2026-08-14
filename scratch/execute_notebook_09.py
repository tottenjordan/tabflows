"""Execution helper script for Notebook 09."""

import io
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import MagicMock, patch

sys.path.insert(0, ".")

notebook_path = "notebooks/09_bakeoff_live_deployment_and_benchmarking.ipynb"
with open(notebook_path, encoding="utf-8") as f:
    nb = json.load(f)

exec_globals = {}

mock_teacher_model = MagicMock()
mock_teacher_model.resource_name = (
    "projects/test/locations/us-central1/models/bakeoff-baseline-full-search"
)
mock_teacher_model.display_name = "bakeoff-baseline-full-search"

mock_student_model = MagicMock()
mock_student_model.resource_name = (
    "projects/test/locations/us-central1/models/bakeoff-distilled-student-v2"
)
mock_student_model.display_name = "bakeoff-distilled-student-v2"

mock_endpoint = MagicMock()
mock_endpoint.resource_name = (
    "projects/test/locations/us-central1/endpoints/bakeoff-endpoint-123"
)
mock_endpoint.display_name = "bakeoff-live-canary-endpoint"
mock_endpoint.predict.return_value = MagicMock(
    predictions=[{"classes": ["0", "1"], "scores": [0.85, 0.15]}]
)


def mock_get_model(name, config=None):
    if "student" in name:
        return mock_student_model
    return mock_teacher_model


with (
    patch("scripts.benchmark_bakeoff_deployment.aiplatform.init"),
    patch("scripts.benchmark_bakeoff_deployment.get_bakeoff_model", side_effect=mock_get_model),
    patch(
        "scripts.benchmark_bakeoff_deployment.deploy_model_to_endpoint",
        return_value=mock_endpoint,
    ),
    patch(
        "scripts.benchmark_bakeoff_deployment.predict_online",
        return_value=[{"classes": ["0", "1"], "scores": [0.85, 0.15]}],
    ),
    patch("scripts.benchmark_bakeoff_deployment.cleanup_endpoint"),
    patch("tabflows.inference.aiplatform.init"),
    patch("tabflows.inference.aiplatform.Endpoint.create", return_value=mock_endpoint),
    patch("tabflows.inference.aiplatform.Endpoint", return_value=mock_endpoint),
    patch("tabflows.inference.aiplatform.Model", return_value=mock_teacher_model),
    patch("tabflows.experiments.aiplatform.init"),
    patch("tabflows.experiments.aiplatform.start_run"),
    patch("tabflows.experiments.aiplatform.log_params"),
):
    code_cell_count = 0
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            code_cell_count += 1
            stdout_io = io.StringIO()
            stderr_io = io.StringIO()
            source_code = (
                "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            )
            try:
                with redirect_stdout(stdout_io), redirect_stderr(stderr_io):
                    exec(source_code, exec_globals)
                out_text = stdout_io.getvalue()
                err_text = stderr_io.getvalue()
                outputs = []
                if out_text:
                    outputs.append(
                        {
                            "name": "stdout",
                            "output_type": "stream",
                            "text": out_text.splitlines(keepends=True),
                        }
                    )
                if err_text:
                    outputs.append(
                        {
                            "name": "stderr",
                            "output_type": "stream",
                            "text": err_text.splitlines(keepends=True),
                        }
                    )
                cell["outputs"] = outputs
                cell["execution_count"] = code_cell_count
            except Exception as e:
                out_text = stdout_io.getvalue()
                err_text = stderr_io.getvalue()
                outputs = []
                if out_text:
                    outputs.append(
                        {
                            "name": "stdout",
                            "output_type": "stream",
                            "text": out_text.splitlines(keepends=True),
                        }
                    )
                outputs.append(
                    {
                        "name": "stderr",
                        "output_type": "stream",
                        "text": [f"{type(e).__name__}: {e}\n"],
                    }
                )
                cell["outputs"] = outputs
                cell["execution_count"] = code_cell_count
                print(f"Error in cell execution: {e}")

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook 09 execution script updated successfully.")
