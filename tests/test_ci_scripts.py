"""Unit test for verifying compile_pipelines_ci.py CI script execution."""

import os
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()


def test_compile_pipelines_ci_function():
    """Test calling compile_pipelines function directly."""
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from scripts.compile_pipelines_ci import compile_pipelines

    compile_pipelines()


def test_compile_pipelines_ci_script_execution():
    """Test executing scripts/compile_pipelines_ci.py as a process."""
    script_path = project_root / "scripts" / "compile_pipelines_ci.py"

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"compile_pipelines_ci.py failed with stderr:\n{result.stderr}"
    expected_msg = (
        "SUCCESS: All AutoML Tabular pipeline templates compiled and verified successfully."
    )
    assert expected_msg in result.stdout


def test_compile_pipelines_ci_with_unbuffered_env():
    """Test executing scripts/compile_pipelines_ci.py with PYTHONUNBUFFERED=1."""
    script_path = project_root / "scripts" / "compile_pipelines_ci.py"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"compile_pipelines_ci.py failed with stderr:\n{result.stderr}"
    expected_msg = (
        "SUCCESS: All AutoML Tabular pipeline templates compiled and verified successfully."
    )
    assert expected_msg in result.stdout


def test_github_ci_yaml_structure():
    """Validate structure and key configurations of .github/workflows/ci.yml."""
    import yaml

    ci_yaml_path = project_root / ".github" / "workflows" / "ci.yml"
    assert ci_yaml_path.exists(), f"Workflow file not found at {ci_yaml_path}"

    with open(ci_yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert isinstance(data, dict), "ci.yml must parse into a dictionary"

    # Validate "on" triggers
    assert "on" in data, "ci.yml must contain 'on' triggers"
    on_block = data["on"]
    assert "push" in on_block and "branches" in on_block["push"]
    assert "pull_request" in on_block and "branches" in on_block["pull_request"]

    expected_branches = ["main", "feat/*", "feat/**"]
    assert on_block["push"]["branches"] == expected_branches
    assert on_block["pull_request"]["branches"] == expected_branches

    # Validate top-level permissions
    assert "permissions" in data
    assert data["permissions"].get("contents") == "read"

    # Validate top-level env defaults
    assert "env" in data
    assert data["env"].get("GCP_PROJECT") == "ci-test-project"
    assert data["env"].get("GCP_LOCATION") == "us-central1"
    assert data["env"].get("GCP_BUCKET_URI") == "gs://ci-test-bucket"
    assert data["env"].get("PYTHONUNBUFFERED") == "1"

    # Validate jobs
    assert "jobs" in data and "ci" in data["jobs"]
    ci_job = data["jobs"]["ci"]

    # Validate steps (specifically setup-uv and uv sync --frozen --dev)
    assert "steps" in ci_job
    steps = ci_job["steps"]
    setup_uv_step = next((s for s in steps if s.get("name") == "Setup uv and Python"), None)
    assert setup_uv_step is not None, "Setup uv step not found"
    assert setup_uv_step.get("with", {}).get("enable-cache") is True

    sync_step = next((s for s in steps if s.get("name") == "Install dependencies"), None)
    assert sync_step is not None, "Install dependencies step not found"
    assert sync_step.get("run") == "uv sync --frozen --dev"
