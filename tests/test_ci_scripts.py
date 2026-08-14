"""Unit test for verifying compile_pipelines_ci.py CI script execution."""

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

    # Validate jobs and env defaults
    assert "jobs" in data and "ci" in data["jobs"]
    ci_job = data["jobs"]["ci"]

    assert "env" in ci_job
    assert ci_job["env"].get("GCP_PROJECT") == "ci-test-project"
    assert ci_job["env"].get("GCP_LOCATION") == "us-central1"
    assert ci_job["env"].get("GCP_BUCKET_URI") == "gs://ci-test-bucket"

    # Validate steps (specifically uv sync --dev)
    assert "steps" in ci_job
    steps = ci_job["steps"]
    sync_step = next((s for s in steps if s.get("name") == "Install dependencies"), None)
    assert sync_step is not None, "Install dependencies step not found"
    assert sync_step.get("run") == "uv sync --dev"
