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
