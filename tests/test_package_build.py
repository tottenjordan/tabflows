"""Unit test for verifying package build with uv build."""

import subprocess
from pathlib import Path


def test_uv_build_package():
    """Test that executing `uv build` produces wheel and tar.gz artifacts in dist/."""
    project_root = Path(__file__).parent.parent.resolve()

    result = subprocess.run(
        ["uv", "build"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"uv build failed with stderr:\n{result.stderr}"

    dist_dir = project_root / "dist"
    assert dist_dir.is_dir(), "dist/ directory was not created"

    wheel_files = list(dist_dir.glob("*.whl"))
    sdist_files = list(dist_dir.glob("*.tar.gz"))

    assert len(wheel_files) > 0, "No .whl package file found in dist/"
    assert len(sdist_files) > 0, "No .tar.gz package file found in dist/"

    assert any(f.name.startswith("tabflows-") for f in wheel_files)
    assert any(f.name.startswith("tabflows-") for f in sdist_files)
