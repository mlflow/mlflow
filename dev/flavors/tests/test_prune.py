import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from flavors._prune import prune_unused_requirements
from pypi import Package


def make_package(versions: list[str]) -> Package:
    fixed = datetime(2023, 10, 4, tzinfo=timezone.utc).isoformat()
    return Package.from_json({
        "releases": {v: [{"filename": f"{v}.whl", "upload_time_iso_8601": fixed}] for v in versions}
    })


def run_prune(src: str, mock_packages: dict[str, Package], tmp_path: Path) -> str:
    yml = tmp_path / "ml-package-versions.yml"
    yml.write_text(src)

    async def fake_get_packages(names):
        return [mock_packages[n] for n in names]

    with mock.patch("flavors._prune.get_packages", new=fake_get_packages):
        asyncio.run(prune_unused_requirements(yml))

    return yml.read_text()


def test_drops_specifier_outside_matrix_range(tmp_path):
    src = """\
sklearn:
  package_info:
    pip_release: "scikit-learn"
  autologging:
    minimum: "1.5.0"
    maximum: "1.8.0"
    requirements:
      "< 1.0.0": ["numpy<2.0"]
      ">= 1.4.0": ["numpy>=2.0"]
    run: |
      pytest tests/sklearn
"""
    expected = """\
sklearn:
  package_info:
    pip_release: "scikit-learn"
  autologging:
    minimum: "1.5.0"
    maximum: "1.8.0"
    requirements:
      ">= 1.4.0": ["numpy>=2.0"]
    run: |
      pytest tests/sklearn
"""
    mock_packages = {"scikit-learn": make_package(["1.5.0", "1.6.0", "1.7.0", "1.8.0"])}
    assert run_prune(src, mock_packages, tmp_path) == expected


def test_drops_requirements_key_when_emptied(tmp_path):
    src = """\
sklearn:
  package_info:
    pip_release: "scikit-learn"
  autologging:
    minimum: "1.5.0"
    maximum: "1.8.0"
    requirements:
      "< 1.0.0": ["numpy<2.0"]
    run: |
      pytest tests/sklearn
"""
    mock_packages = {"scikit-learn": make_package(["1.5.0", "1.6.0", "1.7.0", "1.8.0"])}
    out = run_prune(src, mock_packages, tmp_path)
    assert "requirements:" not in out


def test_preserves_dev_specifier_when_install_dev_set(tmp_path):
    src = """\
sklearn:
  package_info:
    pip_release: "scikit-learn"
    install_dev: |
      uv pip install --system git+https://github.com/scikit-learn/scikit-learn.git
  autologging:
    minimum: "1.5.0"
    maximum: "1.8.0"
    requirements:
      "== dev": ["nightly-helper"]
    run: |
      pytest tests/sklearn
"""
    mock_packages = {"scikit-learn": make_package(["1.5.0", "1.8.0"])}
    out = run_prune(src, mock_packages, tmp_path)
    assert '"== dev"' in out


def test_does_not_rewrite_when_nothing_to_prune(tmp_path):
    src = """\
sklearn:
  package_info:
    pip_release: "scikit-learn"
  autologging:
    minimum: "1.5.0"
    maximum: "1.8.0"
    requirements:
      ">= 1.4.0": ["numpy>=2.0"]
    run: |
      pytest tests/sklearn
"""
    mock_packages = {"scikit-learn": make_package(["1.5.0", "1.6.0", "1.7.0", "1.8.0"])}
    yml = tmp_path / "ml-package-versions.yml"
    yml.write_text(src)
    mtime = yml.stat().st_mtime

    async def fake_get_packages(names):
        return [mock_packages[n] for n in names]

    with mock.patch("flavors._prune.get_packages", new=fake_get_packages):
        asyncio.run(prune_unused_requirements(yml))

    assert yml.stat().st_mtime == mtime
    assert yml.read_text() == src
