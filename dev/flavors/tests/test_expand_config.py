import asyncio
from pathlib import Path
from unittest import mock

import pytest
from flavors._loader import load
from flavors._matrix import MatrixItem, expand_config
from pypi import Package


@pytest.fixture(autouse=True)
def change_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


def _package(releases: dict[str, str]) -> Package:
    return Package.from_json({
        "releases": {
            version: [
                {
                    "filename": f"{version}.whl",
                    "upload_time_iso_8601": "2023-10-04T16:38:57Z",
                    "requires_python": requires_python,
                }
            ]
            for version, requires_python in releases.items()
        },
    })


def _expand(src: str, packages: dict[str, Package]) -> set[MatrixItem]:
    async def fake_get_packages(names: list[str]) -> list[Package]:
        return [packages[n] for n in names]

    versions_yaml = Path("ml-package-versions.yml")
    versions_yaml.write_text(src)
    with mock.patch("flavors._matrix.get_packages", new=fake_get_packages):
        return asyncio.run(expand_config(load(versions_yaml)))


def test_tracing_sdk_resolves_python_for_its_own_version() -> None:
    src = """
foo:
  package_info:
    pip_release: foo
  autologging:
    minimum: "1.0.0"
    maximum: "2.0.0"
    run: pytest tests/foo
    test_tracing_sdk: true
"""
    # The minimum version supports Python 3.10 but the latest version requires 3.11. The
    # tracing SDK job tests the latest version, so it must not inherit the Python version
    # resolved for whichever version the autologging loop happened to visit last.
    packages = {"foo": _package({"1.0.0": ">=3.10", "2.0.0": ">=3.11"})}
    matrix = {item.job_name: item for item in _expand(src, packages)}

    assert matrix["foo / autologging / 1.0.0"].python == "3.10"
    assert matrix["foo / autologging / 2.0.0"].python == "3.11"
    assert matrix["foo / tracing-sdk / 2.0.0"].python == "3.11"
