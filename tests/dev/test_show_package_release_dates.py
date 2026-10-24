import asyncio
import json
from pathlib import Path
from unittest import mock

import pytest
import yaml

from dev import show_package_release_dates


@pytest.mark.parametrize("output_json", [False, True])
def test_main(output_json, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    distributions = [("scikit-learn", "1.6.1"), ("numpy", "2.2.0")]
    path = tmp_path / "package-versions.json"
    with (
        mock.patch.object(
            show_package_release_dates, "get_distributions", return_value=distributions
        ) as get_distributions,
        mock.patch.object(
            show_package_release_dates,
            "get_packages",
            return_value=[RuntimeError("PyPI unavailable")] * 2,
        ) as get_packages,
    ):
        asyncio.run(
            show_package_release_dates.main(path)
            if output_json
            else show_package_release_dates.main()
        )

    get_distributions.assert_called_once_with()
    get_packages.assert_awaited_once_with(["scikit-learn", "numpy"], return_exceptions=True)
    assert capsys.readouterr().out == (
        "Package      Version Age past 7d cooldown\n"
        "---------------------------------------\n"
        "scikit-learn 1.6.1                     \n"
        "numpy        2.2.0                     \n"
    )
    if output_json:
        assert path.read_text() == (
            '{\n  "packages": {\n    "numpy": "2.2.0",\n    "scikit-learn": "1.6.1"\n  }\n}\n'
        )
    else:
        assert not path.exists()


def test_json_preserved_after_pypi_failure(tmp_path):
    path = tmp_path / "package-versions.json"
    distributions = [("numpy", "2.2.0")]
    with (
        mock.patch.object(
            show_package_release_dates, "get_distributions", return_value=distributions
        ) as get_distributions,
        mock.patch.object(
            show_package_release_dates, "get_packages", side_effect=RuntimeError("PyPI unavailable")
        ) as get_packages,
        pytest.raises(RuntimeError, match="PyPI unavailable"),
    ):
        asyncio.run(show_package_release_dates.main(path))

    get_distributions.assert_called_once_with()
    get_packages.assert_awaited_once_with(["numpy"], return_exceptions=True)
    assert json.loads(path.read_text()) == {"packages": dict(distributions)}


def test_scheduled_snapshots_use_shared_matrix_steps():
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((root / ".github/workflows/cross-version-tests.yml").read_text())
    steps = workflow["jobs"]["test1"]["steps"]
    assert steps is workflow["jobs"]["test2"]["steps"]
    index = next(
        i for i, step in enumerate(steps) if step.get("uses") == "./.github/actions/show-versions"
    )
    assert steps[index]["with"]["output-json"] == (
        "${{ github.event_name == 'schedule' && 'package-versions.json' || '' }}"
    )
    upload = steps[index + 1]
    assert upload["if"] == (
        "github.event_name == 'schedule' && !cancelled()"
        " && hashFiles('package-versions.json') != ''"
    )
    assert upload["uses"].startswith("actions/upload-artifact@")
    assert upload["with"] == {
        "name": "package-versions-${{ matrix.name }}-${{ matrix.category }}-${{ matrix.version }}",
        "path": "package-versions.json",
        "retention-days": 7,
        "if-no-files-found": "error",
        "overwrite": True,
    }
