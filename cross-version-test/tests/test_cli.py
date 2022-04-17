from pathlib import Path
import typing as t

import click
from click.testing import CliRunner

from cross_version_test import cli


def _run_command(
    cli: click.BaseCommand,
    args: t.Optional[t.Sequence[str]] = None,
    catch_exceptions: bool = False,
) -> int:
    runner = CliRunner()
    res = runner.invoke(cli, args, catch_exceptions=catch_exceptions)
    return res.exit_code


def test_cli_commands(tmp_path: Path) -> None:
    versions_yaml = tmp_path.joinpath("versions.yaml")
    versions_yaml.write_text(
        """
sklearn:
  package_info:
    pip_release: "scikit-learn"
  models:
    minimum: "1.0.2"
    maximum: "1.0.2"
    run: |
      python -c 'import sklearn; assert sklearn.__version__ == "1.0.2"'
"""
    )
    ref_versions_yaml = tmp_path.joinpath("ref_versions.yaml")
    ref_versions_yaml.write_text(
        """
sklearn:
  package_info:
    pip_release: "scikit-learn"
  models:
    minimum: "1.0.2"
    maximum: "1.0.2"
    run: |
      python -c 'import sklearn; assert "1.0.2" == sklearn.__version__'
"""
    )
    args = [
        "--versions-yaml",
        str(versions_yaml),
        "--ref-versions-yaml",
        str(ref_versions_yaml),
    ]
    assert _run_command(cli.diff, args) == 0
    args = [
        "--versions-yaml",
        str(versions_yaml),
        "--changed-files",
        "mlflow/sklearn/__init__.py",
    ]
    assert _run_command(cli.diff, args) == 0
    assert _run_command(cli.list_command) == 0
    args = [
        "--pattern",
        "sklearn_1.0.2_models",
    ]
    assert _run_command(cli.build, args) == 0
    assert _run_command(cli.run, args) == 0
    assert _run_command(cli.down) == 0
