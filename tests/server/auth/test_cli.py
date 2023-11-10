from click.testing import CliRunner

from mlflow.server.auth import cli


def test_cli():
    runner = CliRunner()
    res = runner.invoke(cli.commands, ["--help"], catch_exceptions=False)
    assert res.exit_code == 0, res.output
