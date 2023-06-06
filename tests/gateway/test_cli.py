from click.testing import CliRunner
from mlflow.gateway.cli import start, update


def test_start(tmp_path):
    config = tmp_path.joinpath("config.yaml")
    config.write_text("")
    runner = CliRunner()
    res = runner.invoke(
        start,
        [
            "--config-path",
            config,
            "--host",
            "localhost",
            "--port",
            5000,
        ],
        catch_exceptions=False,
    )
    assert res.exit_code == 0
    assert "Starting gateway" in res.output


def test_update(tmp_path):
    config = tmp_path.joinpath("config.yaml")
    config.write_text("")
    runner = CliRunner()
    res = runner.invoke(
        update,
        [],
        catch_exceptions=False,
    )
    assert res.exit_code == 0
    assert "Updating gateway" in res.output
