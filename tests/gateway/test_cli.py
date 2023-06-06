from click.testing import CliRunner
from mlflow.gateway.cli import start, update


def test_start(tmp_path):
    config_path = tmp_path.joinpath("config.yaml")
    config_path.write_text("")
    runner = CliRunner()
    res = runner.invoke(
        start,
        [
            "--config-path",
            config_path,
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
    config_path = tmp_path.joinpath("config.yaml")
    config_path.write_text("")
    runner = CliRunner()
    res = runner.invoke(
        update,
        [
            "--config-path",
            config_path,
        ],
        catch_exceptions=False,
    )
    assert res.exit_code == 0
    assert "Updating gateway" in res.output
