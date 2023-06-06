from click.testing import CliRunner
from mlflow.gateway.cli import start_gateway


def test_start_gateway(tmp_path):
    config = tmp_path.joinpath("config.yaml")
    config.write_text("")
    runner = CliRunner()
    runner.invoke(
        start_gateway,
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
