import subprocess
import sys
import time
from unittest import mock

import pytest
import requests
from click.testing import CliRunner

from mlflow.deployments import cli

from tests.helper_functions import get_safe_port

pytest.importorskip("mlflow.gateway")

_TEST_CONFIG = """
routes:
  - name: chat
    route_type: llm/v1/chat
    model:
      provider: openai
      name: gpt-4o-mini
      config:
        openai_api_key: sk-openai
"""


def test_start_help():
    runner = CliRunner()
    res = runner.invoke(
        cli.start_server,
        ["--help"],
        catch_exceptions=False,
    )
    assert res.exit_code == 0


def test_start_invalid_config(tmp_path):
    runner = CliRunner()
    config = tmp_path.joinpath("config.yml")
    res = runner.invoke(
        cli.start_server,
        ["--config-path", config],
        catch_exceptions=False,
    )
    assert res.exit_code == 2
    assert "does not exist" in res.output

    config.write_text("\t")
    res = runner.invoke(
        cli.start_server,
        ["--config-path", config],
        catch_exceptions=False,
    )
    assert res.exit_code == 2
    assert "not a valid yaml file" in res.output

    config.write_text(
        """
routes:
  - model:
    name: invalid
"""
    )
    res = runner.invoke(
        cli.start_server,
        ["--config-path", config],
        catch_exceptions=False,
    )
    assert res.exit_code == 2
    assert "The gateway configuration is invalid" in res.output


def test_start_server(tmp_path):
    config = tmp_path.joinpath("config.yml")
    config.write_text(_TEST_CONFIG)
    port = get_safe_port()
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlflow",
            "deployments",
            "start-server",
            "--config-path",
            config,
            "--port",
            str(port),
        ]
    ) as prc:
        try:
            for _ in range(5):
                try:
                    if requests.get(f"http://localhost:{port}/health").ok:
                        break
                except requests.exceptions.ConnectionError:
                    time.sleep(1)
            else:
                raise Exception("Server did not start in time")
        finally:
            prc.terminate()


def test_start_server_fail_on_windows(tmp_path):
    config = tmp_path.joinpath("config.yml")
    config.write_text(_TEST_CONFIG)

    runner = CliRunner()
    with mock.patch("mlflow.deployments.cli.is_windows", return_value=True):
        result = runner.invoke(cli.start_server, ["--config-path", config], catch_exceptions=True)
        assert result.exit_code == 1
        assert "MLflow AI Gateway does not support Windows" in result.output
