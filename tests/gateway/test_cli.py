import pytest
from click.testing import CliRunner

from mlflow.gateway import cli as gateway_cli
from mlflow.gateway.cli import start


def test_start_help():
    runner = CliRunner()
    res = runner.invoke(
        start,
        ["--help"],
        catch_exceptions=False,
    )
    assert res.exit_code == 0


def test_start_invalid_config(tmp_path):
    runner = CliRunner()
    config = tmp_path.joinpath("config.yml")
    res = runner.invoke(
        start,
        ["--config-path", config],
        catch_exceptions=False,
    )
    assert res.exit_code == 2
    assert "does not exist" in res.output

    config.write_text("\t")
    res = runner.invoke(
        start,
        ["--config-path", config],
        catch_exceptions=False,
    )
    assert res.exit_code == 2
    assert "not a valid yaml file" in res.output

    config.write_text(
        """
endpoints:
  - model:
    name: invalid
"""
    )
    res = runner.invoke(
        start,
        ["--config-path", config],
        catch_exceptions=False,
    )
    assert res.exit_code == 2
    assert "The gateway configuration is invalid" in res.output


def test_start_warns_deprecation(monkeypatch, tmp_path):
    runner = CliRunner()
    config = tmp_path.joinpath("config.yml")
    config.write_text("unused")

    monkeypatch.setattr(gateway_cli, "_validate_config", lambda _: None)
    monkeypatch.setattr(gateway_cli, "_record_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(gateway_cli, "run_app", lambda **_kwargs: None)
    monkeypatch.setattr(gateway_cli, "is_windows", lambda: False)

    with pytest.warns(FutureWarning, match="mlflow.gateway.cli.start.*deprecated"):
        res = runner.invoke(
            start,
            ["--config-path", config],
            catch_exceptions=False,
        )
    assert res.exit_code == 0
