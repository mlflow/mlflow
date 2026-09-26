from unittest import mock

import pytest
from packaging.version import Version

import mlflow
from mlflow.pydantic_ai import autolog as pydantic_ai_autolog
from mlflow.pydantic_ai import autolog_v2


def _call_autolog(**kwargs):
    # Exercise version dispatch directly, independent of global autologging configuration state.
    pydantic_ai_autolog.__wrapped__(**kwargs)


@pytest.mark.parametrize("version", ["2.5.0", "2.15.0"])
def test_supported_v2_version_uses_v2_autologging(monkeypatch, version):
    monkeypatch.setattr(
        mlflow.pydantic_ai,
        "_get_pydantic_ai_version",
        lambda: Version(version),
    )

    with mock.patch.object(autolog_v2, "setup_autologging") as setup_autologging:
        _call_autolog()

    setup_autologging.assert_called_once_with()


def test_full_distribution_version_takes_precedence(monkeypatch):
    version_lookup = mock.Mock(return_value=Version("2.15.0"))
    monkeypatch.setattr(mlflow.pydantic_ai, "get_installed_version", version_lookup)

    assert mlflow.pydantic_ai._get_pydantic_ai_version() == Version("2.15.0")
    version_lookup.assert_called_once_with("pydantic-ai")


def test_slim_distribution_version_uses_v2_autologging(monkeypatch):
    versions = {"pydantic-ai-slim": Version("2.15.0")}
    version_lookup = mock.Mock(side_effect=versions.get)
    monkeypatch.setattr(mlflow.pydantic_ai, "get_installed_version", version_lookup)

    with mock.patch.object(autolog_v2, "setup_autologging") as setup_autologging:
        _call_autolog()

    assert version_lookup.call_args_list == [
        mock.call("pydantic-ai"),
        mock.call("pydantic-ai-slim"),
    ]
    setup_autologging.assert_called_once_with()


@pytest.mark.parametrize("version", ["2.0.0", "2.4.0"])
def test_unsupported_v2_version_does_not_enable_autologging(monkeypatch, version):
    monkeypatch.setattr(
        mlflow.pydantic_ai,
        "_get_pydantic_ai_version",
        lambda: Version(version),
    )

    with (
        mock.patch.object(autolog_v2, "setup_autologging") as setup_autologging,
        mock.patch.object(
            mlflow.pydantic_ai,
            "_get_tool_manager_module_path",
        ) as legacy_setup_probe,
        mock.patch.object(mlflow.pydantic_ai._logger, "warning") as warning,
    ):
        _call_autolog()

    setup_autologging.assert_not_called()
    legacy_setup_probe.assert_not_called()
    warning.assert_called_once_with(
        "MLflow Pydantic AI autologging requires pydantic-ai >= %s for Pydantic AI "
        "2.x, but version %s is installed. Autologging has not been enabled. Please "
        "upgrade pydantic-ai.",
        Version("2.5.0"),
        Version(version),
    )


def test_disabling_autologging_does_not_warn_for_unsupported_v2(monkeypatch):
    monkeypatch.setattr(
        mlflow.pydantic_ai,
        "_get_pydantic_ai_version",
        lambda: Version("2.4.0"),
    )

    with mock.patch.object(mlflow.pydantic_ai._logger, "warning") as warning:
        _call_autolog(disable=True)

    warning.assert_not_called()
