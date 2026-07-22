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


@pytest.mark.parametrize("version", ["2.0.0", "2.4.0"])
def test_unsupported_v2_version_does_not_enable_autologging(monkeypatch, caplog, version):
    monkeypatch.setattr(
        mlflow.pydantic_ai,
        "_get_pydantic_ai_version",
        lambda: Version(version),
    )

    with (
        mock.patch.object(autolog_v2, "setup_autologging") as setup_autologging,
        mock.patch("mlflow.pydantic_ai.setup_autologging") as setup_legacy_autologging,
    ):
        _call_autolog()

    setup_autologging.assert_not_called()
    setup_legacy_autologging.assert_not_called()
    assert (
        f"requires pydantic-ai >= 2.5.0 for Pydantic AI 2.x, but version {version} is installed"
        in caplog.text
    )


def test_disabling_autologging_does_not_warn_for_unsupported_v2(monkeypatch, caplog):
    monkeypatch.setattr(
        mlflow.pydantic_ai,
        "_get_pydantic_ai_version",
        lambda: Version("2.4.0"),
    )

    _call_autolog(disable=True)

    assert "requires pydantic-ai" not in caplog.text
