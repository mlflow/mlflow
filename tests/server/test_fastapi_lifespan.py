from unittest import mock

import pytest

from mlflow.server import fastapi_app


@pytest.mark.asyncio
async def test_lifespan_reaps_containers_when_docker_present():
    with (
        mock.patch("mlflow.server.fastapi_app.shutil.which", return_value="/usr/bin/docker"),
        mock.patch("mlflow.server.fastapi_app.assistant_sandbox_enabled", return_value=True),
        mock.patch("mlflow.server.sandbox.reap_orphaned_sandbox_containers") as reap_containers,
        mock.patch("mlflow.server.assistant.session.reap_stale_sandbox_homes") as reap_homes,
    ):
        async with fastapi_app._lifespan(mock.MagicMock()):
            pass

    reap_containers.assert_called_once()
    reap_homes.assert_called_once()


@pytest.mark.asyncio
async def test_lifespan_skips_container_reap_without_docker_but_still_reaps_homes():
    # Container reaping needs a docker executable; home reaping is unconditional (cheap no-op when
    # nothing was ever sandboxed).
    with (
        mock.patch("mlflow.server.fastapi_app.shutil.which", return_value=None),
        mock.patch("mlflow.server.fastapi_app.assistant_sandbox_enabled", return_value=False),
        mock.patch("mlflow.server.sandbox.reap_orphaned_sandbox_containers") as reap_containers,
        mock.patch("mlflow.server.assistant.session.reap_stale_sandbox_homes") as reap_homes,
    ):
        async with fastapi_app._lifespan(mock.MagicMock()):
            pass

    reap_containers.assert_not_called()
    reap_homes.assert_called_once()


@pytest.mark.asyncio
async def test_lifespan_reaps_containers_even_when_sandbox_disabled_in_this_process():
    # Crash-recovery: a server restarted with the sandbox off must still reap a container the
    # previous generation left running, as long as docker is present.
    with (
        mock.patch("mlflow.server.fastapi_app.shutil.which", return_value="/usr/bin/docker"),
        mock.patch("mlflow.server.fastapi_app.assistant_sandbox_enabled", return_value=False),
        mock.patch("mlflow.server.sandbox.reap_orphaned_sandbox_containers") as reap_containers,
        mock.patch("mlflow.server.assistant.session.reap_stale_sandbox_homes"),
    ):
        async with fastapi_app._lifespan(mock.MagicMock()):
            pass

    reap_containers.assert_called_once()


@pytest.mark.asyncio
async def test_lifespan_container_reap_error_does_not_skip_home_reap():
    # Separate error boundaries: a container-reap failure must not skip home reaping.
    with (
        mock.patch("mlflow.server.fastapi_app.shutil.which", return_value="/usr/bin/docker"),
        mock.patch("mlflow.server.fastapi_app.assistant_sandbox_enabled", return_value=True),
        mock.patch(
            "mlflow.server.sandbox.reap_orphaned_sandbox_containers",
            side_effect=Exception("boom"),
        ),
        mock.patch("mlflow.server.assistant.session.reap_stale_sandbox_homes") as reap_homes,
    ):
        async with fastapi_app._lifespan(mock.MagicMock()):
            pass

    reap_homes.assert_called_once()


@pytest.mark.asyncio
async def test_lifespan_warns_when_remote_but_not_sandboxed(monkeypatch, caplog):
    monkeypatch.setenv("MLFLOW_ENABLE_REMOTE_ASSISTANT", "true")
    with (
        mock.patch("mlflow.server.fastapi_app.shutil.which", return_value=None),
        mock.patch("mlflow.server.fastapi_app.assistant_sandbox_enabled", return_value=False),
        mock.patch("mlflow.server.assistant.session.reap_stale_sandbox_homes"),
    ):
        with caplog.at_level("WARNING"):
            async with fastapi_app._lifespan(mock.MagicMock()):
                pass

    assert any(
        "remote mode but the Docker sandbox is not active" in r.message for r in caplog.records
    )
