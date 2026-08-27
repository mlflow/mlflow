from unittest import mock

import pytest

from mlflow.server import fastapi_app


@pytest.mark.asyncio
async def test_lifespan_reaps_sandbox_artifacts_when_sandbox_active():
    with (
        mock.patch("mlflow.server.fastapi_app.assistant_sandbox_enabled", return_value=True),
        mock.patch("mlflow.server.sandbox.reap_orphaned_sandbox_containers") as reap_containers,
        mock.patch("mlflow.server.assistant.session.reap_stale_sandbox_homes") as reap_homes,
    ):
        async with fastapi_app._lifespan(mock.MagicMock()):
            pass

    reap_containers.assert_called_once()
    reap_homes.assert_called_once()


@pytest.mark.asyncio
async def test_lifespan_is_noop_when_sandbox_inactive():
    with (
        mock.patch("mlflow.server.fastapi_app.assistant_sandbox_enabled", return_value=False),
        mock.patch("mlflow.server.sandbox.reap_orphaned_sandbox_containers") as reap_containers,
        mock.patch("mlflow.server.assistant.session.reap_stale_sandbox_homes") as reap_homes,
    ):
        async with fastapi_app._lifespan(mock.MagicMock()):
            pass

    reap_containers.assert_not_called()
    reap_homes.assert_not_called()


@pytest.mark.asyncio
async def test_lifespan_swallows_reap_errors():
    with (
        mock.patch("mlflow.server.fastapi_app.assistant_sandbox_enabled", return_value=True),
        mock.patch(
            "mlflow.server.sandbox.reap_orphaned_sandbox_containers",
            side_effect=Exception("boom"),
        ),
        mock.patch("mlflow.server.assistant.session.reap_stale_sandbox_homes"),
    ):
        # A cleanup failure must not stop the server from starting.
        async with fastapi_app._lifespan(mock.MagicMock()):
            pass
