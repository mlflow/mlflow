from unittest import mock

import pytest

from mlflow.server import fastapi_app


@pytest.fixture(autouse=True)
def reap_sessions():
    # Keep the real sweep away from the machine's temp directory.
    with mock.patch("mlflow.server.assistant.session.reap_stale_sessions") as reap:
        yield reap


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


async def _startup_warnings(monkeypatch, caplog, *, remote, sandbox, docker):
    """The WARNING messages the lifespan logs for this Assistant setup."""
    for name, value in [
        ("MLFLOW_ENABLE_REMOTE_ASSISTANT", remote),
        ("MLFLOW_ENABLE_ASSISTANT_SANDBOX", sandbox),
    ]:
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    docker_path = "/usr/bin/docker" if docker else None
    with (
        # The lifespan and the sandbox check both look for the docker executable.
        mock.patch("mlflow.server.fastapi_app.shutil.which", return_value=docker_path),
        mock.patch("mlflow.assistant.providers.base.shutil.which", return_value=docker_path),
        mock.patch("mlflow.server.sandbox.reap_orphaned_sandbox_containers"),
        mock.patch("mlflow.server.assistant.session.reap_stale_sandbox_homes"),
        caplog.at_level("WARNING"),
    ):
        async with fastapi_app._lifespan(mock.MagicMock()):
            pass
    return [r.message for r in caplog.records if r.levelname == "WARNING"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("remote", "sandbox"),
    [
        # Remote mode with Docker present turns the sandbox on.
        ("true", None),
        ("false", "true"),
    ],
)
async def test_lifespan_warns_that_the_sandbox_does_not_limit_network_access(
    monkeypatch, caplog, remote, sandbox
):
    warnings = await _startup_warnings(
        monkeypatch, caplog, remote=remote, sandbox=sandbox, docker=True
    )

    [message] = warnings
    assert "the sandbox does not limit network access" in message
    assert "host.docker.internal" in message
    assert "other sessions' sandboxes" in message
    assert "cloud metadata endpoints" in message
    assert "requests to the server host bypass it" in message
    assert "host-level firewall rules" in message


@pytest.mark.asyncio
async def test_lifespan_warns_that_a_forced_sandbox_has_no_docker(monkeypatch, caplog):
    warnings = await _startup_warnings(
        monkeypatch, caplog, remote="true", sandbox="true", docker=False
    )

    [message] = warnings
    assert "no `docker` executable was found, so its turns will fail" in message


@pytest.mark.asyncio
@pytest.mark.parametrize(("remote", "sandbox"), [(None, None), ("false", "false")])
async def test_lifespan_does_not_warn_without_the_sandbox_or_remote_mode(
    monkeypatch, caplog, remote, sandbox
):
    assert (
        await _startup_warnings(monkeypatch, caplog, remote=remote, sandbox=sandbox, docker=True)
        == []
    )


@pytest.mark.asyncio
async def test_lifespan_reaps_expired_sessions_even_if_home_reap_fails(reap_sessions):
    with (
        mock.patch("mlflow.server.fastapi_app.shutil.which", return_value=None),
        mock.patch(
            "mlflow.server.assistant.session.reap_stale_sandbox_homes",
            side_effect=Exception("boom"),
        ),
    ):
        async with fastapi_app._lifespan(mock.MagicMock()):
            pass

    reap_sessions.assert_called_once()
