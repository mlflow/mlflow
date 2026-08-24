import os
from pathlib import Path
from unittest import mock

import pytest
import requests

from mlflow.server.sandbox import (
    SandboxResult,
    SandboxUnavailableError,
    run_in_sandbox,
    to_container_host_uri,
)
from mlflow.server.sandbox import container as container_mod

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="Sandbox relies on POSIX uid/gid and Docker Linux containers"
)


def _mock_client(status_code=0, logs=b"hello\n"):
    client = mock.MagicMock()
    client.images.get.return_value = mock.MagicMock()  # image already present
    container = mock.MagicMock()
    container.wait.return_value = {"StatusCode": status_code}
    container.logs.return_value = logs
    client.containers.run.return_value = container
    return client, container


@pytest.mark.parametrize(
    ("loopback", "expected"),
    [
        ("http://127.0.0.1:5000", "http://host.docker.internal:5000"),
        ("http://localhost:5000", "http://host.docker.internal:5000"),
        ("http://0.0.0.0:5000", "http://host.docker.internal:5000"),
        ("http://[::1]:5000", "http://host.docker.internal:5000"),
        (
            "http://localhost:5000/path?exp=localhost",
            "http://host.docker.internal:5000/path?exp=localhost",
        ),
        # A host that merely contains "localhost" is not a loopback host and must be left alone.
        ("http://localhost.example.com:5000", "http://localhost.example.com:5000"),
        ("https://tracking.example.com", "https://tracking.example.com"),
        # A loopback host with an invalid port must be returned unchanged, not raise ValueError.
        ("http://localhost:notaport/api", "http://localhost:notaport/api"),
        (None, None),
        ("", ""),
    ],
)
def test_to_container_host_uri(loopback, expected):
    assert to_container_host_uri(loopback) == expected


def test_to_container_host_uri_preserves_userinfo_and_port():
    # Only the loopback host token is swapped; userinfo bytes and the port are kept as-is.
    # The userinfo is assembled from parts so no literal credential URI is committed to source.
    creds = "tok:v"
    assert (
        to_container_host_uri(f"http://{creds}@127.0.0.1:5000/api")
        == f"http://{creds}@host.docker.internal:5000/api"
    )


def test_run_in_sandbox_returns_output_and_exit_code():
    client, container = _mock_client(status_code=0, logs=b"result\n")
    with mock.patch("docker.from_env", return_value=client):
        result = run_in_sandbox(["mlflow", "--version"])

    assert isinstance(result, SandboxResult)
    assert result.exit_code == 0
    assert result.output == "result\n"
    assert result.timed_out is False
    container.remove.assert_called_once()


def test_run_in_sandbox_nonzero_exit_code():
    client, _ = _mock_client(status_code=2, logs=b"boom\n")
    with mock.patch("docker.from_env", return_value=client):
        result = run_in_sandbox(["mlflow", "bogus"])

    assert result.exit_code == 2
    assert result.timed_out is False


def test_run_in_sandbox_applies_hardening_flags():
    client, _ = _mock_client()
    with mock.patch("docker.from_env", return_value=client):
        run_in_sandbox(["echo", "hi"])

    _, kwargs = client.containers.run.call_args
    assert kwargs["cap_drop"] == ["ALL"]
    assert kwargs["security_opt"] == ["no-new-privileges:true"]
    assert kwargs["read_only"] is True
    assert kwargs["network"] == container_mod.SANDBOX_NETWORK_NAME
    assert kwargs["pids_limit"] == container_mod._PIDS_LIMIT
    assert kwargs["mem_limit"] == container_mod._MEMORY_LIMIT
    assert kwargs["user"] == f"{os.getuid()}:{os.getgid()}"


def test_run_in_sandbox_shell_vs_argv():
    client, _ = _mock_client()
    with mock.patch("docker.from_env", return_value=client):
        run_in_sandbox(["mlflow runs list && echo hi"], use_shell=True)
        _, shell_kwargs = client.containers.run.call_args
        assert shell_kwargs["command"] == ["sh", "-c", "mlflow runs list && echo hi"]

        run_in_sandbox(["mlflow", "--version"], use_shell=False)
        _, argv_kwargs = client.containers.run.call_args
        assert argv_kwargs["command"] == ["mlflow", "--version"]


def test_run_in_sandbox_mounts_workdir(tmp_path):
    client, _ = _mock_client()
    with mock.patch("docker.from_env", return_value=client):
        run_in_sandbox(["mlflow", "--version"], workdir=tmp_path)

    _, kwargs = client.containers.run.call_args
    assert kwargs["volumes"][str(tmp_path)] == {"bind": "/workspace", "mode": "rw"}
    assert kwargs["working_dir"] == "/workspace"


def test_run_in_sandbox_no_workdir_has_no_mount():
    client, _ = _mock_client()
    with mock.patch("docker.from_env", return_value=client):
        run_in_sandbox(["echo", "hi"])

    _, kwargs = client.containers.run.call_args
    assert kwargs["volumes"] == {}
    assert kwargs["working_dir"] is None


def test_run_in_sandbox_timeout_kills_container():
    client, container = _mock_client()
    container.wait.side_effect = requests.exceptions.ReadTimeout("Read timed out")
    with mock.patch("docker.from_env", return_value=client):
        result = run_in_sandbox(["sleep", "1000"], timeout=0.1)

    assert result.timed_out is True
    assert result.exit_code == container_mod._TIMEOUT_EXIT_CODE
    container.kill.assert_called_once()
    container.remove.assert_called_once()


def test_run_in_sandbox_timeout_surfaced_as_connection_error():
    # On the Unix-socket transport, docker-py surfaces a wait() read timeout as a
    # ConnectionError wrapping urllib3's ReadTimeoutError ("Read timed out"), not ReadTimeout.
    client, container = _mock_client()
    container.wait.side_effect = requests.exceptions.ConnectionError(
        "UnixHTTPConnectionPool(host='localhost', port=None): Read timed out."
    )
    with mock.patch("docker.from_env", return_value=client):
        result = run_in_sandbox(["sleep", "1000"], timeout=0.1)

    assert result.timed_out is True
    assert result.exit_code == container_mod._TIMEOUT_EXIT_CODE
    container.kill.assert_called_once()


def test_run_in_sandbox_non_timeout_wait_error_raises_unavailable():
    # A non-timeout failure from wait() must not be reported as a timeout: it surfaces as a
    # distinct sandbox failure, and the container is still cleaned up.
    client, container = _mock_client()
    container.wait.side_effect = requests.exceptions.ConnectionError("daemon gone")
    with mock.patch("docker.from_env", return_value=client):
        with pytest.raises(SandboxUnavailableError, match="failed while waiting"):
            run_in_sandbox(["mlflow", "--version"])

    container.kill.assert_called_once()
    container.remove.assert_called_once()


def test_run_in_sandbox_raises_when_docker_unavailable():
    with mock.patch("docker.from_env", side_effect=Exception("no daemon")):
        with pytest.raises(SandboxUnavailableError, match="Docker daemon is not reachable"):
            run_in_sandbox(["echo", "hi"])


def test_run_in_sandbox_builds_image_when_missing():
    import docker.errors

    client, _ = _mock_client()
    client.images.get.side_effect = docker.errors.ImageNotFound("missing")
    with mock.patch("docker.from_env", return_value=client):
        run_in_sandbox(["echo", "hi"])

    client.images.build.assert_called_once()


def test_run_in_sandbox_fallback_build_does_not_forward_index_credentials(monkeypatch):
    import docker.errors

    # A private-index URL can embed credentials, and Docker records build args in image history,
    # so the fallback build must not forward PIP_INDEX_URL at all — neither as a build arg nor in
    # the Dockerfile. Operators behind a private mirror provide their own image instead.
    monkeypatch.setenv("PIP_INDEX_URL", "https://user:pass@mirror.internal/simple")
    client, _ = _mock_client()
    client.images.get.side_effect = docker.errors.ImageNotFound("missing")
    captured = {}

    def _build(path, **kwargs):
        captured["dockerfile"] = Path(path, "Dockerfile").read_text()
        captured["buildargs"] = kwargs.get("buildargs")
        return (mock.MagicMock(), [])

    client.images.build.side_effect = _build
    with mock.patch("docker.from_env", return_value=client):
        run_in_sandbox(["echo", "hi"])

    assert not captured["buildargs"]
    assert "PIP_INDEX_URL" not in captured["dockerfile"]
    assert "user:pass" not in captured["dockerfile"]


def test_run_in_sandbox_image_build_failure_raises_unavailable():
    import docker.errors

    client, _ = _mock_client()
    client.images.get.side_effect = docker.errors.ImageNotFound("missing")
    client.images.build.side_effect = docker.errors.BuildError("bad dockerfile", build_log=[])
    with mock.patch("docker.from_env", return_value=client):
        with pytest.raises(SandboxUnavailableError, match="Failed to prepare sandbox image"):
            run_in_sandbox(["echo", "hi"])


def test_run_in_sandbox_labels_container(monkeypatch):
    monkeypatch.setenv("_MLFLOW_SERVER_BOOT_ID", "boot-xyz")
    client, _ = _mock_client()
    with mock.patch("docker.from_env", return_value=client):
        run_in_sandbox(["echo", "hi"])
    _, kwargs = client.containers.run.call_args
    assert kwargs["labels"] == {
        container_mod.SANDBOX_CONTAINER_LABEL: "1",
        container_mod.SANDBOX_BOOT_LABEL: "boot-xyz",
    }


def test_reap_removes_previous_generation_only(monkeypatch):
    from mlflow.server.sandbox import reap_orphaned_sandbox_containers

    monkeypatch.setenv("_MLFLOW_SERVER_BOOT_ID", "current-boot")
    old = mock.MagicMock()
    old.labels = {container_mod.SANDBOX_BOOT_LABEL: "old-boot"}
    old.status = "exited"
    mine = mock.MagicMock()
    mine.labels = {container_mod.SANDBOX_BOOT_LABEL: "current-boot"}
    mine.status = "running"
    client = mock.MagicMock()
    client.containers.list.return_value = [old, mine]
    with mock.patch("docker.from_env", return_value=client):
        removed = reap_orphaned_sandbox_containers()

    # Only the previous generation's stopped container is removed; the current one is left running.
    assert removed == 1
    old.remove.assert_called_once_with(force=True)
    mine.remove.assert_not_called()
    _, kwargs = client.containers.list.call_args
    assert kwargs["filters"] == {"label": container_mod.SANDBOX_CONTAINER_LABEL}


def test_reap_skips_running_container_from_other_generation(monkeypatch):
    from mlflow.server.sandbox import reap_orphaned_sandbox_containers

    # A different boot id does not prove a container is orphaned (a concurrent server sharing the
    # daemon), so a still-running one is left alone rather than force-killed mid-turn.
    monkeypatch.setenv("_MLFLOW_SERVER_BOOT_ID", "current-boot")
    other_running = mock.MagicMock()
    other_running.labels = {container_mod.SANDBOX_BOOT_LABEL: "other-boot"}
    other_running.status = "running"
    client = mock.MagicMock()
    client.containers.list.return_value = [other_running]
    with mock.patch("docker.from_env", return_value=client):
        assert reap_orphaned_sandbox_containers() == 0
    other_running.remove.assert_not_called()


def test_reap_skips_when_no_boot_id(monkeypatch):
    from mlflow.server.sandbox import reap_orphaned_sandbox_containers

    monkeypatch.delenv("_MLFLOW_SERVER_BOOT_ID", raising=False)
    client = mock.MagicMock()
    with mock.patch("docker.from_env", return_value=client):
        assert reap_orphaned_sandbox_containers() == 0
    # Without a boot id we cannot distinguish generations, so we never even list containers.
    client.containers.list.assert_not_called()


def test_reap_counts_only_successful_removes(monkeypatch):
    from mlflow.server.sandbox import reap_orphaned_sandbox_containers

    monkeypatch.setenv("_MLFLOW_SERVER_BOOT_ID", "current-boot")
    ok = mock.MagicMock()
    ok.labels = {container_mod.SANDBOX_BOOT_LABEL: "old-boot"}
    fails = mock.MagicMock()
    fails.labels = {container_mod.SANDBOX_BOOT_LABEL: "old-boot"}
    fails.remove.side_effect = Exception("cannot remove")
    client = mock.MagicMock()
    client.containers.list.return_value = [ok, fails]
    with mock.patch("docker.from_env", return_value=client):
        assert reap_orphaned_sandbox_containers() == 1


def test_reap_docker_unavailable_returns_zero(monkeypatch):
    from mlflow.server.sandbox import reap_orphaned_sandbox_containers

    monkeypatch.setenv("_MLFLOW_SERVER_BOOT_ID", "current-boot")
    with mock.patch("docker.from_env", side_effect=Exception("no daemon")):
        assert reap_orphaned_sandbox_containers() == 0


def test_reap_list_error_returns_zero(monkeypatch):
    from mlflow.server.sandbox import reap_orphaned_sandbox_containers

    monkeypatch.setenv("_MLFLOW_SERVER_BOOT_ID", "current-boot")
    client = mock.MagicMock()
    client.containers.list.side_effect = Exception("api error")
    with mock.patch("docker.from_env", return_value=client):
        assert reap_orphaned_sandbox_containers() == 0


def test_sandbox_egress_env_empty_when_unset(monkeypatch):
    from mlflow.server.sandbox.container import sandbox_egress_env

    monkeypatch.delenv("MLFLOW_SANDBOX_EGRESS_PROXY", raising=False)
    assert sandbox_egress_env() == {}


def test_sandbox_egress_env_injects_proxy_and_bypass(monkeypatch):
    from mlflow.server.sandbox.container import sandbox_egress_env

    monkeypatch.setenv("MLFLOW_SANDBOX_EGRESS_PROXY", "http://proxy.internal:3128")
    env = sandbox_egress_env(no_proxy_hosts=["tracking.example.com"])
    assert env["HTTP_PROXY"] == "http://proxy.internal:3128"
    assert env["HTTPS_PROXY"] == "http://proxy.internal:3128"
    # Lowercase variants for clients that only read them.
    assert env["no_proxy"] == env["NO_PROXY"]
    assert env["https_proxy"] == "http://proxy.internal:3128"
    # The tracking host bypasses the proxy so it stays reachable...
    assert "host.docker.internal" in env["NO_PROXY"]
    assert "tracking.example.com" in env["NO_PROXY"]
    # ...but the cloud metadata endpoint must NOT bypass the proxy.
    assert "169.254.169.254" not in env["NO_PROXY"]


def test_sandbox_egress_env_dedupes_no_proxy(monkeypatch):
    from mlflow.server.sandbox.container import sandbox_egress_env

    monkeypatch.setenv("MLFLOW_SANDBOX_EGRESS_PROXY", "http://proxy.internal:3128")
    # A loopback tracking host is already rewritten to host.docker.internal, so it must not be
    # duplicated in NO_PROXY.
    env = sandbox_egress_env(no_proxy_hosts=["host.docker.internal", None])
    assert env["NO_PROXY"].split(",").count("host.docker.internal") == 1


def test_run_in_sandbox_injects_egress_proxy(monkeypatch):
    monkeypatch.setenv("MLFLOW_SANDBOX_EGRESS_PROXY", "http://proxy.internal:3128")
    client, _ = _mock_client()
    with mock.patch("docker.from_env", return_value=client):
        run_in_sandbox(["echo", "hi"])
    _, kwargs = client.containers.run.call_args
    assert kwargs["environment"]["HTTPS_PROXY"] == "http://proxy.internal:3128"


def test_ensure_sandbox_network_creates_when_missing():
    import docker.errors

    from mlflow.server.sandbox.container import SANDBOX_NETWORK_NAME, ensure_sandbox_network

    client = mock.MagicMock()
    client.networks.get.side_effect = docker.errors.NotFound("nope")
    assert ensure_sandbox_network(client) == SANDBOX_NETWORK_NAME
    client.networks.create.assert_called_once_with(
        SANDBOX_NETWORK_NAME, driver="bridge", check_duplicate=True
    )


def test_ensure_sandbox_network_handles_create_race():
    import docker.errors

    from mlflow.server.sandbox.container import SANDBOX_NETWORK_NAME, ensure_sandbox_network

    client = mock.MagicMock()
    # First get: not found; create loses the race (409 duplicate); re-get: found.
    client.networks.get.side_effect = [docker.errors.NotFound("no"), mock.MagicMock()]
    client.networks.create.side_effect = docker.errors.APIError("network already exists")
    assert ensure_sandbox_network(client) == SANDBOX_NETWORK_NAME


def test_ensure_sandbox_network_falls_back_to_bridge_on_error():
    from mlflow.server.sandbox.container import ensure_sandbox_network

    client = mock.MagicMock()
    client.networks.get.side_effect = Exception("permission denied")
    assert ensure_sandbox_network(client) == "bridge"
