import os
import sys
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


@pytest.fixture(autouse=True)
def _no_source_tree(tmp_path_factory, monkeypatch):
    # Tests run from a dev checkout, and CI sets MLFLOW_HOME to it, so without this the fallback
    # image build would copy this repository into each test's build context. Tests of source
    # installs set their own source tree.
    monkeypatch.delenv("MLFLOW_HOME", raising=False)
    package_dir = tmp_path_factory.mktemp("not-a-checkout") / "mlflow"
    monkeypatch.setattr(container_mod, "_MLFLOW_PACKAGE_DIR", package_dir)


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


def test_run_in_sandbox_connect_timeout_is_not_a_read_timeout():
    # A connection-establishment timeout (daemon unreachable) is a ConnectionError whose message
    # says "timed out" but NOT "read timed out". It must surface as a sandbox failure rather than
    # be misreported as the command timing out, so the operator sees the daemon is down.
    client, container = _mock_client()
    container.wait.side_effect = requests.exceptions.ConnectionError(
        "HTTPConnectionPool(host='localhost', port=2375): Max retries exceeded "
        "(Caused by ConnectTimeoutError(...): Connection to localhost timed out.)"
    )
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


def test_minimal_sandbox_dockerfile_uses_the_server_python_minor_version(tmp_path):
    dockerfile = container_mod._minimal_sandbox_dockerfile(str(tmp_path))
    expected_tag = f"{sys.version_info.major}.{sys.version_info.minor}"
    assert dockerfile.startswith(f"FROM python:{expected_tag}-slim\n")


def _make_source_tree(root, name="mlflow"):
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(f'[project]\nname = "{name}"\n')
    (root / "README.md").write_text("readme")
    (root / "LICENSE.txt").write_text("license")
    package = root / "mlflow"
    (package / "server").mkdir(parents=True)
    (package / "__init__.py").touch()
    (package / "server" / "handlers.py").touch()
    return root


def test_minimal_sandbox_dockerfile_installs_mlflow_from_source_home(tmp_path, monkeypatch):
    source = _make_source_tree(tmp_path / "src")
    monkeypatch.setenv("MLFLOW_HOME", str(source))
    context = tmp_path / "context"
    context.mkdir()

    dockerfile = container_mod._minimal_sandbox_dockerfile(str(context))

    assert dockerfile.endswith("COPY mlflow-source /opt/mlflow\nRUN pip install /opt/mlflow\n")
    assert (context / "mlflow-source" / "mlflow" / "server" / "handlers.py").is_file()


def test_minimal_sandbox_dockerfile_installs_from_the_server_checkout(tmp_path, monkeypatch):
    checkout = _make_source_tree(tmp_path / "checkout")
    monkeypatch.setattr(container_mod, "_mlflow_source_root", lambda: checkout)
    context = tmp_path / "context"
    context.mkdir()

    dockerfile = container_mod._minimal_sandbox_dockerfile(str(context))

    assert "COPY mlflow-source /opt/mlflow" in dockerfile
    assert (context / "mlflow-source" / "pyproject.toml").is_file()


def test_minimal_sandbox_dockerfile_prefers_mlflow_home_over_the_checkout(tmp_path, monkeypatch):
    home = _make_source_tree(tmp_path / "home")
    (home / "mlflow" / "from_home.py").touch()
    monkeypatch.setenv("MLFLOW_HOME", str(home))
    checkout = _make_source_tree(tmp_path / "checkout")
    monkeypatch.setattr(container_mod, "_mlflow_source_root", lambda: checkout)
    context = tmp_path / "context"
    context.mkdir()

    container_mod._minimal_sandbox_dockerfile(str(context))

    assert (context / "mlflow-source" / "mlflow" / "from_home.py").is_file()


def test_source_copy_leaves_out_server_data_and_frontend_files(tmp_path, monkeypatch):
    source = _make_source_tree(tmp_path / "src")
    # Local data a server writes when run from the checkout, outside and inside the package.
    (source / "mlflow.db").touch()
    (source / "basic_auth.db").touch()
    (source / "mlartifacts").mkdir()
    (source / "mlflow" / "stray.db").touch()
    (source / "mlflow" / "stray.db-wal").touch()
    (source / "mlflow" / "store.sqlite").touch()
    (source / "mlflow" / "server" / "js" / "build").mkdir(parents=True)
    (source / "mlflow" / "java").mkdir()
    (source / "mlflow" / "__pycache__").mkdir()
    (source / "mlflow" / "outside").symlink_to(tmp_path / "src" / "README.md")
    monkeypatch.setenv("MLFLOW_HOME", str(source))
    context = tmp_path / "context"
    context.mkdir()

    container_mod._minimal_sandbox_dockerfile(str(context))

    copied = context / "mlflow-source"
    assert sorted(p.name for p in copied.iterdir()) == [
        "LICENSE.txt",
        "README.md",
        "mlflow",
        "pyproject.toml",
    ]
    assert sorted(p.name for p in (copied / "mlflow").iterdir()) == [
        "__init__.py",
        "outside",
        "server",
    ]
    assert sorted(p.name for p in (copied / "mlflow" / "server").iterdir()) == ["handlers.py"]
    # A symlink stays a link, so its target's contents are never copied into the image.
    assert (copied / "mlflow" / "outside").is_symlink()


def test_minimal_sandbox_dockerfile_without_a_source_tree_uses_the_install_step(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(container_mod, "VERSION", "9.9.9")
    with (
        mock.patch(
            "mlflow.models.docker_utils._pip_mlflow_install_step",
            return_value="RUN pip install mlflow==9.9.9",
        ) as step,
        mock.patch.object(container_mod._logger, "warning") as warning,
    ):
        dockerfile = container_mod._minimal_sandbox_dockerfile(str(tmp_path))

    step.assert_called_once_with(str(tmp_path), None)
    assert dockerfile.endswith("RUN pip install mlflow==9.9.9\n")
    # A released version installs its own version from PyPI, so there is nothing to warn about.
    warning.assert_not_called()


def test_minimal_sandbox_dockerfile_warns_when_a_dev_build_installs_the_dev_branch(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(container_mod, "VERSION", "9.9.9.dev0")
    with mock.patch.object(container_mod._logger, "warning") as warning:
        container_mod._minimal_sandbox_dockerfile(str(tmp_path))

    warning.assert_called_once()
    assert "MLFLOW_HOME" in warning.call_args.args[0]


def test_mlflow_project_name_matches_the_repository_pyproject():
    pyproject = Path(container_mod.__file__).resolve().parents[3] / "pyproject.toml"

    assert container_mod._MLFLOW_PROJECT_NAME.search(pyproject.read_text())


def test_mlflow_source_root_finds_the_checkout_of_a_dev_build(tmp_path, monkeypatch):
    checkout = _make_source_tree(tmp_path)
    monkeypatch.setattr(container_mod, "_MLFLOW_PACKAGE_DIR", checkout / "mlflow")
    monkeypatch.setattr(container_mod, "VERSION", "9.9.9.dev0")

    assert container_mod._mlflow_source_root() == checkout


@pytest.mark.parametrize(
    ("version", "project_name"),
    [
        # A released version installs from PyPI even when it runs from a checkout.
        ("9.9.9", "mlflow"),
        # A dev build vendored inside another project must not install that project.
        ("9.9.9.dev0", "other-project"),
    ],
)
def test_mlflow_source_root_is_none_unless_a_dev_build_runs_from_mlflow(
    tmp_path, monkeypatch, version, project_name
):
    checkout = _make_source_tree(tmp_path, name=project_name)
    monkeypatch.setattr(container_mod, "_MLFLOW_PACKAGE_DIR", checkout / "mlflow")
    monkeypatch.setattr(container_mod, "VERSION", version)

    assert container_mod._mlflow_source_root() is None


def test_mlflow_source_root_is_none_for_a_dev_build_installed_into_site_packages(
    tmp_path, monkeypatch
):
    (tmp_path / "mlflow").mkdir()
    monkeypatch.setattr(container_mod, "_MLFLOW_PACKAGE_DIR", tmp_path / "mlflow")
    monkeypatch.setattr(container_mod, "VERSION", "9.9.9.dev0")

    assert container_mod._mlflow_source_root() is None


def test_run_in_sandbox_image_build_failure_includes_the_build_output():
    import docker.errors

    client, _ = _mock_client()
    client.images.get.side_effect = docker.errors.ImageNotFound("missing")
    client.images.build.side_effect = docker.errors.BuildError(
        "The command '/bin/sh -c pip install mlflow' returned a non-zero code: 1",
        build_log=[
            {"stream": "Step 2/2 : RUN pip install mlflow\n"},
            {"stream": "ERROR: No matching distribution found for some-package\n"},
        ],
    )
    with mock.patch("docker.from_env", return_value=client):
        with pytest.raises(SandboxUnavailableError, match="No matching distribution found"):
            run_in_sandbox(["echo", "hi"])


def test_build_log_tail_keeps_the_last_output_lines(monkeypatch):
    monkeypatch.setattr(container_mod, "_BUILD_LOG_TAIL_LINES", 2)
    build_log = [
        {"stream": "line 1\nline 2\n"},
        {"aux": {"ID": "sha256:abc"}},
        {"stream": "\n"},
        {"stream": "line 3\n"},
        # The error chunk repeats BuildError.msg, which the message already includes.
        {"error": "The command returned a non-zero code: 1"},
    ]

    assert container_mod._build_log_tail(iter(build_log)) == "line 2\nline 3"


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


@pytest.mark.parametrize("status", ["running", "created", "restarting", "paused", "removing"])
def test_reap_skips_non_terminal_container_from_other_generation(monkeypatch, status):
    from mlflow.server.sandbox import reap_orphaned_sandbox_containers

    # Only terminal-state (exited/dead) containers are reaped. A container in any live or
    # transitional state — running, or still starting up (created/restarting) — may belong to a
    # concurrent server sharing the daemon, so a different boot id does not prove it is orphaned
    # and force-removing it could kill another server's turn. It is left for a later startup to
    # reap once it reaches a terminal state.
    monkeypatch.setenv("_MLFLOW_SERVER_BOOT_ID", "current-boot")
    other = mock.MagicMock()
    other.labels = {container_mod.SANDBOX_BOOT_LABEL: "other-boot"}
    other.status = status
    client = mock.MagicMock()
    client.containers.list.return_value = [other]
    with mock.patch("docker.from_env", return_value=client):
        assert reap_orphaned_sandbox_containers() == 0
    other.remove.assert_not_called()


def test_reap_removes_dead_container_from_other_generation(monkeypatch):
    from mlflow.server.sandbox import reap_orphaned_sandbox_containers

    # "dead" is a terminal state (a container that could not be fully removed), so it is reaped
    # alongside "exited".
    monkeypatch.setenv("_MLFLOW_SERVER_BOOT_ID", "current-boot")
    dead = mock.MagicMock()
    dead.labels = {container_mod.SANDBOX_BOOT_LABEL: "old-boot"}
    dead.status = "dead"
    client = mock.MagicMock()
    client.containers.list.return_value = [dead]
    with mock.patch("docker.from_env", return_value=client):
        assert reap_orphaned_sandbox_containers() == 1
    dead.remove.assert_called_once_with(force=True)


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
    ok.status = "exited"
    fails = mock.MagicMock()
    fails.labels = {container_mod.SANDBOX_BOOT_LABEL: "old-boot"}
    fails.status = "exited"
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
    env = sandbox_egress_env()
    assert env["HTTP_PROXY"] == "http://proxy.internal:3128"
    assert env["HTTPS_PROXY"] == "http://proxy.internal:3128"
    # Lowercase variants for clients that only read them.
    assert env["no_proxy"] == env["NO_PROXY"]
    assert env["https_proxy"] == "http://proxy.internal:3128"
    # The self-host bypass entries let the container reach a local tracking server...
    assert "host.docker.internal" in env["NO_PROXY"]
    assert "localhost" in env["NO_PROXY"]
    assert "127.0.0.1" in env["NO_PROXY"]
    # ...but the cloud metadata endpoint must NOT bypass the proxy.
    assert "169.254.169.254" not in env["NO_PROXY"]


def test_sandbox_egress_env_bypass_list_is_fixed_and_not_caller_controlled(monkeypatch):
    from mlflow.server.sandbox.container import _EGRESS_PROXY_BYPASS_HOSTS, sandbox_egress_env

    monkeypatch.setenv("MLFLOW_SANDBOX_EGRESS_PROXY", "http://proxy.internal:3128")
    # NO_PROXY is exactly the fixed self-host bypass list. It is not derived from any request or
    # caller-supplied value, so a remote caller cannot name an internal host (e.g. its own
    # request-derived tracking URI) to carve that destination out of the proxy.
    env = sandbox_egress_env()
    assert env["NO_PROXY"].split(",") == list(_EGRESS_PROXY_BYPASS_HOSTS)


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
