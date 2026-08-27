import asyncio
import os
import threading
from pathlib import Path
from unittest import mock

import pytest

from mlflow.server.sandbox import SandboxUnavailableError, start_sandbox_process
from mlflow.server.sandbox import container as container_mod

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="Sandbox relies on POSIX uid/gid and Docker Linux containers"
)


def _run(coro):
    return asyncio.run(coro)


async def _collect(agen):
    return [item async for item in agen]


def _streaming_container(chunks, stderr=b"", status=0):
    container = mock.MagicMock()

    def logs(**kwargs):
        if kwargs.get("stream"):
            return iter(chunks)
        if kwargs.get("stderr") and not kwargs.get("stdout", True):
            return stderr
        return b""

    container.logs.side_effect = logs
    container.wait.return_value = {"StatusCode": status}
    client = mock.MagicMock()
    client.images.get.return_value = mock.MagicMock()  # image present
    client.containers.run.return_value = container
    return client, container


def test_start_sandbox_process_builds_command_and_mounts(tmp_path):
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    workdir = tmp_path / "project"
    workdir.mkdir()
    client, _ = _streaming_container([])
    with mock.patch("docker.from_env", return_value=client):
        proc = start_sandbox_process(
            ["claude", "-p", "--verbose"],
            workdir=workdir,
            environment={"MLFLOW_TRACKING_URI": "http://host.docker.internal:5000"},
            stdin_data=b"hello",
            input_files={"system_prompt.txt": "SYS"},
            home_dir=home_dir,
        )

    _, kwargs = client.containers.run.call_args
    assert kwargs["command"] == ["sh", "-c", "exec claude -p --verbose < /sandbox-io/stdin"]
    assert kwargs["cap_drop"] == ["ALL"]
    assert kwargs["security_opt"] == ["no-new-privileges:true"]
    # The CLI container has a writable rootfs (documented deviation), so read_only is not set,
    # but it still runs as the server user so bind-mounted files are not left root-owned.
    assert "read_only" not in kwargs
    assert kwargs["user"] == f"{os.getuid()}:{os.getgid()}"
    # The streaming container (writable rootfs, running the untrusted CLI) must keep the same
    # resource caps as the run-to-completion sandbox; a regression dropping them should fail here.
    assert kwargs["mem_limit"] == container_mod._MEMORY_LIMIT
    assert kwargs["memswap_limit"] == container_mod._MEMORY_LIMIT
    assert kwargs["nano_cpus"] == container_mod._NANO_CPUS
    assert kwargs["pids_limit"] == container_mod._PIDS_LIMIT
    assert kwargs["working_dir"] == "/workspace"
    assert kwargs["environment"]["HOME"] == "/home/sandbox"

    volumes = kwargs["volumes"]
    assert volumes[str(workdir)] == {"bind": "/workspace", "mode": "rw"}
    assert volumes[str(home_dir)] == {"bind": "/home/sandbox", "mode": "rw"}
    io_host = next(k for k, v in volumes.items() if v["bind"] == "/sandbox-io")
    assert volumes[io_host]["mode"] == "ro"
    assert (Path(io_host) / "stdin").read_bytes() == b"hello"
    assert (Path(io_host) / "system_prompt.txt").read_text() == "SYS"

    proc.cleanup()


def test_start_requires_cli_image_present():
    import docker.errors

    client, _ = _streaming_container([])
    client.images.get.side_effect = docker.errors.ImageNotFound("missing")
    with mock.patch("docker.from_env", return_value=client):
        with pytest.raises(SandboxUnavailableError, match="was not found"):
            start_sandbox_process(["claude", "-p"])
    # The CLI image is never auto-built (would produce a CLI-less image).
    client.images.build.assert_not_called()


def test_ephemeral_home_created_and_cleaned_when_no_home_dir():
    client, _ = _streaming_container([])
    with mock.patch("docker.from_env", return_value=client):
        proc = start_sandbox_process(["claude", "-p"])
        _, kwargs = client.containers.run.call_args
        home_host = next(k for k, v in kwargs["volumes"].items() if v["bind"] == "/home/sandbox")
        assert Path(home_host).exists()
        proc.cleanup()
        assert not Path(home_host).exists()


def test_iter_stdout_lines_splits_streamed_chunks():
    client, _ = _streaming_container([b'{"a":1}\n{"b":', b"2}\nlast"])
    with mock.patch("docker.from_env", return_value=client):
        proc = start_sandbox_process(["claude", "-p"])
        lines = _run(_collect(proc.iter_stdout_lines()))

    assert lines == [b'{"a":1}', b'{"b":2}', b"last"]


def test_iter_stdout_lines_propagates_stream_error():
    client, container = _streaming_container([])

    def logs(**kwargs):
        if kwargs.get("stream"):

            def gen():
                yield b"partial\n"
                raise RuntimeError("stream died")

            return gen()
        return b""

    container.logs.side_effect = logs
    with mock.patch("docker.from_env", return_value=client):
        proc = start_sandbox_process(["x"])
        with pytest.raises(SandboxUnavailableError, match="output stream failed"):
            _run(_collect(proc.iter_stdout_lines()))


def test_wait_returns_exit_code():
    client, _ = _streaming_container([], status=3)
    with mock.patch("docker.from_env", return_value=client):
        proc = start_sandbox_process(["x"])
        rc = _run(proc.wait())

    assert rc == 3
    assert proc.returncode == 3


def test_read_stderr_returns_stderr_logs():
    client, _ = _streaming_container([], stderr=b"boom")
    with mock.patch("docker.from_env", return_value=client):
        proc = start_sandbox_process(["x"])
        err = _run(proc.read_stderr())

    assert err == b"boom"


def test_kill_and_cleanup_remove_container():
    client, container = _streaming_container([])
    with mock.patch("docker.from_env", return_value=client):
        proc = start_sandbox_process(["x"])
        proc.kill()
        proc.cleanup()

    container.kill.assert_called_once()
    container.remove.assert_called_once()


def test_start_raises_when_docker_unavailable():
    with mock.patch("docker.from_env", side_effect=Exception("no daemon")):
        with pytest.raises(SandboxUnavailableError, match="Docker daemon is not reachable"):
            start_sandbox_process(["x"])


def test_start_container_failure_cleans_io_and_raises():
    client, _ = _streaming_container([])
    client.containers.run.side_effect = Exception("boom")
    captured = {}
    real_mkdtemp = __import__("tempfile").mkdtemp

    def _spy_mkdtemp(*args, **kwargs):
        path = real_mkdtemp(*args, **kwargs)
        captured["io_dir"] = path
        return path

    with (
        mock.patch("docker.from_env", return_value=client),
        mock.patch("mlflow.server.sandbox.streaming.tempfile.mkdtemp", side_effect=_spy_mkdtemp),
    ):
        with pytest.raises(SandboxUnavailableError, match="Failed to start sandbox container"):
            start_sandbox_process(["x"])

    # The IO scratch dir is cleaned up when the container fails to start.
    assert not Path(captured["io_dir"]).exists()


def test_iter_stdout_lines_idle_timeout_kills_container():
    stop = threading.Event()
    container = mock.MagicMock()

    def logs(**kwargs):
        if kwargs.get("stream"):

            def gen():
                stop.wait(5)  # block (no output) until the watchdog "kills" the container
                return
                yield  # make this a generator

            return gen()
        return b""

    container.logs.side_effect = logs
    container.kill.side_effect = lambda: stop.set()  # killing ends the (blocked) logs stream
    container.wait.return_value = {"StatusCode": 137}
    client = mock.MagicMock()
    client.images.get.return_value = mock.MagicMock()
    client.containers.run.return_value = container

    with mock.patch("docker.from_env", return_value=client):
        proc = start_sandbox_process(["sleep"], idle_timeout=0.2)
        lines = _run(_collect(proc.iter_stdout_lines()))

    assert proc.timed_out is True
    assert lines == []
    container.kill.assert_called()


def test_no_idle_timeout_when_disabled():
    client, _ = _streaming_container([b"a\n", b"b\n"])
    with mock.patch("docker.from_env", return_value=client):
        proc = start_sandbox_process(["x"], idle_timeout=None)
        lines = _run(_collect(proc.iter_stdout_lines()))
    assert lines == [b"a", b"b"]
    assert proc.timed_out is False
