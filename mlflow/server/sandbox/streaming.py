"""Streaming sandbox execution.

``run_in_sandbox`` (see ``container.py``) is run-to-completion: start, wait, read logs. The
assistant's CLI providers (``claude_code``, ``codex``) instead spawn a vendor CLI that streams
JSON events on stdout while it runs and reads its prompt from stdin, so they need a streaming
transport. ``start_sandbox_process`` starts the CLI in a hardened container, feeds stdin from a
buffer via a mounted file, and returns a ``SandboxProcess`` that exposes the container's stdout
as an async line iterator plus ``wait()``/``read_stderr()``/``kill()`` — the subset of the
asyncio subprocess interface the providers rely on.

Two deliberate, documented differences from ``run_in_sandbox``'s hardening:

- stdout is streamed live (``container.logs(stream=True, follow=True)``) rather than read after
  the container exits.
- the root filesystem is writable (the vendor CLI and its language runtime write caches outside
  ``$HOME``), while ``$HOME`` is a bind-mounted host directory holding the CLI's ``--resume``
  session state and caches. ``cap_drop=ALL``, ``no-new-privileges``, the resource caps, the
  network posture, and running as the server's ``uid:gid`` are retained.

The image contents, credential format, and mount behavior are validated with a live Docker
daemon rather than in unit tests (the flag is experimental and off by default).
"""

import asyncio
import logging
import os
import shlex
import shutil
import tempfile
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path

from mlflow.environment_variables import MLFLOW_ASSISTANT_SANDBOX_CLI_IMAGE
from mlflow.server.sandbox.container import (
    _MEMORY_LIMIT,
    _NANO_CPUS,
    _PIDS_LIMIT,
    _WORKSPACE_MOUNT,
    SandboxUnavailableError,
    _get_client,
    ensure_sandbox_network,
    sandbox_container_labels,
    sandbox_egress_env,
)

_logger = logging.getLogger(__name__)

# Mount points inside the container.
_IO_MOUNT = "/sandbox-io"  # read-only: stdin buffer + any input files (e.g. system prompt)
_HOME_MOUNT = "/home/sandbox"  # writable bind mount: CLI caches + --resume state
_STDIN_FILE = "stdin"

# Kill the container if it produces no output for this long. A stuck CLI (e.g. an unreachable
# model endpoint) otherwise streams nothing and the turn would hang indefinitely. This is an
# idle timeout, not a total-runtime cap, so a turn that keeps streaming output is never cut.
_DEFAULT_STREAM_IDLE_TIMEOUT = 120.0
# Max total stdout bytes buffered between the drain thread and the consumer. Bounds memory if the
# CLI emits faster than a slow SSE client drains: the drain thread blocks (applying backpressure
# to the container) once this many unconsumed bytes are in flight, instead of the queue growing
# without limit. Bounding by bytes rather than line count keeps the ceiling fixed regardless of
# how large individual lines are.
_MAX_BUFFERED_BYTES = 256 * 1024 * 1024
# Max bytes for a single stdout line, enforced while accumulating an unterminated line: without
# it, output that never emits a newline would grow the drain buffer without limit and could OOM
# the server. A stream-json line (e.g. a large tool result) can legitimately be tens of MB, so
# this sits above that but within the in-flight budget; exceeding it aborts the stream.
_MAX_LINE_BYTES = 128 * 1024 * 1024


def sandbox_input_path(name: str) -> str:
    """In-container path of an input file passed via ``start_sandbox_process(input_files=...)``."""
    return f"{_IO_MOUNT}/{name}"


class SandboxProcess:
    """A running sandbox container exposed with a subset of the asyncio subprocess interface.

    The providers iterate ``iter_stdout_lines()`` for streamed output, call ``wait()`` for the
    exit code, ``read_stderr()`` on failure, and ``kill()`` to cancel. ``cleanup()`` removes the
    container and the temporary IO directory (and an ephemeral ``$HOME`` directory, if one was
    created because no caller-managed one was supplied).
    """

    def __init__(
        self,
        container,
        io_dir: Path,
        ephemeral_home: Path | None = None,
        idle_timeout: float | None = None,
    ) -> None:
        self._container = container
        self._io_dir = io_dir
        self._ephemeral_home = ephemeral_home
        self._idle_timeout = idle_timeout
        self._returncode: int | None = None
        self._timed_out = False

    @property
    def container_id(self) -> str:
        return self._container.id

    @property
    def returncode(self) -> int | None:
        return self._returncode

    @property
    def timed_out(self) -> bool:
        """Whether the container was killed for producing no output within ``idle_timeout``."""
        return self._timed_out

    async def iter_stdout_lines(self) -> AsyncIterator[bytes]:
        """Yield stdout lines (bytes, newline-stripped) as the container produces them.

        ``container.logs(stream=True, follow=True)`` is a blocking generator, so it is drained
        on a worker thread that hands items to this coroutine through a queue. Buffering is
        bounded to ``_MAX_BUFFERED_BYTES`` of in-flight output: the drain thread blocks before
        enqueuing once that many unconsumed bytes are queued, so a slow SSE client applies
        backpressure to the container rather than letting the queue grow until the server is
        OOM-killed.
        """
        loop = asyncio.get_running_loop()
        # Items are bytes lines, a terminal sentinel, or an Exception raised while draining.
        queue: asyncio.Queue[bytes | object | Exception] = asyncio.Queue()
        # Bounds in-flight bytes: the drain thread waits until a line fits under the budget before
        # enqueuing it (blocking when the consumer is behind) and the consumer subtracts a line's
        # bytes after dequeuing it. Guarded by the condition so both threads see a consistent total.
        budget = threading.Condition()
        buffered_bytes = 0
        sentinel = object()
        last_activity = time.monotonic()
        stop_watchdog = threading.Event()

        def _bump() -> None:
            nonlocal last_activity
            last_activity = time.monotonic()

        def _safe_put(item: object) -> None:
            # The consumer may be torn down (loop closed) while this daemon thread is still
            # blocked in container.logs(); a closed loop makes call_soon_threadsafe raise. Drop
            # the item rather than surface a stray traceback from the thread.
            try:
                loop.call_soon_threadsafe(queue.put_nowait, item)
            except RuntimeError:
                pass

        def _put_line(line: bytes) -> bool:
            # Backpressure: wait until this line fits within the in-flight byte budget before
            # enqueuing, so a slow reader can't grow the queue without bound. A line is always
            # admitted when nothing is buffered (so a single line up to _MAX_LINE_BYTES still
            # passes even if it alone exceeds the budget). Give up if the consumer has torn down
            # (its finally sets stop_watchdog) so this daemon thread does not park forever.
            nonlocal buffered_bytes
            with budget:
                while buffered_bytes and buffered_bytes + len(line) > _MAX_BUFFERED_BYTES:
                    if stop_watchdog.is_set():
                        return False
                    budget.wait(timeout=1.0)
                    # Blocking here means the container produced this line and we are waiting on a
                    # slow consumer — that is backpressure, not an idle container, so keep the idle
                    # watchdog from killing a live turn. A truly gone consumer sets stop_watchdog.
                    _bump()
                buffered_bytes += len(line)
            _safe_put(line)
            return True

        def _drain() -> None:
            # bytearray append is amortized O(1); a plain bytes `+=` is O(n^2) for a single very
            # large stream-json line (tool results can be tens of MB).
            buffer = bytearray()
            try:
                for chunk in self._container.logs(
                    stream=True, follow=True, stdout=True, stderr=False
                ):
                    _bump()
                    buffer.extend(chunk)
                    while (newline := buffer.find(b"\n")) != -1:
                        if not _put_line(bytes(buffer[:newline])):
                            return
                        del buffer[: newline + 1]
                    if len(buffer) > _MAX_LINE_BYTES:
                        # An unterminated line past the cap would grow without bound. Stop the
                        # container so logs() ends, and surface it rather than risking an OOM.
                        self.kill()
                        raise SandboxUnavailableError(
                            f"Sandbox emitted a single stdout line of {len(buffer)} bytes with no "
                            f"newline, over the {_MAX_LINE_BYTES}-byte cap; aborting the stream to "
                            "bound server memory."
                        )
                if buffer and not _put_line(bytes(buffer)):
                    return
            except Exception as e:
                _safe_put(e)
            finally:
                _safe_put(sentinel)

        def _watchdog() -> None:
            # Kill the container if it produces no output for `idle_timeout`; that ends the
            # blocking logs() stream so the drain loop completes and the turn does not hang.
            interval = min(self._idle_timeout, 5.0)
            while not stop_watchdog.wait(interval):
                if time.monotonic() - last_activity > self._idle_timeout:
                    self._timed_out = True
                    self.kill()
                    return

        threading.Thread(target=_drain, name="mlflow-sandbox-stdout-drain", daemon=True).start()
        if self._idle_timeout:
            threading.Thread(
                target=_watchdog, name="mlflow-sandbox-idle-watchdog", daemon=True
            ).start()

        try:
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                if isinstance(item, Exception):
                    raise SandboxUnavailableError(f"Sandbox output stream failed: {item}")
                # Free the bytes this line held so the drain thread may enqueue more.
                with budget:
                    buffered_bytes -= len(item)
                    budget.notify()
                yield item
        finally:
            # The drain thread's budget.wait() times out every second and re-checks this, so it
            # observes the teardown and exits; wake it now so it does not linger up to a second.
            with budget:
                stop_watchdog.set()
                budget.notify_all()

    async def wait(self) -> int:
        outcome = await asyncio.to_thread(self._container.wait)
        self._returncode = outcome.get("StatusCode", 0) if isinstance(outcome, dict) else 0
        return self._returncode

    async def read_stderr(self) -> bytes:
        try:
            return await asyncio.to_thread(lambda: self._container.logs(stdout=False, stderr=True))
        except Exception:
            return b""

    def kill(self) -> None:
        try:
            self._container.kill()
        except Exception:
            pass

    async def akill(self) -> None:
        """``kill()`` off the event loop — it makes a blocking docker-py socket call."""
        await asyncio.to_thread(self.kill)

    def cleanup(self) -> None:
        try:
            self._container.remove(force=True)
        except Exception:
            pass
        shutil.rmtree(self._io_dir, ignore_errors=True)
        if self._ephemeral_home is not None:
            shutil.rmtree(self._ephemeral_home, ignore_errors=True)

    async def aclose(self) -> None:
        """``cleanup()`` off the event loop — force-removing the container blocks on docker-py, so
        running it inline would stall the server on a slow/unhealthy daemon during teardown.
        """
        await asyncio.to_thread(self.cleanup)


def start_sandbox_process(
    argv: list[str],
    *,
    workdir: Path | None = None,
    environment: dict[str, str] | None = None,
    stdin_data: bytes = b"",
    input_files: dict[str, str] | None = None,
    home_dir: Path | None = None,
    idle_timeout: float | None = _DEFAULT_STREAM_IDLE_TIMEOUT,
) -> SandboxProcess:
    """Start ``argv`` in a hardened, streaming sandbox container.

    Args:
        argv: The command to run inside the container (in-container paths).
        workdir: Host directory bind-mounted read-write at ``/workspace`` and used as the
            working directory. When None the container has no workspace mount.
        environment: Extra environment variables to set inside the container.
        stdin_data: Bytes written to a mounted file and fed to ``argv`` on stdin.
        input_files: name -> text files written into the read-only ``/sandbox-io`` mount (e.g.
            a system-prompt file the CLI reads by path).
        home_dir: Caller-managed host directory bind-mounted read-write at ``$HOME`` so CLI
            state (such as ``--resume`` session history) persists across calls. It must be
            writable by the current user (the container runs as this uid:gid). When None an
            ephemeral ``$HOME`` is created and removed by ``cleanup()``.
        idle_timeout: Kill the container if it produces no output for this many seconds (see
            ``SandboxProcess.timed_out``). None disables it. This bounds a stuck CLI without
            cutting a turn that keeps streaming.

    Returns:
        A SandboxProcess wrapping the started container.

    Raises:
        SandboxUnavailableError: If Docker is unavailable, the image is missing, or the
            container cannot be started.
    """
    if os.name == "nt":
        raise SandboxUnavailableError("The sandbox requires a POSIX host running Docker.")

    client = _get_client()
    image = MLFLOW_ASSISTANT_SANDBOX_CLI_IMAGE.get()
    _require_image(client, image)

    io_dir = Path(tempfile.mkdtemp(prefix="mlflow-sandbox-io-"))
    (io_dir / _STDIN_FILE).write_bytes(stdin_data)
    for name, text in (input_files or {}).items():
        (io_dir / name).write_text(text, encoding="utf-8")

    # $HOME must be writable by the container's uid:gid (set below). A caller-managed dir is
    # created server-side (so it is owned by this user); otherwise use an ephemeral one.
    ephemeral_home = None
    if home_dir is None:
        ephemeral_home = Path(tempfile.mkdtemp(prefix="mlflow-sandbox-home-"))
        home_host = ephemeral_home
    else:
        home_host = home_dir

    env = dict(environment or {})
    env.setdefault("HOME", _HOME_MOUNT)
    env.update(sandbox_egress_env())

    volumes = {
        str(io_dir): {"bind": _IO_MOUNT, "mode": "ro"},
        str(home_host): {"bind": _HOME_MOUNT, "mode": "rw"},
    }
    working_dir = None
    if workdir is not None:
        volumes[str(workdir)] = {"bind": _WORKSPACE_MOUNT, "mode": "rw"}
        working_dir = _WORKSPACE_MOUNT

    # Redirect the CLI's stdin from the mounted buffer file rather than opening an attach
    # socket. shlex-join is safe here: argv is built by the provider, not user input, and the
    # container is the isolation boundary regardless.
    command = ["sh", "-c", f"exec {shlex.join(argv)} < {_IO_MOUNT}/{_STDIN_FILE}"]

    try:
        container = client.containers.run(
            image,
            command=command,
            detach=True,
            labels=sandbox_container_labels(),
            network=ensure_sandbox_network(client),
            extra_hosts={"host.docker.internal": "host-gateway"},
            mem_limit=_MEMORY_LIMIT,
            memswap_limit=_MEMORY_LIMIT,
            nano_cpus=_NANO_CPUS,
            pids_limit=_PIDS_LIMIT,
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
            # Run as the server's user so files created in the bind-mounted workspace/HOME are
            # owned by the server, not root. The rootfs stays writable (unlike run_in_sandbox):
            # the CLI and its language runtime write caches outside $HOME. NOTE: if the MLflow
            # server itself runs as root (uid 0), this is 0:gid and the sandbox is root too — the
            # non-root posture holds only when the server runs as a non-root user.
            user=f"{os.getuid()}:{os.getgid()}",
            working_dir=working_dir,
            environment=env,
            volumes=volumes,
        )
    except Exception as e:
        shutil.rmtree(io_dir, ignore_errors=True)
        if ephemeral_home is not None:
            shutil.rmtree(ephemeral_home, ignore_errors=True)
        raise SandboxUnavailableError(f"Failed to start sandbox container: {e}") from e

    return SandboxProcess(
        container, io_dir, ephemeral_home=ephemeral_home, idle_timeout=idle_timeout
    )


def _require_image(client, image: str) -> None:
    """Ensure the CLI sandbox image is present. Unlike the generic run-to-completion sandbox,
    this never auto-builds a fallback: the image must contain the provider CLI, so a missing
    one is an operator setup error, not something to paper over with a CLI-less build.
    """
    import docker.errors

    try:
        client.images.get(image)
    except docker.errors.ImageNotFound as e:
        raise SandboxUnavailableError(
            f"Assistant sandbox image {image!r} was not found. Build or pull an image that "
            "contains the provider CLI, or set MLFLOW_ASSISTANT_SANDBOX_CLI_IMAGE."
        ) from e
    except Exception as e:
        raise SandboxUnavailableError(f"Could not check sandbox image {image!r}: {e}") from e
