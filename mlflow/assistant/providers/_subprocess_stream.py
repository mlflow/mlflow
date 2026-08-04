"""Event-loop-agnostic subprocess streaming for the Assistant's CLI providers.

The ``claude_code`` and ``codex`` providers stream a CLI agent's stdout line by
line from within an ``async`` request handler. Using
``asyncio.create_subprocess_exec`` for this only works on a
``ProactorEventLoop``; on Windows uvicorn serves multi-worker (the default is 4)
or ``--reload`` servers on a ``SelectorEventLoop``, where the spawn raises
``NotImplementedError`` -- so the Assistant's CLI providers return nothing out
of the box (see https://github.com/mlflow/mlflow/issues/24405).

``SubprocessLineStream`` sidesteps the event loop entirely: it spawns with
``subprocess.Popen`` and pumps the pipes on plain threads, handing lines back to
the caller's running loop via ``call_soon_threadsafe``. Because it never creates
an asyncio subprocess transport (and never associates a socket with a completion
port), it runs correctly on any loop and does not perturb uvicorn's serving loop.
"""

import asyncio
import concurrent.futures
import subprocess
import threading
from pathlib import Path

# Sentinel pushed onto the queue when stdout reaches EOF.
_EOF = object()
_QUEUE_MAX_SIZE = 100


class SubprocessLineStream:
    """Stream a subprocess's stdout lines without an asyncio subprocess transport."""

    def __init__(
        self,
        cmd: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        input_bytes: bytes | None = None,
        queue_max_size: int = _QUEUE_MAX_SIZE,
    ):
        self._loop = asyncio.get_running_loop()
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=queue_max_size)
        self._stderr = bytearray()
        self._killed = False
        self._closed = threading.Event()
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=env,
        )
        # Pump stdout and drain stderr on separate threads: reading only stdout
        # while the child fills a ~64KB stderr pipe (or vice versa) would
        # deadlock the child. A dedicated writer thread feeds stdin for the same
        # reason -- writing a large prompt inline could block before the child
        # starts reading.
        pid = self._proc.pid
        self._threads = [
            threading.Thread(
                target=self._pump_stdout, name=f"subprocess-stdout-{pid}", daemon=True
            ),
            threading.Thread(
                target=self._drain_stderr, name=f"subprocess-stderr-{pid}", daemon=True
            ),
        ]
        if input_bytes is not None:
            self._threads.append(
                threading.Thread(
                    target=self._write_stdin,
                    args=(input_bytes,),
                    name=f"subprocess-stdin-{pid}",
                    daemon=True,
                )
            )
        for thread in self._threads:
            thread.start()

    @property
    def pid(self) -> int:
        return self._proc.pid

    @property
    def returncode(self) -> int | None:
        return self._proc.returncode

    @property
    def killed(self) -> bool:
        return self._killed

    def _push(self, item: object) -> bool:
        # Block the stdout thread when the async consumer falls behind. This lets
        # the OS pipe apply backpressure to chatty CLI processes instead of
        # growing this queue without bound.
        try:
            future = asyncio.run_coroutine_threadsafe(self._queue.put(item), self._loop)
        except RuntimeError:
            return False

        while True:
            try:
                future.result(timeout=0.1)
                return True
            except concurrent.futures.TimeoutError:
                if self._closed.is_set() or self._loop.is_closed() or not self._loop.is_running():
                    future.cancel()
                    return False
            except (concurrent.futures.CancelledError, RuntimeError):
                return False

    def _pump_stdout(self) -> None:
        try:
            for line in self._proc.stdout:
                if not self._push(line):
                    return
        finally:
            self._push(_EOF)

    def _drain_stderr(self) -> None:
        self._stderr.extend(self._proc.stderr.read())

    def _write_stdin(self, data: bytes) -> None:
        try:
            self._proc.stdin.write(data)
            self._proc.stdin.close()
        except OSError:
            # The child may exit before consuming all input (e.g. on error).
            pass

    async def lines(self) -> "asyncio.AsyncGenerator[bytes, None]":
        try:
            while True:
                item = await self._queue.get()
                if item is _EOF:
                    return
                yield item
        finally:
            self._closed.set()

    async def wait(self) -> int:
        return await self._loop.run_in_executor(None, self._proc.wait)

    async def read_stderr(self) -> bytes:
        # stderr is fully buffered once its reader thread finishes.
        await self._loop.run_in_executor(None, self._threads[1].join)
        return bytes(self._stderr)

    def kill(self) -> None:
        """Kill the process, recording that termination was initiated by us.

        ``returncode`` alone cannot tell an intentional kill from a genuine
        failure on Windows (both surface as a positive exit code), so callers
        should consult ``killed`` to classify the outcome as interrupted.
        """
        if self._proc.returncode is None:
            self._killed = True
            self._proc.kill()
