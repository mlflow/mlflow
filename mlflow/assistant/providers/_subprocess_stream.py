"""Event-loop-independent subprocess streaming for Assistant CLI providers."""

import asyncio
import concurrent.futures
import subprocess
import threading
from collections.abc import AsyncIterator
from contextlib import suppress
from pathlib import Path

_MAX_BUFFERED_BYTES = 256 * 1024 * 1024
_MAX_LINE_BYTES = 128 * 1024 * 1024
_MAX_STDERR_BYTES = 1024 * 1024
_EOF = object()


class SubprocessLineStream:
    """Stream a subprocess's stdout without using an asyncio subprocess transport."""

    def __init__(
        self,
        cmd: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        input_bytes: bytes | None = None,
        max_buffered_bytes: int = _MAX_BUFFERED_BYTES,
        max_line_bytes: int = _MAX_LINE_BYTES,
        max_stderr_bytes: int = _MAX_STDERR_BYTES,
    ) -> None:
        self._loop = asyncio.get_running_loop()
        self._queue: asyncio.Queue[bytes | object | Exception] = asyncio.Queue()
        self._budget = threading.Condition()
        self._buffered_bytes = 0
        self._max_buffered_bytes = max_buffered_bytes
        self._max_line_bytes = max_line_bytes
        self._max_stderr_bytes = max_stderr_bytes
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
        pid = self._proc.pid
        self._stdout_thread = threading.Thread(
            target=self._pump_stdout, name=f"subprocess-stdout-{pid}", daemon=True
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, name=f"subprocess-stderr-{pid}", daemon=True
        )
        self._threads = [self._stdout_thread, self._stderr_thread]
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

    def _safe_put(self, item: bytes | object | Exception) -> bool:
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

    def _put_line(self, line: bytes) -> bool:
        with self._budget:
            while self._buffered_bytes and (
                self._buffered_bytes + len(line) > self._max_buffered_bytes
            ):
                if self._closed.is_set():
                    return False
                self._budget.wait(timeout=0.1)
            self._buffered_bytes += len(line)
        if self._safe_put(line):
            return True
        with self._budget:
            self._buffered_bytes -= len(line)
            self._budget.notify()
        return False

    def _pump_stdout(self) -> None:
        assert self._proc.stdout is not None
        try:
            while line := self._proc.stdout.readline(self._max_line_bytes + 1):
                if len(line) > self._max_line_bytes:
                    self.kill()
                    raise RuntimeError(
                        f"Subprocess emitted a stdout line over the "
                        f"{self._max_line_bytes}-byte limit"
                    )
                if not self._put_line(line):
                    return
        except Exception as e:
            self._safe_put(e)
        finally:
            self._safe_put(_EOF)

    def _drain_stderr(self) -> None:
        assert self._proc.stderr is not None
        while chunk := self._proc.stderr.read(64 * 1024):
            self._stderr.extend(chunk)
            if len(self._stderr) > self._max_stderr_bytes:
                del self._stderr[: -self._max_stderr_bytes]

    def _write_stdin(self, data: bytes) -> None:
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.write(data)
        except OSError:
            pass
        finally:
            with suppress(OSError):
                self._proc.stdin.close()

    async def lines(self) -> AsyncIterator[bytes]:
        try:
            while True:
                item = await self._queue.get()
                if item is _EOF:
                    return
                if isinstance(item, Exception):
                    raise item
                with self._budget:
                    self._buffered_bytes -= len(item)
                    self._budget.notify()
                yield item
        finally:
            with self._budget:
                self._closed.set()
                self._budget.notify_all()

    async def wait(self) -> int:
        return await asyncio.to_thread(self._proc.wait)

    async def read_stderr(self) -> bytes:
        await asyncio.to_thread(self._stderr_thread.join)
        return bytes(self._stderr)

    def kill(self) -> None:
        if self._proc.poll() is None:
            self._killed = True
            self._proc.kill()
