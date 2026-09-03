import errno
import os
import select
import signal
import sys
import time
from pathlib import Path

import pytest

# `pty` (and the `tty`/`termios` it pulls in) is POSIX-only. Guard the import so this helper
# module is importable on Windows; the tests that call `run_interactive` are skipped there.
if sys.platform != "win32":
    import pty


def _read_until(fd: int, output: bytearray, expected: str, timeout: float = 10) -> None:
    """Append PTY output until the expected prompt appears or fail with the transcript.

    Args:
        fd: File descriptor for the controlling pseudo-terminal.
        output: Mutable buffer that accumulates all output read from the terminal.
        expected: Prompt text that signals the caller can send its next response.
        timeout: Maximum seconds to wait for the prompt.
    """
    expected_bytes = expected.encode()
    deadline = time.monotonic() + timeout
    while expected_bytes not in output and time.monotonic() < deadline:
        readable, _, _ = select.select([fd], [], [], 0.1)
        if not readable:
            continue
        try:
            chunk = os.read(fd, 4096)
        except OSError as error:
            if error.errno == errno.EIO:
                break
            raise
        if not chunk:
            break
        output.extend(chunk)
    if expected_bytes not in output:
        pytest.fail(f"Did not see {expected!r} in PTY output:\n{output.decode(errors='replace')}")


def run_interactive(
    command: list[str], cwd: Path, env: dict[str, str], interactions: list[tuple[str, bytes]]
) -> tuple[int, str]:
    """Drive a TTY-based command by responding when each expected prompt appears.

    Args:
        command: Executable and arguments to run in the child process.
        cwd: Working directory for the child process.
        env: Complete environment passed to the child process.
        interactions: Ordered pairs of expected prompt text and response bytes to send.

    Returns:
        The process exit code and complete decoded terminal transcript.
    """
    pid, fd = pty.fork()
    if pid == 0:
        os.chdir(cwd)
        os.execve(command[0], command, env)

    output = bytearray()
    reaped = False
    try:
        for expected, response in interactions:
            _read_until(fd, output, expected)
            os.write(fd, response)

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            readable, _, _ = select.select([fd], [], [], 0.1)
            if readable:
                try:
                    output.extend(os.read(fd, 4096))
                except OSError as error:
                    if error.errno != errno.EIO:
                        raise
            waited_pid, status = os.waitpid(pid, os.WNOHANG)
            if waited_pid == pid:
                reaped = True
                return os.waitstatus_to_exitcode(status), output.decode(errors="replace")
        pytest.fail(f"Interactive process did not exit:\n{output.decode(errors='replace')}")
    finally:
        os.close(fd)
        if not reaped:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            os.waitpid(pid, 0)
