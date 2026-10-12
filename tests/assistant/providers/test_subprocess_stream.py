import ast
import asyncio
import errno
import os
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import mlflow
from mlflow.assistant.providers._subprocess_stream import (
    SubprocessLineStream,
    assistant_subprocess_env,
)


async def _collect(stream: SubprocessLineStream) -> list[bytes]:
    return [line async for line in stream.lines()]


@pytest.mark.asyncio
async def test_streams_stdout_stdin_and_stderr():
    stream = SubprocessLineStream(
        [
            sys.executable,
            "-c",
            "import sys; print(sys.stdin.read().upper()); sys.stderr.write('error')",
        ],
        input_bytes=b"hello",
    )

    assert [line.strip() for line in await _collect(stream)] == [b"HELLO"]
    assert await stream.wait() == 0
    assert await stream.read_stderr() == b"error"


@pytest.mark.asyncio
async def test_nonzero_exit_is_exposed_with_stderr():
    stream = SubprocessLineStream([
        sys.executable,
        "-c",
        "import sys; sys.stderr.write('boom'); sys.exit(3)",
    ])

    assert await _collect(stream) == []
    assert await stream.wait() == 3
    assert stream.returncode == 3
    assert await stream.read_stderr() == b"boom"


@pytest.mark.asyncio
async def test_large_stdin_does_not_deadlock():
    data = b"x" * (1024 * 1024)
    stream = SubprocessLineStream(
        [sys.executable, "-c", "import sys; print(len(sys.stdin.buffer.read()))"],
        input_bytes=data,
    )

    assert [line.strip() for line in await _collect(stream)] == [str(len(data)).encode()]
    assert await stream.wait() == 0


@pytest.mark.parametrize(
    "write_error",
    [
        BrokenPipeError("broken pipe"),
        OSError(errno.EPIPE, "broken pipe"),
    ],
)
def test_stdin_pipe_errors_are_ignored(write_error):
    stream = object.__new__(SubprocessLineStream)
    stream._proc = MagicMock()
    stream._proc.stdin.write.side_effect = write_error

    stream._write_stdin(b"prompt")

    stream._proc.stdin.write.assert_called_once_with(b"prompt")
    stream._proc.stdin.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_stdout_backpressure_is_bounded_by_bytes():
    stream = SubprocessLineStream(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        max_buffered_bytes=5,
    )
    first_done = threading.Event()
    second_done = threading.Event()
    first = threading.Thread(
        target=lambda: (stream._put_line(b"12345"), first_done.set()), name="put-first-line"
    )
    second = threading.Thread(
        target=lambda: (stream._put_line(b"6"), second_done.set()), name="put-second-line"
    )
    first.start()
    await asyncio.wait_for(asyncio.to_thread(first_done.wait), 1)
    second.start()
    await asyncio.sleep(0.1)
    assert not second_done.is_set()

    lines = stream.lines()
    assert await anext(lines) == b"12345"
    await asyncio.wait_for(asyncio.to_thread(second_done.wait), 1)
    await lines.aclose()
    stream.kill()
    await stream.wait()
    first.join(timeout=1)
    second.join(timeout=1)


@pytest.mark.asyncio
async def test_oversized_unterminated_line_kills_process():
    stream = SubprocessLineStream(
        [
            sys.executable,
            "-c",
            "import sys,time; sys.stdout.write('x' * 100); sys.stdout.flush(); time.sleep(60)",
        ],
        max_line_bytes=10,
    )

    with pytest.raises(RuntimeError, match="stdout line over the 10-byte limit"):
        await _collect(stream)
    await stream.wait()
    assert stream.killed


@pytest.mark.asyncio
async def test_line_limit_counts_trailing_newline():
    stream = SubprocessLineStream(
        [
            sys.executable,
            "-c",
            "import sys,time; sys.stdout.write('x' * 10 + '\\n'); "
            "sys.stdout.flush(); time.sleep(60)",
        ],
        max_line_bytes=10,
    )

    with pytest.raises(RuntimeError, match="stdout line over the 10-byte limit"):
        await _collect(stream)
    await stream.wait()
    assert stream.killed


@pytest.mark.asyncio
async def test_stderr_keeps_bounded_tail():
    stream = SubprocessLineStream(
        [sys.executable, "-c", "import sys; sys.stderr.write('0123456789')"],
        max_stderr_bytes=4,
    )
    assert await _collect(stream) == []
    await stream.wait()
    assert await stream.read_stderr() == b"6789"


@pytest.mark.asyncio
async def test_kill_records_intentional_termination():
    stream = SubprocessLineStream([sys.executable, "-c", "import time; time.sleep(60)"])

    stream.kill()
    await stream.wait()

    assert stream.killed
    assert stream.returncode is not None


@pytest.mark.asyncio
async def test_closing_consumer_unblocks_producer():
    stream = SubprocessLineStream(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        max_buffered_bytes=1,
    )
    first_done = threading.Event()
    first = threading.Thread(
        target=lambda: (stream._put_line(b"a"), first_done.set()), name="put-initial-line"
    )
    first.start()
    await asyncio.wait_for(asyncio.to_thread(first_done.wait), 1)
    producer_done = threading.Event()
    producer = threading.Thread(
        target=lambda: (stream._put_line(b"b"), producer_done.set()), name="put-blocked-line"
    )
    producer.start()

    lines = stream.lines()
    assert await anext(lines) == b"a"
    await lines.aclose()
    await asyncio.wait_for(asyncio.to_thread(producer_done.wait), 1)
    stream.kill()
    await stream.wait()
    first.join(timeout=1)
    producer.join(timeout=1)


@pytest.mark.asyncio
async def test_cancelling_consumer_task_closes_stream():
    stream = SubprocessLineStream([sys.executable, "-c", "import time; time.sleep(60)"])
    consumer = asyncio.create_task(anext(stream.lines()))
    await asyncio.sleep(0)

    consumer.cancel()
    with pytest.raises(asyncio.CancelledError, match=".*"):
        await consumer

    assert stream._closed.is_set()
    stream.kill()
    await stream.wait()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only event loop regression")
def test_streams_stdout_on_windows_selector_event_loop():
    async def run():
        stream = SubprocessLineStream([sys.executable, "-c", "print('ok')"])
        assert [line.strip() for line in await _collect(stream)] == [b"ok"]
        assert await stream.wait() == 0

    loop = asyncio.SelectorEventLoop()
    try:
        loop.run_until_complete(run())
    finally:
        loop.close()


_SECRET_NAMES = ("_MLFLOW_ASSISTANT_DELEGATION_SIGNING_KEY", "_MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN")


def test_assistant_subprocess_env_leaves_out_the_server_internal_secrets(monkeypatch):
    for name in _SECRET_NAMES:
        monkeypatch.setenv(name, "secret")
    monkeypatch.setenv("SOME_SETTING", "kept")

    env = assistant_subprocess_env(MLFLOW_TRACKING_URI="http://localhost:5000")

    assert not set(_SECRET_NAMES) & set(env)
    assert env["SOME_SETTING"] == "kept"
    assert env["MLFLOW_TRACKING_URI"] == "http://localhost:5000"
    # Also for a given environment, and when a secret comes in as an override.
    given = dict.fromkeys(_SECRET_NAMES, "secret")
    assert assistant_subprocess_env(given, **{_SECRET_NAMES[1]: "secret"}) == {}


@pytest.mark.asyncio
async def test_stream_subprocess_never_gets_the_server_internal_secrets(monkeypatch):
    for name in _SECRET_NAMES:
        monkeypatch.setenv(name, "secret")
    code = f"import os; print(sorted(n for n in {list(_SECRET_NAMES)!r} if n in os.environ))"

    for env in (None, {**os.environ}):
        stream = SubprocessLineStream([sys.executable, "-c", code], env=env)
        assert [line.strip() for line in await _collect(stream)] == [b"[]"]
        assert await stream.wait() == 0


_ENV_SPAWNS = {
    "subprocess": {"run", "Popen", "call", "check_call", "check_output"},
    "asyncio": {"create_subprocess_exec", "create_subprocess_shell"},
}
_ENV_SPAWN_NAMES = {
    "Popen",
    "check_call",
    "check_output",
    "create_subprocess_exec",
    "create_subprocess_shell",
}


def _is_env_spawn(node: ast.AST) -> bool:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return node.attr in _ENV_SPAWNS.get(node.value.id, set())
    return isinstance(node, ast.Name) and node.id in _ENV_SPAWN_NAMES


def _is_os_spawn(node: ast.AST) -> bool:
    # These cannot be given a scrubbed environment, so the Assistant must not use them.
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
        and (node.attr in {"system", "popen"} or node.attr.startswith(("spawn", "exec")))
    )


def _is_scrubbed_env_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "assistant_subprocess_env"
    )


def _unscrubbed_spawn_calls(path: Path) -> list[int]:
    tree = ast.parse(path.read_text())
    # Variables assigned from assistant_subprocess_env(...) anywhere in the module. Names are not
    # tracked per scope: a name assigned from it in one function and reused for an unscrubbed
    # environment in another function is not caught.
    scrubbed_names = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and _is_scrubbed_env_call(node.value)
        for target in node.targets
        if isinstance(target, ast.Name)
    }

    def is_scrubbed(value: ast.AST) -> bool:
        return _is_scrubbed_env_call(value) or (
            isinstance(value, ast.Name) and value.id in scrubbed_names
        )

    unsafe = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _is_os_spawn(node.func):
            unsafe.append(node.lineno)
        # A spawn, either called directly or passed to a helper such as asyncio.to_thread.
        elif _is_env_spawn(node.func) or any(_is_env_spawn(arg) for arg in node.args):
            env = next((kw.value for kw in node.keywords if kw.arg == "env"), None)
            if env is None or not is_scrubbed(env):
                unsafe.append(node.lineno)
    return unsafe


def test_every_assistant_subprocess_is_started_with_a_scrubbed_environment():
    # A spawn without assistant_subprocess_env() can pass on the server's internal secrets.
    root = Path(mlflow.__file__).parent
    # Bundled skills are scripts that a coding agent runs on its own, not code the server runs.
    skills = root / "assistant" / "skills"
    missing = {
        str(path.relative_to(root)): lines
        for directory in ("assistant", "server/assistant")
        for path in sorted((root / directory).rglob("*.py"))
        if not path.is_relative_to(skills)
        if (lines := _unscrubbed_spawn_calls(path))
    }
    assert missing == {}


def test_unscrubbed_spawn_check_flags_unsafe_spawns(tmp_path):
    source = tmp_path / "spawns.py"
    source.write_text(
        "import asyncio, os, subprocess\n"
        "from subprocess import Popen\n"
        "subprocess.run(['a'])\n"  # 3: no env
        "subprocess.run(['a'], env=os.environ.copy())\n"  # 4: not scrubbed
        "asyncio.to_thread(subprocess.run, ['a'], env=None)\n"  # 5: not scrubbed
        "asyncio.create_subprocess_exec('a')\n"  # 6: no env
        "Popen(['a'])\n"  # 7: no env
        "os.system('a')\n"  # 8: cannot be scrubbed
        "subprocess.run(['a'], env=assistant_subprocess_env())\n"  # 9: scrubbed
        "env = assistant_subprocess_env(X='1')\n"
        "asyncio.to_thread(subprocess.run, ['a'], env=env)\n"  # 11: scrubbed
    )

    assert _unscrubbed_spawn_calls(source) == [3, 4, 5, 6, 7, 8]
