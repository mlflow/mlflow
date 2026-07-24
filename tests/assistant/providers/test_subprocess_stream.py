import sys

import pytest

from mlflow.assistant.providers._subprocess_stream import SubprocessLineStream


async def _collect(stream: SubprocessLineStream) -> list[bytes]:
    return [line async for line in stream.lines()]


@pytest.mark.asyncio
async def test_streams_stdout_lines_and_exits_zero():
    stream = SubprocessLineStream(
        [sys.executable, "-c", "print('a'); print('b')"],
    )
    lines = await _collect(stream)
    returncode = await stream.wait()

    assert [line.strip() for line in lines] == [b"a", b"b"]
    assert returncode == 0
    assert stream.returncode == 0
    assert stream.killed is False


@pytest.mark.asyncio
async def test_captures_stderr_on_nonzero_exit():
    stream = SubprocessLineStream(
        [sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"],
    )
    assert await _collect(stream) == []
    assert await stream.wait() == 3
    assert (await stream.read_stderr()).strip() == b"boom"


@pytest.mark.asyncio
async def test_feeds_stdin_from_input_bytes():
    stream = SubprocessLineStream(
        [sys.executable, "-c", "import sys; sys.stdout.write(sys.stdin.read().upper())"],
        input_bytes=b"hello",
    )
    lines = await _collect(stream)
    await stream.wait()

    assert b"".join(lines) == b"HELLO"


@pytest.mark.asyncio
async def test_kill_records_killed_flag():
    # A process that would otherwise run indefinitely.
    stream = SubprocessLineStream(
        [sys.executable, "-c", "import time; time.sleep(60)"],
    )
    stream.kill()
    await stream.wait()

    assert stream.killed is True
    assert stream.returncode is not None


@pytest.mark.asyncio
async def test_pid_is_exposed():
    stream = SubprocessLineStream([sys.executable, "-c", "pass"])
    assert isinstance(stream.pid, int)
    await stream.wait()
