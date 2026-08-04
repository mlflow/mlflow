import asyncio
import sys
import threading

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


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only event loop regression")
def test_streams_stdout_on_windows_selector_event_loop():
    async def run():
        stream = SubprocessLineStream([sys.executable, "-c", "print('ok')"])
        lines = await _collect(stream)
        assert await stream.wait() == 0
        assert [line.strip() for line in lines] == [b"ok"]

    loop = asyncio.SelectorEventLoop()
    try:
        loop.run_until_complete(run())
    finally:
        loop.close()


@pytest.mark.asyncio
async def test_stdout_push_blocks_when_queue_is_full():
    stream = SubprocessLineStream(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        queue_max_size=1,
    )

    first_push_finished = threading.Event()
    second_push_finished = threading.Event()

    def push_first_line():
        assert stream._push(b"first\n") is True
        first_push_finished.set()

    def push_second_line():
        assert stream._push(b"second\n") is True
        second_push_finished.set()

    first_thread = threading.Thread(target=push_first_line, name="test-push-first-line")
    thread = threading.Thread(target=push_second_line, name="test-push-second-line")
    first_thread.start()
    await asyncio.wait_for(asyncio.to_thread(first_push_finished.wait), timeout=1)
    thread.start()

    await asyncio.sleep(0.1)
    assert second_push_finished.is_set() is False

    lines = stream.lines()
    assert await anext(lines) == b"first\n"
    await asyncio.wait_for(asyncio.to_thread(second_push_finished.wait), timeout=1)
    await lines.aclose()

    stream.kill()
    await stream.wait()
    first_thread.join(timeout=1)
    thread.join(timeout=1)


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
