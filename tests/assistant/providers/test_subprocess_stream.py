import asyncio
import sys
import threading

import pytest

from mlflow.assistant.providers._subprocess_stream import SubprocessLineStream


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
