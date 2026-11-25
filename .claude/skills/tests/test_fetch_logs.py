from collections.abc import AsyncIterator
from pathlib import Path
from unittest import mock

import pytest
from skills.commands.fetch_logs import (
    download_raw_log,
    fetch_single_job_logs,
    format_job_output,
    log_block_ranges,
    pytest_summary_ranges,
    resolve_urls,
    step_line_range,
)
from skills.github import GitHubClient, Job, JobStep


@pytest.fixture
def job() -> Job:
    return Job(
        id=456,
        run_id=123,
        url="https://api.github.com/repos/mlflow/mlflow/actions/jobs/456",
        name="Python 3.11",
        workflow_name="Cross-version tests",
        status="completed",
        conclusion="failure",
        html_url="https://github.com/mlflow/mlflow/actions/runs/123/job/456",
        started_at="2026-11-08T01:23:00Z",
        completed_at="2026-11-08T01:23:30Z",
        steps=[
            JobStep(
                name=name,
                status="completed",
                conclusion=conclusion,
                number=number,
                started_at=f"2026-11-08T01:23:{start:02d}Z",
                completed_at=f"2026-11-08T01:23:{end:02d}Z",
            )
            for number, name, conclusion, start, end in [
                (1, "Setup", "success", 0, 9),
                (2, "Run tests", "failure", 10, 15),
                (3, "Run additional tests", "failure", 20, 25),
            ]
        ],
    )


@pytest.mark.asyncio
async def test_fetch_single_job_logs(tmp_path: Path, job: Job) -> None:
    raw_log_path = tmp_path / "123" / "456.log"
    raw_log_path.parent.mkdir()
    lines = [
        (1, ">>> package versions"),
        (2, "pytest==9.1.1"),
        (3, "<<< package versions"),
        (4, ">>> snapshot changes"),
        (5, "Baseline: https://github.com/mlflow/mlflow/actions/runs/122"),
        (6, "Updated pytest v9.1.0 -> v9.1.1"),
        (7, "<<< snapshot changes"),
        (10, "====== test session starts ======"),
        (11, "\x1b[31m====== short test summary info ======\x1b[0m"),
        (12, "FAILED tests/test_example.py::test_example - AssertionError"),
        (15, "====== 1 failed, 42 passed in 12.34s ======"),
        (16, "Between steps"),
        (20, "====== short test summary info ======"),
        (21, "FAILED tests/test_other.py::test_other - ValueError"),
        (25, "====== 1 failed in 5.00s ======"),
        (26, "Cleanup"),
    ]
    raw = "".join(f"2026-11-08T01:23:{seconds:02d}.1234567Z {text}\n" for seconds, text in lines)
    raw_log_path.write_text(raw)
    client = mock.Mock(spec=GitHubClient)

    with mock.patch("skills.commands.fetch_logs.LOG_CACHE_DIR", tmp_path):
        logs = await fetch_single_job_logs(client, job)

    client.get_raw.assert_not_called()
    assert raw_log_path.read_text() == raw.replace("\x1b[31m", "").replace("\x1b[0m", "")
    assert list(raw_log_path.parent.iterdir()) == [raw_log_path]
    assert format_job_output(logs) == (
        "## Cross-version tests / Python 3.11\n"
        "URL: https://github.com/mlflow/mlflow/actions/runs/123/job/456\n"
        f"Raw log: {raw_log_path}\n"
        "Failed step lines: 8-11, 13-15\n"
        "Pytest summary lines: 9-11, 13-15\n"
        "Package versions lines: 2-2\n"
        "Snapshot changes lines: 5-6"
    )


@pytest.mark.asyncio
async def test_download_raw_log_strips_colors_across_chunks(tmp_path: Path, job: Job) -> None:
    chunks = [b"2026-11-08T01:23:10.1234567Z \x1b[3", b"1mFAILED\x1b[0m\n"]

    async def iter_chunks(size: int) -> AsyncIterator[bytes]:
        for chunk in chunks:
            yield chunk

    response = mock.MagicMock()
    response.content.iter_chunked.side_effect = iter_chunks
    response.__aenter__.return_value = response
    client = mock.Mock(spec=GitHubClient)
    client.get_raw = mock.AsyncMock(return_value=response)

    with mock.patch("skills.commands.fetch_logs.LOG_CACHE_DIR", tmp_path):
        log_path = await download_raw_log(client, job)

    client.get_raw.assert_awaited_once_with(f"{job.url}/logs")
    response.__aenter__.assert_awaited_once()
    response.__aexit__.assert_awaited_once()
    response.raise_for_status.assert_called_once_with()
    response.content.iter_chunked.assert_called_once_with(64 * 1024)
    assert log_path.read_text() == "2026-11-08T01:23:10.1234567Z FAILED\n"
    assert list(log_path.parent.iterdir()) == [log_path]


@pytest.mark.parametrize(
    ("failed_steps", "expected"),
    [
        (False, "Conclusion: failure (no failed step recorded)\n"),
        (True, ""),
    ],
)
@pytest.mark.asyncio
async def test_fetch_single_job_logs_without_sections(
    tmp_path: Path, job: Job, failed_steps: bool, expected: str
) -> None:
    raw_log_path = tmp_path / "123" / "456.log"
    raw_log_path.parent.mkdir()
    raw_log_path.write_text("No timestamp or section\n")
    if failed_steps:
        for step in job.steps:
            step.started_at = None
            step.completed_at = None
    else:
        job.steps = []
    client = mock.Mock(spec=GitHubClient)

    with mock.patch("skills.commands.fetch_logs.LOG_CACHE_DIR", tmp_path):
        logs = await fetch_single_job_logs(client, job)

    client.get_raw.assert_not_called()
    assert format_job_output(logs) == (
        "## Cross-version tests / Python 3.11\n"
        "URL: https://github.com/mlflow/mlflow/actions/runs/123/job/456\n"
        f"{expected}"
        f"Raw log: {raw_log_path}"
    )


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        ([], None),
        (["2026-11-08T01:23:09.000Z before", "untimestamped"], None),
        (
            [
                "2026-11-08T01:23:09.999Z before",
                "2026-11-08T01:23:10.000Z",
                "untimestamped continuation",
                "2026-11-08T01:23:15.999Z end",
                "2026-11-08T01:23:16.000Z after",
            ],
            (2, 4),
        ),
        (["2026-11-08T01:23:12.000Z only"], (1, 1)),
    ],
)
def test_step_line_range(
    tmp_path: Path, job: Job, lines: list[str], expected: tuple[int, int] | None
) -> None:
    log_path = tmp_path / "job.log"
    log_path.write_text("\n".join(lines))
    assert step_line_range(log_path, job.steps[1]) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("unrelated\n", []),
        (">>> block\n<<< block\n", []),
        (">>> block\nincomplete\n", []),
        (
            "before\n>>> block\nfirst\nsecond\n<<< block\n"
            ">>> block\nthird\n<<< block\n>>> block\nincomplete\n",
            [(3, 4), (7, 7)],
        ),
        (
            "2026-11-08T01:23:00.000Z >>> block\n"
            "2026-11-08T01:23:01.000Z content\n"
            "2026-11-08T01:23:02.000Z <<< block\n",
            [(2, 2)],
        ),
    ],
)
def test_log_block_ranges(tmp_path: Path, text: str, expected: list[tuple[int, int]]) -> None:
    log_path = tmp_path / "job.log"
    log_path.write_text(text)
    assert log_block_ranges(log_path, ">>> block", "<<< block") == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", []),
        ("====== test session starts ======\nnoise\n", []),
        ("====== 42 passed in 12.34s ======\ncleanup\n", [(1, 1)]),
        ("====== no tests ran in 0.01s ======\ncleanup\n", [(1, 1)]),
        (
            "= 1 failed, 2 passed, 3 skipped, 4 xfailed, 5 xpassed, 6 warnings in 1.00s =\n"
            "cleanup\n",
            [(1, 1)],
        ),
        ("====== short test summary info ======\nFAILED test\n", [(1, 2)]),
        (
            "====== short test summary info ======\nFAILED test\n"
            "= 1 failed, 2 passed, 3 skipped, 4 xfailed, 5 xpassed, 6 warnings in 1.00s =\n"
            "cleanup\n",
            [(1, 3)],
        ),
        (
            "====== short test summary info ======\nFAILED first\n"
            "====== 1 failed in 1s ======\nnoise\n"
            "====== short test summary info ======\nFAILED second\n"
            "====== 1 failed, 2 passed in 2s ======\ncleanup\n",
            [(1, 3), (5, 7)],
        ),
        (
            "====== short test summary info ======\nFAILED test\n"
            "====== warnings summary ======\nwarning\n",
            [(1, 2)],
        ),
    ],
)
def test_pytest_summary_ranges(tmp_path: Path, text: str, expected: list[tuple[int, int]]) -> None:
    log_path = tmp_path / "job.log"
    log_path.write_text(text)
    assert pytest_summary_ranges(log_path) == expected


async def empty_jobs() -> AsyncIterator[None]:
    if False:
        yield None


@pytest.mark.parametrize(
    ("url", "job_id", "attempt"),
    [
        ("https://github.com/mlflow/mlflow/actions/runs/123/job/456", 456, None),
        ("https://github.com/mlflow/mlflow/actions/runs/123/job/456?query=1", 456, None),
        ("https://github.com/mlflow/mlflow/actions/runs/123/job/456#fragment", 456, None),
        ("https://github.com/mlflow/mlflow/actions/runs/123", None, None),
        ("https://github.com/mlflow/mlflow/actions/runs/123?query=1", None, None),
        ("https://github.com/mlflow/mlflow/actions/runs/123#fragment", None, None),
        ("https://github.com/mlflow/mlflow/actions/runs/123/attempts/2", None, 2),
        (
            "https://github.com/mlflow/mlflow/actions/runs/123/attempts/2?query=1",
            None,
            2,
        ),
        (
            "https://github.com/mlflow/mlflow/actions/runs/123/attempts/2#fragment",
            None,
            2,
        ),
    ],
)
@pytest.mark.asyncio
async def test_resolve_urls_routes_complete_job_and_run_paths(
    url: str, job_id: int | None, attempt: int | None
) -> None:
    client = mock.Mock()
    client.get_job = mock.AsyncMock(return_value=mock.sentinel.job)
    client.get_jobs = mock.Mock(side_effect=lambda *args: empty_jobs())

    jobs = await resolve_urls(client, [url])

    if job_id is not None:
        assert jobs == [mock.sentinel.job]
        client.get_job.assert_awaited_once_with("mlflow", "mlflow", job_id)
        client.get_jobs.assert_not_called()
    else:
        assert jobs == []
        client.get_job.assert_not_awaited()
        client.get_jobs.assert_called_once_with("mlflow", "mlflow", 123, attempt)


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/mlflow/mlflow/actions/runs/123/attempts/2/job/456",
        "https://github.com/mlflow/mlflow/actions/runs/123/job/456/attempts/2",
        "https://github.com/mlflow/mlflow/actions/runs/123/job/",
        "https://github.com/mlflow/mlflow/actions/runs/123/attempts/2/job/456/extra",
        "https://github.com/mlflow/mlflow/actions/runs/123/attempts/2/extra",
    ],
)
@pytest.mark.asyncio
async def test_resolve_urls_rejects_malformed_job_and_attempt_paths(url: str) -> None:
    client = mock.Mock()
    client.get_job = mock.AsyncMock()
    client.get_jobs = mock.Mock(side_effect=lambda *args: empty_jobs())

    with pytest.raises(SystemExit, match="^1$"):
        await resolve_urls(client, [url])

    client.get_job.assert_not_awaited()
    client.get_jobs.assert_not_called()
