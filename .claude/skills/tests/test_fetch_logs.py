from collections.abc import AsyncIterator
from pathlib import Path
from unittest import mock

import pytest
import yaml
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from skills.commands.fetch_logs import resolve_urls

SCHEMA_PATH = Path(__file__).parents[1] / "nightly" / "output.schema.yaml"


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


def test_nightly_job_schema_accepts_only_canonical_job_urls() -> None:
    schema = yaml.safe_load(SCHEMA_PATH.read_text())
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema["$defs"]["job"])

    for url in [
        "https://github.com/mlflow/mlflow/actions/runs/123/job/456",
        "https://github.com/mlflow/mlflow/actions/runs/40546956218/job/122340009526",
    ]:
        assert validator.is_valid({
            "workflow_name": "Tests",
            "job_name": "core",
            "job_run_url": url,
        })

    for url in [
        "https://github.com/mlflow/mlflow/actions/runs/123/attempts/2/job/456",
        "https://github.com/mlflow/mlflow/actions/runs/123",
        "https://github.com/other/repo/actions/runs/123/job/456",
        "https://github.com/mlflow/mlflow/actions/runs/abc/job/456",
        "https://github.com/mlflow/mlflow/actions/runs/123/job/abc",
    ]:
        assert not validator.is_valid({
            "workflow_name": "Tests",
            "job_name": "core",
            "job_run_url": url,
        })
