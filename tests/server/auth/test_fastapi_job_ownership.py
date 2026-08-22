# The native FastAPI job routes gate fetch/cancel-by-id on ownership (the recorded
# creator must match the caller); submit and search only require authentication.

import asyncio
from types import SimpleNamespace

import mlflow.server.jobs as jobs_mod
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.server import auth as a


def _decide(path, method, username, monkeypatch, *, creator="alice", missing=False):
    def _get_job(job_id):
        if missing:
            raise MlflowException("no such job", error_code=RESOURCE_DOES_NOT_EXIST)
        return SimpleNamespace(creator=creator)

    monkeypatch.setattr(jobs_mod, "get_job", _get_job)
    validator = a._find_fastapi_validator(path, method)
    return asyncio.run(validator(username, None))


def test_get_job_owner_allowed(monkeypatch):
    assert _decide("/ajax-api/3.0/jobs/job-1", "GET", "alice", monkeypatch, creator="alice") is True


def test_get_job_non_owner_denied(monkeypatch):
    assert _decide("/ajax-api/3.0/jobs/job-1", "GET", "bob", monkeypatch, creator="alice") is False


def test_get_job_no_creator_denied(monkeypatch):
    assert _decide("/ajax-api/3.0/jobs/job-1", "GET", "alice", monkeypatch, creator=None) is False


def test_get_missing_job_denied(monkeypatch):
    assert _decide("/ajax-api/3.0/jobs/job-x", "GET", "alice", monkeypatch, missing=True) is False


def test_cancel_owner_allowed(monkeypatch):
    path = "/ajax-api/3.0/jobs/cancel/job-1"
    assert _decide(path, "PATCH", "alice", monkeypatch, creator="alice") is True


def test_cancel_non_owner_denied(monkeypatch):
    path = "/ajax-api/3.0/jobs/cancel/job-1"
    assert _decide(path, "PATCH", "bob", monkeypatch, creator="alice") is False


def test_submit_requires_only_authentication():
    # POST /jobs/ has no job id: any authenticated user may submit (creator is recorded).
    validator = a._find_fastapi_validator("/ajax-api/3.0/jobs/", "POST")
    assert asyncio.run(validator("anyone", None)) is True


def test_search_requires_only_authentication():
    validator = a._find_fastapi_validator("/ajax-api/3.0/jobs/search", "POST")
    assert asyncio.run(validator("anyone", None)) is True


def test_job_id_from_path():
    assert a._job_id_from_path("/ajax-api/3.0/jobs/abc") == "abc"
    assert a._job_id_from_path("/ajax-api/3.0/jobs/cancel/abc") == "abc"
    assert a._job_id_from_path("/ajax-api/3.0/jobs/") is None
    assert a._job_id_from_path("/ajax-api/3.0/jobs/search") is None
