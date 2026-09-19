# The native FastAPI job routes gate fetch/cancel-by-id on ownership (the recorded
# creator must match the caller); submit only requires authentication, and search is
# narrowed to the caller's own jobs by a response filter.

import asyncio
import json
from types import SimpleNamespace

import mlflow.server.jobs as jobs_mod
from mlflow.entities._job import Job as JobEntity
from mlflow.entities._job_status import JobStatus
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.server import auth as a
from mlflow.server import job_api


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


def _job_dict(job_id, creator):
    return {"job_id": job_id, "job_name": "fn", "params": {"secret": job_id}, "creator": creator}


def test_search_response_filter_registered_for_search_endpoint():
    request = SimpleNamespace(scope={"endpoint": job_api.search_jobs})
    assert a._find_fastapi_response_filter(request) is a._filter_search_jobs


def test_search_response_filter_keeps_only_callers_jobs():
    body = json.dumps({
        "jobs": [
            _job_dict("alice-1", "alice"),
            _job_dict("bob-1", "bob"),
            _job_dict("legacy-1", None),
            _job_dict("alice-2", "alice"),
        ]
    }).encode()

    filtered = json.loads(a._filter_search_jobs("alice", body, SimpleNamespace()))

    assert [job["job_id"] for job in filtered["jobs"]] == ["alice-1", "alice-2"]


def test_search_response_filter_hides_creatorless_jobs_from_non_admins():
    # A job with no recorded creator has no owner to match, so it must stay hidden, mirroring
    # the fail-closed ownership check on the per-id routes.
    body = json.dumps({"jobs": [_job_dict("legacy-1", None), _job_dict("bob-1", "bob")]}).encode()

    filtered = json.loads(a._filter_search_jobs("alice", body, SimpleNamespace()))

    assert filtered["jobs"] == []


def test_search_response_filter_handles_empty_result():
    filtered = json.loads(a._filter_search_jobs("alice", b'{"jobs": []}', SimpleNamespace()))
    assert filtered == {"jobs": []}


def test_job_response_model_carries_creator():
    entity = JobEntity(
        job_id="job-1",
        creation_time=1,
        job_name="fn",
        params="{}",
        timeout=None,
        status=JobStatus.PENDING,
        result=None,
        retry_count=0,
        last_update_time=1,
        creator="alice",
    )
    assert job_api.Job.from_job_entity(entity).creator == "alice"


def test_job_id_from_path():
    assert a._job_id_from_path("/ajax-api/3.0/jobs/abc") == "abc"
    assert a._job_id_from_path("/ajax-api/3.0/jobs/cancel/abc") == "abc"
    assert a._job_id_from_path("/ajax-api/3.0/jobs/") is None
    assert a._job_id_from_path("/ajax-api/3.0/jobs/search") is None


def _run_submit_handler(monkeypatch, request_state):
    # Drive the native FastAPI submit handler with the pieces it resolves at call time
    # stubbed out, capturing the creator it forwards to server.jobs.submit_job.
    import mlflow.server.jobs as jobs_mod
    import mlflow.server.jobs.utils as jobs_utils
    from mlflow.server import job_api

    captured = {}

    def _fake_submit(function, params, timeout, creator=None):
        captured["creator"] = creator
        return SimpleNamespace()

    monkeypatch.setattr(jobs_utils, "get_job_fn_fullname", lambda name: "pkg.fn")
    monkeypatch.setattr(jobs_utils, "_load_function", lambda fullname: lambda: None)
    monkeypatch.setattr(jobs_mod, "submit_job", _fake_submit)
    monkeypatch.setattr(job_api.Job, "from_job_entity", classmethod(lambda cls, job: None))

    payload = job_api.SubmitJobPayload(job_name="pkg.fn", params={}, timeout=None)
    job_api.submit_job(payload, SimpleNamespace(state=request_state))
    return captured["creator"]


def test_submit_records_creator_from_request_state(monkeypatch):
    # The FastAPI middleware stamps request.state.username; the handler must forward it as the
    # creator so ownership checks on get/cancel recognize the submitter.
    assert _run_submit_handler(monkeypatch, SimpleNamespace(username="alice")) == "alice"


def test_submit_creator_none_when_unauthenticated(monkeypatch):
    # No username on request.state (auth disabled) -> creator is None, not an error.
    assert _run_submit_handler(monkeypatch, SimpleNamespace()) is None
