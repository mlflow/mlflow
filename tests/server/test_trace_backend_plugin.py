import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
import requests
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import SpanContext, Status, StatusCode, TraceFlags

from mlflow import MlflowClient
from mlflow.entities import IssueReference, Span
from mlflow.server.constants import ARTIFACT_ROOT_ENV_VAR, BACKEND_STORE_URI_ENV_VAR
from mlflow.store.tracking.dbmodels.models import (
    SqlAssessments,
    SqlEntityAssociation,
    SqlIssue,
    SqlSpan,
    SqlTraceInfo,
)
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracing.client import TracingClient
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.utils import find_free_port

from tests.tracing.helper import create_test_trace_info


@pytest.fixture
def json_backend_server(tmp_path, db_uri):
    distribution = tmp_path / "test_json_trace_backend-0.0.0.dist-info"
    distribution.mkdir()
    (distribution / "METADATA").write_text("Name: test-json-trace-backend\nVersion: 0.0.0\n")
    (distribution / "entry_points.txt").write_text(
        "[mlflow.trace_backends]\njson-test = tests.server.trace_backend_plugin:build_backend\n"
    )
    repository = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([
            str(tmp_path),
            str(repository),
            *([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []),
        ]),
        BACKEND_STORE_URI_ENV_VAR: db_uri,
        ARTIFACT_ROOT_ENV_VAR: tmp_path.as_uri(),
        "MLFLOW_TRACE_BACKEND": "json-test",
        "MLFLOW_TEST_JSON_TRACE_BACKEND_PATH": str(tmp_path / "traces.json"),
        "MLFLOW_ENABLE_TELEMETRY": "false",
    }

    @contextmanager
    def start():
        port = find_free_port()
        uri = f"http://127.0.0.1:{port}"
        with subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "mlflow.server.fastapi_app:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--lifespan",
                "off",
            ],
            env=env,
            cwd=repository,
        ) as process:
            try:
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(f"Trace backend server exited: {process.returncode}")
                    try:
                        response = requests.get(f"{uri}/health", timeout=1)
                    except requests.ConnectionError:
                        pass
                    else:
                        if response.status_code == 200:
                            break
                    time.sleep(0.1)
                else:
                    raise TimeoutError("Trace backend server did not become healthy")
                yield MlflowClient(uri), TracingClient(uri)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)

    return start


def test_installed_json_plugin_over_http_survives_server_restart(
    json_backend_server, db_uri, tmp_path
):
    trace_id = f"tr-{123:032x}"
    with json_backend_server() as (client, tracing):
        experiment_id = client.create_experiment("json-plugin")
        other_experiment_id = client.create_experiment("no-traces")
        run = client.create_run(experiment_id)
        client.log_metric(run.info.run_id, "quality", 1.0)
        info = create_test_trace_info(trace_id, experiment_id, request_time=1000)
        assert tracing.start_trace(info).trace_id == trace_id
        span = Span(
            ReadableSpan(
                name="json-plugin-root",
                context=SpanContext(123, 456, is_remote=False, trace_flags=TraceFlags(1)),
                start_time=1_000_000_000,
                end_time=1_001_000_000,
                status=Status(StatusCode.OK),
                resource=Resource.get_empty(),
                attributes={
                    SpanAttributeKey.REQUEST_ID: json.dumps(trace_id),
                    SpanAttributeKey.INPUTS: json.dumps({"question": "hello"}),
                    SpanAttributeKey.OUTPUTS: json.dumps({"answer": "world"}),
                },
            )
        )
        tracing.log_spans(experiment_id, [span])
        assert tracing.store.get_trace(trace_id).data.spans[0].inputs == {"question": "hello"}
        assert tracing.store.batch_get_traces([trace_id])[0].data.spans[0].outputs == {
            "answer": "world"
        }
        assert (
            tracing.store.batch_get_trace_infos([trace_id], experiment_ids=[other_experiment_id])
            == []
        )
        assert [info.trace_id for info in tracing.store.search_traces([experiment_id])[0]] == [
            trace_id
        ]

        issue = tracing._create_issue(
            experiment_id, "Issue", "Independent storage", source_run_id=run.info.run_id
        )
        empty_issue = tracing._create_issue(experiment_id, "No evidence", "No impacted traces")
        assessment = tracing.log_assessment(
            trace_id, IssueReference(issue_id=issue.issue_id, issue_name=issue.name)
        )
        assert assessment.assessment_id is not None
        page = tracing._search_issues(
            experiment_id=experiment_id,
            include_trace_count=True,
            include_total_count=True,
            max_results=1,
        )
        assert [(row.issue_id, row.trace_count) for row in page] == [(issue.issue_id, 1)]
        assert page.total_count == 2
        assert page.token == "memory-issues:1"
        client.link_traces_to_run([trace_id], run.info.run_id)

    native_store = SqlAlchemyStore(db_uri, tmp_path.as_uri())
    try:
        with native_store.ManagedSessionMaker() as session:
            assert session.query(SqlTraceInfo).count() == 0
            assert session.query(SqlSpan).count() == 0
            assert session.query(SqlAssessments).count() == 0
            assert session.query(SqlIssue).count() == 0
            association = session.query(SqlEntityAssociation).one()
            assert association.source_id == trace_id
            assert association.destination_id == run.info.run_id
        assert native_store.get_run(run.info.run_id).data.metrics == {"quality": 1.0}

        with json_backend_server() as (client, tracing):
            trace = tracing.store.get_trace(trace_id)
            assert trace.data.spans[0].inputs == {"question": "hello"}
            assert trace.data.spans[0].outputs == {"answer": "world"}
            assert trace.info.assessments[0].assessment_id == assessment.assessment_id
            assert tracing._get_issue(issue.issue_id).source_run_id == run.info.run_id
            page = tracing._search_issues(
                experiment_id=experiment_id,
                include_trace_count=True,
                max_results=1,
                page_token=page.token,
            )
            assert [(row.issue_id, row.trace_count) for row in page] == [(empty_issue.issue_id, 0)]
            assert client.delete_traces(experiment_id, trace_ids=[trace_id]) == 1
            assert tracing.store.batch_get_trace_infos([trace_id]) == []
            assert {
                row.trace_count
                for row in tracing._search_issues(
                    experiment_id=experiment_id, include_trace_count=True
                )
            } == {0}
            assert client.get_run(run.info.run_id).data.metrics == {"quality": 1.0}

        with native_store.ManagedSessionMaker() as session:
            assert session.query(SqlEntityAssociation).count() == 0
        persisted = json.loads((tmp_path / "traces.json").read_text())
        assert persisted["traces"] == {}
        assert persisted["spans"] == {}
        assert persisted["assessments"] == []
        assert len(persisted["issues"]) == 2
    finally:
        native_store._dispose_engine()
