import sqlite3
from unittest.mock import Mock

import pytest
from flask import Flask, g
from sqlalchemy.exc import OperationalError
from sqlparse.exceptions import SQLParseError
from starlette.testclient import TestClient

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import TEMPORARILY_UNAVAILABLE
from mlflow.server import handlers
from mlflow.server._request_keys import SCOPED_EXPERIMENT_FILTER, SCOPED_EXPERIMENT_IDS
from mlflow.server._search_limits import scoped_search_limit_error

from tests.server.test_mcp_server_api import _create_registry_fastapi_app


@pytest.mark.parametrize(
    "cause",
    [
        sqlite3.OperationalError("too many SQL variables"),
        OperationalError("SELECT", [], sqlite3.OperationalError("too many SQL variables")),
        SQLParseError("Maximum number of tokens exceeded (10000)."),
        SQLParseError("Maximum grouping depth exceeded (100)."),
    ],
)
@pytest.mark.parametrize("chaining", ["cause", "context"])
def test_scoped_search_limit_error_sanitizes_limit_errors(cause, chaining):
    error = MlflowException("private scope values and query")
    if chaining == "cause":
        error.__cause__ = cause
    else:
        error.__context__ = cause
    result = scoped_search_limit_error(error)
    assert result.error_code == "INVALID_PARAMETER_VALUE"
    assert result.get_http_status_code() == 400
    assert "authorization scope exceeds a backend limit" in result.message
    assert "private" not in result.message


@pytest.mark.parametrize(
    "cause",
    [
        None,
        sqlite3.OperationalError("database is locked"),
        sqlite3.OperationalError("no such table: experiments"),
        OperationalError("SELECT", [], sqlite3.OperationalError("database is locked")),
        SQLParseError("Invalid syntax"),
        ValueError("too many SQL variables"),
    ],
)
def test_scoped_search_limit_error_preserves_unrelated_errors(cause):
    error = MlflowException("original failure")
    error.__cause__ = cause
    assert scoped_search_limit_error(error) is error


def test_scoped_search_limit_error_handles_cyclic_exception_chain():
    error = MlflowException("original failure")
    cause = ValueError("unrelated")
    error.__cause__ = cause
    cause.__context__ = error
    assert scoped_search_limit_error(error) is error


@pytest.mark.parametrize("sqlstate", ["54000", "22003", "08006"])
@pytest.mark.parametrize("attribute", ["sqlstate", "pgcode"])
def test_scoped_search_limit_error_classifies_postgres_sqlstates(sqlstate, attribute):
    cause = Exception("private query")
    setattr(cause, attribute, sqlstate)
    error = MlflowException("original failure")
    error.__cause__ = OperationalError("SELECT", [], cause)
    result = scoped_search_limit_error(error)
    if sqlstate == "08006":
        assert result is error
    else:
        assert result.error_code == "INVALID_PARAMETER_VALUE"
        assert result.get_http_status_code() == 400


@pytest.mark.parametrize("framework", ["flask", "fastapi"])
def test_search_limit_without_injected_scope_preserves_status(monkeypatch, framework):
    error = MlflowException("original failure", TEMPORARILY_UNAVAILABLE)
    error.__cause__ = OperationalError(
        "SELECT", [], sqlite3.OperationalError("too many SQL variables")
    )
    search = Mock(side_effect=error, __name__="search")
    if framework == "flask":
        app = Flask(__name__)
        app.add_url_rule("/search", view_func=handlers.catch_mlflow_exception(search))
        response = app.test_client().get("/search")
        body = response.json
    else:
        store = Mock(search_mcp_servers=search)
        monkeypatch.setattr(handlers, "_get_tracking_store", lambda: store)
        response = TestClient(_create_registry_fastapi_app()).get("/api/3.0/mlflow/mcp-servers")
        body = response.json()
    assert response.status_code == 503
    assert body["message"] == "original failure"
    assert body["error_code"] == "TEMPORARILY_UNAVAILABLE"
    search.assert_called_once()


@pytest.mark.parametrize(
    ("handler", "method", "payload", "scope_key"),
    [
        (
            handlers._search_experiments,
            "search_experiments",
            {"filter": "experiment_id IN ('1')", "max_results": 1},
            SCOPED_EXPERIMENT_FILTER,
        ),
        (
            handlers._search_logged_models,
            "search_logged_models",
            {"experiment_ids": ["1"]},
            SCOPED_EXPERIMENT_IDS,
        ),
        (
            handlers._batch_get_traces,
            "batch_get_traces",
            {"trace_ids": ["tr-1"], "experiment_ids": ["1"]},
            SCOPED_EXPERIMENT_IDS,
        ),
        (
            handlers._batch_get_trace_infos,
            "batch_get_trace_infos",
            {"trace_ids": ["tr-1"], "experiment_ids": ["1"]},
            SCOPED_EXPERIMENT_IDS,
        ),
    ],
)
@pytest.mark.parametrize("scoped", [False, True])
@pytest.mark.parametrize("limit", [False, True])
def test_scoped_store_call_translates_only_limit_errors(
    monkeypatch, handler, method, payload, scope_key, scoped, limit
):
    error = MlflowException("original failure", TEMPORARILY_UNAVAILABLE)
    error.__cause__ = sqlite3.OperationalError(
        "too many SQL variables" if limit else "database is locked"
    )
    store = Mock()
    search = Mock(side_effect=error)
    setattr(store, method, search)
    monkeypatch.setattr(handlers, "_get_tracking_store", lambda: store)
    app = Flask(__name__)
    app.add_url_rule("/search", view_func=handler, methods=["POST"])

    @app.before_request
    def inject_scope():
        if scoped:
            setattr(g, scope_key, payload.get("filter", ["1"]))

    response = app.test_client().post("/search", json=payload)
    if scoped and limit:
        assert response.status_code == 400
        assert response.json["error_code"] == "INVALID_PARAMETER_VALUE"
        assert "authorization scope exceeds a backend limit" in response.json["message"]
    else:
        assert response.status_code == 503
        assert response.json["message"] == "original failure"
        assert response.json["error_code"] == "TEMPORARILY_UNAVAILABLE"
    search.assert_called_once()


@pytest.mark.parametrize("scope_key", [SCOPED_EXPERIMENT_FILTER, SCOPED_EXPERIMENT_IDS])
def test_http_exception_wrapper_preserves_errors_with_injected_scope(scope_key):
    error = MlflowException("original failure", TEMPORARILY_UNAVAILABLE)
    error.__cause__ = sqlite3.OperationalError("too many SQL variables")
    handler = Mock(side_effect=error, __name__="handler")
    app = Flask(__name__)
    with app.test_request_context():
        setattr(g, scope_key, [])
        response = handlers.catch_mlflow_exception(handler)()
    assert response.status_code == 503
    assert response.json["message"] == "original failure"
    assert response.json["error_code"] == "TEMPORARILY_UNAVAILABLE"
    handler.assert_called_once_with()
