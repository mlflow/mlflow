import logging

import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.databricks.review_queues import _client, _grants
from mlflow.protos.databricks_pb2 import PERMISSION_DENIED

_ME = ("GET", "/api/2.0/preview/scim/v2/Me")
_EXPERIMENT = ("GET", "/api/2.0/mlflow/experiments/get")
_PERMISSIONS = "/api/2.0/permissions/experiments/1"
_UC = "/api/2.1/unity-catalog/permissions"
_UC_TAGS = [
    {"key": "mlflow.experiment.databricksTraceAnnotationStorageTable", "value": "main.traces.ann"},
    {"key": "mlflow.experiment.databricksTraceSpanStorageTable", "value": "main.traces.spans"},
    {"key": "mlflow.experiment.databricksTraceLogStorageTable", "value": "main.traces.logs"},
]


def _acl(*entries):
    return {
        "access_control_list": [
            {"user_name": user, "all_permissions": [{"permission_level": level, "inherited": inh}]}
            for user, level, inh in entries
        ]
    }


def _denied():
    return MlflowException("Not allowed to grant.", PERMISSION_DENIED)


class _Api:
    """Answers `_client.call_api` and `_client.call` from fixed routes and records each call."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def call_api(self, method, endpoint, *, json=None, params=None):
        self.calls.append((method, endpoint, json))
        response = self.routes.get((method, endpoint), {})
        if isinstance(response, Exception):
            raise response
        return response

    def call(self, method, path, *, json=None, params=None):
        response = self.routes[(method, path, (params or {}).get("page_token"))]
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture
def api(monkeypatch):
    def make(routes):
        fake = _Api({
            _ME: {"userName": "owner@example.com"},
            ("GET", _PERMISSIONS): _acl(("owner@example.com", "CAN_MANAGE", False)),
            **routes,
        })
        monkeypatch.setattr(_client, "call_api", fake.call_api)
        monkeypatch.setattr(_client, "call", fake.call)
        return fake

    return make


def _dataset_record(dataset_id):
    return {"kind": "DATASET_RECORD", "dataset_record": {"dataset_id": dataset_id}}


def _uc_grant(securable, name, privileges, users):
    return (
        "PATCH",
        f"{_UC}/{securable}/{name}",
        {"changes": [{"principal": u, "add": privileges} for u in users]},
    )


def test_v3_experiment_grants_can_edit_and_leaves_higher_levels_alone(api):
    fake = api({
        _EXPERIMENT: {"experiment": {"tags": []}},
        ("GET", _PERMISSIONS): _acl(
            ("owner@example.com", "CAN_MANAGE", False),
            # An inherited level counts, and names compare without case.
            ("Bob@example.com", "CAN_EDIT", True),
            ("carol@example.com", "CAN_READ", False),
            ("dave@example.com", "CAN_RUN", False),
        ),
    })

    _grants.grant_reviewer_access(
        "1",
        ["alice@example.com", "bob@example.com", "carol@example.com", "dave@example.com"],
    )

    assert fake.calls == [
        (*_ME, None),
        (*_EXPERIMENT, None),
        ("GET", _PERMISSIONS, None),
        (
            "PATCH",
            _PERMISSIONS,
            {
                "access_control_list": [
                    {"user_name": u, "permission_level": "CAN_EDIT"}
                    for u in ["alice@example.com", "carol@example.com", "dave@example.com"]
                ]
            },
        ),
    ]


def test_uc_backed_experiment_grants_can_read_and_the_trace_tables(api):
    fake = api({_EXPERIMENT: {"experiment": {"tags": _UC_TAGS}}})

    _grants.grant_reviewer_access("1", ["alice@example.com", "bob@example.com"])

    users = ["alice@example.com", "bob@example.com"]
    assert fake.calls[2:] == [
        ("GET", _PERMISSIONS, None),
        (
            "PATCH",
            _PERMISSIONS,
            {
                "access_control_list": [
                    {"user_name": u, "permission_level": "CAN_READ"} for u in users
                ]
            },
        ),
        _uc_grant("catalog", "main", ["USE_CATALOG"], users),
        _uc_grant("schema", "main.traces", ["USE_SCHEMA"], users),
        _uc_grant("table", "main.traces.ann", ["SELECT", "MODIFY"], users),
        _uc_grant("table", "main.traces.spans", ["SELECT"], users),
        _uc_grant("table", "main.traces.logs", ["SELECT"], users),
    ]


def test_grants_the_tables_of_datasets_in_the_queue(api):
    items = "experiments/1/reviewQueues/rq1/items"
    fake = api({
        _EXPERIMENT: {"experiment": {"tags": []}},
        ("GET", items, None): {
            "review_queue_items": [_dataset_record("d1"), {"kind": "V4_TRACE"}],
            "next_page_token": "tok",
        },
        ("GET", items, "tok"): {
            "review_queue_items": [_dataset_record("d1"), _dataset_record("d2")]
        },
        ("GET", "datasets/d1", None): {"name": "data.eval.set_a"},
        ("GET", "datasets/d2", None): {"name": "data.other.set_b"},
    })

    _grants.grant_reviewer_access("1", ["alice@example.com"], queue_id="rq1")

    users = ["alice@example.com"]
    assert fake.calls[4:] == [
        _uc_grant("catalog", "data", ["USE_CATALOG"], users),
        _uc_grant("schema", "data.eval", ["USE_SCHEMA"], users),
        _uc_grant("schema", "data.other", ["USE_SCHEMA"], users),
        _uc_grant("table", "data.eval.set_a", ["SELECT", "MODIFY"], users),
        _uc_grant("table", "data.other.set_b", ["SELECT", "MODIFY"], users),
    ]


def test_notebook_experiment_uses_notebook_permissions(api):
    notebook = "/api/2.0/permissions/notebooks/1"
    fake = api({
        _EXPERIMENT: {
            "experiment": {"tags": [{"key": "mlflow.experimentType", "value": "NOTEBOOK"}]}
        },
        ("GET", notebook): _acl(),
    })

    _grants.grant_reviewer_access("1", ["alice@example.com"])

    assert [c[:2] for c in fake.calls[2:]] == [("GET", notebook), ("PATCH", notebook)]


@pytest.mark.parametrize("name", ["", "main", "main.traces", "main..ann", ".traces.ann"])
def test_tag_that_isnt_a_table_name_means_v3(api, name):
    tag = {"key": "mlflow.experiment.databricksTraceAnnotationStorageTable", "value": name}
    fake = api({_EXPERIMENT: {"experiment": {"tags": [tag]}}})

    _grants.grant_reviewer_access("1", ["alice@example.com"])

    assert fake.calls[-1][2]["access_control_list"][0]["permission_level"] == "CAN_EDIT"
    assert not any(_UC in c[1] for c in fake.calls)


def test_failed_grants_are_warned_and_every_grant_is_attempted(api, caplog):
    fake = api({
        _EXPERIMENT: {"experiment": {"tags": _UC_TAGS}},
        ("PATCH", _PERMISSIONS): _denied(),
        ("PATCH", f"{_UC}/catalog/main"): _denied(),
        ("PATCH", f"{_UC}/table/main.traces.spans"): _denied(),
    })

    with caplog.at_level(logging.WARNING, logger=_grants.__name__):
        _grants.grant_reviewer_access("1", ["alice@example.com"])

    assert [c[1] for c in fake.calls if c[0] == "PATCH"] == [
        _PERMISSIONS,
        f"{_UC}/catalog/main",
        f"{_UC}/schema/main.traces",
        f"{_UC}/table/main.traces.ann",
        f"{_UC}/table/main.traces.spans",
        f"{_UC}/table/main.traces.logs",
    ]
    [record] = caplog.records
    assert "Couldn't give alice@example.com all the access" in record.message
    assert (
        "Still needed: CAN_READ on experiment '1'; USE CATALOG on catalog 'main'; "
        "SELECT on table 'main.traces.spans'."
    ) in record.message
    assert "Not allowed to grant." in record.message


def test_experiment_lookup_failure_is_warned_and_grants_nothing(api, caplog):
    fake = api({_EXPERIMENT: MlflowException("Experiment not found.")})

    with caplog.at_level(logging.WARNING, logger=_grants.__name__):
        _grants.grant_reviewer_access("1", ["alice@example.com"])

    assert [c[:2] for c in fake.calls] == [_ME, _EXPERIMENT]
    assert "Still needed: access to experiment '1'" in caplog.records[0].message


def test_dataset_that_cant_be_found_is_warned(api, caplog):
    items = "experiments/1/reviewQueues/rq1/items"
    api({
        _EXPERIMENT: {"experiment": {"tags": []}},
        ("GET", items, None): {"review_queue_items": [_dataset_record("d1")]},
        ("GET", "datasets/d1", None): MlflowException("Dataset not found."),
    })

    with caplog.at_level(logging.WARNING, logger=_grants.__name__):
        _grants.grant_reviewer_access("1", ["alice@example.com"], queue_id="rq1")

    assert "SELECT, MODIFY on the table of dataset 'd1'" in caplog.records[0].message


def test_caller_lookup_failure_still_grants_the_others(api):
    fake = api({_ME: _denied(), _EXPERIMENT: {"experiment": {"tags": []}}})

    _grants.grant_reviewer_access("1", ["owner@example.com", "alice@example.com"])

    # The caller wasn't recognized, but already has CAN_MANAGE, so only alice is granted.
    assert fake.calls[-1][2] == {
        "access_control_list": [{"user_name": "alice@example.com", "permission_level": "CAN_EDIT"}]
    }


def test_only_the_caller_means_nothing_to_grant(api):
    fake = api({})

    _grants.grant_reviewer_access("1", ["Owner@example.com"])

    assert fake.calls == [(*_ME, None)]


def test_unexpected_error_is_warned_instead_of_raised(monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise RuntimeError("No credentials.")

    monkeypatch.setattr(_client, "call_api", broken)
    monkeypatch.setattr(_grants, "_without_caller", lambda users: users)

    with caplog.at_level(logging.WARNING, logger=_grants.__name__):
        _grants.grant_reviewer_access("1", ["alice@example.com"])

    assert "Couldn't give alice@example.com the access" in caplog.records[0].message
    assert "No credentials." in caplog.records[0].message
