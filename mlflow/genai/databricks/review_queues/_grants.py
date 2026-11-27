"""
Grants reviewers the access they need when they're added to a review queue, as the review queue
UI does.

Every grant is made as the caller, so it only succeeds where the caller may grant: CAN_MANAGE on
the experiment, or MANAGE on a Unity Catalog object. Granting is best-effort: a grant that fails is
reported in one warning and never fails the call that added the reviewers.
"""

import logging
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from mlflow.exceptions import MlflowException
from mlflow.genai.databricks.review_queues import _client
from mlflow.genai.databricks.review_queues.entities import ItemKind
from mlflow.utils.mlflow_tags import (
    MLFLOW_EXPERIMENT_DATABRICKS_TRACE_ANNOTATIONS_TABLE,
    MLFLOW_EXPERIMENT_DATABRICKS_TRACE_LOG_STORAGE_TABLE,
    MLFLOW_EXPERIMENT_DATABRICKS_TRACE_SPAN_STORAGE_TABLE,
)

_logger = logging.getLogger(__name__)

_EXPERIMENT_TYPE_TAG = "mlflow.experimentType"
_NOTEBOOK_EXPERIMENT_TYPE = "NOTEBOOK"

# Notebook levels nest as CAN_READ < CAN_RUN < CAN_EDIT < CAN_MANAGE, and experiments have the same
# levels without CAN_RUN. A level that isn't listed counts as below all of them.
_LEVEL_RANK = {"CAN_READ": 1, "CAN_RUN": 2, "CAN_EDIT": 3, "CAN_MANAGE": 4}

_READ = ("SELECT",)
_READ_WRITE = ("SELECT", "MODIFY")


@dataclass(frozen=True)
class _Table:
    full_name: str
    catalog: str
    schema: str


@dataclass(frozen=True)
class _Failure:
    grant: str
    error: MlflowException


def _parse_table(full_name: str | None) -> _Table | None:
    if not full_name:
        return None
    catalog, _, rest = full_name.partition(".")
    schema, _, table = rest.partition(".")
    if not (catalog and schema and table):
        return None
    return _Table(full_name=full_name, catalog=catalog, schema=schema)


def grant_reviewer_access(
    experiment_id: str, users: list[str], *, queue_id: str | None = None
) -> None:
    """
    Grant ``users`` the access they need to review a queue in ``experiment_id``:

    - CAN_READ on the experiment when its traces are stored in Unity Catalog, otherwise CAN_EDIT.
      Users who already have that level or higher are left alone, so nobody is downgraded.
    - For Unity Catalog, USE CATALOG and USE SCHEMA, then SELECT on the trace spans and logs
      tables, and SELECT and MODIFY on the trace annotation table and on the table of each dataset
      whose records are in ``queue_id``.

    The caller is skipped, and if they can't be identified, nothing is granted. This never
    raises: grants that fail are listed in one warning.
    """
    try:
        _grant_reviewer_access(experiment_id, users, queue_id)
    except Exception as e:
        # Granting must never fail the call that added the reviewers.
        _logger.warning(
            "Couldn't give %s the access they need to review this queue: %s",
            ", ".join(users),
            e,
        )


def _grant_reviewer_access(experiment_id: str, users: list[str], queue_id: str | None) -> None:
    try:
        caller = _caller()
    except MlflowException as e:
        # The caller might be one of `users`, and having MANAGE on a table doesn't let them read
        # it, so granting anyway could give them data access. Grant nothing instead.
        _logger.warning(
            "Couldn't give %s the access they need to review this queue, because the current "
            "user couldn't be identified: %s",
            ", ".join(users),
            e.message,
        )
        return
    # Adding reviewers shouldn't change the caller's own access.
    if not (users := [u for u in users if u.lower() != caller.lower()]):
        return
    failures: list[_Failure] = []
    trace_tables = _grant_experiment_access(experiment_id, users, failures)
    dataset_tables = _queue_dataset_tables(experiment_id, queue_id, failures) if queue_id else []
    table_grants = trace_tables + [(table, _READ_WRITE) for table in dataset_tables]
    _grant_table_access(users, table_grants, failures)
    if failures:
        _warn(users, failures)


def _caller() -> str:
    if not (name := _client.call_api("GET", "/api/2.0/preview/scim/v2/Me").get("userName")):
        raise MlflowException("The response doesn't include a user name.")
    return name


def _permission_ranks(acl: list[dict[str, Any]]) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for entry in acl:
        principal = entry.get("user_name") or entry.get("service_principal_name")
        if not principal:
            continue
        rank = max(
            (
                _LEVEL_RANK.get(p.get("permission_level"), 0)
                for p in entry.get("all_permissions", [])
            ),
            default=0,
        )
        ranks[principal.lower()] = max(ranks.get(principal.lower(), 0), rank)
    return ranks


def _grant_experiment_access(
    experiment_id: str, users: list[str], failures: list[_Failure]
) -> list[tuple[_Table, tuple[str, ...]]]:
    """Grant ``users`` access to the experiment, and return the trace table grants they need."""
    try:
        experiment = _client.call_api(
            "GET", "/api/2.0/mlflow/experiments/get", params={"experiment_id": experiment_id}
        ).get("experiment", {})
    except MlflowException as e:
        # Its trace tables are unknown too, but the queue's datasets can still be granted.
        failures.append(_Failure(f"access to experiment {experiment_id!r}", e))
        return []

    tags = {tag.get("key"): tag.get("value") for tag in experiment.get("tags", [])}
    trace_tables = [
        (table, privileges)
        for name, privileges in [
            (tags.get(MLFLOW_EXPERIMENT_DATABRICKS_TRACE_ANNOTATIONS_TABLE), _READ_WRITE),
            (tags.get(MLFLOW_EXPERIMENT_DATABRICKS_TRACE_SPAN_STORAGE_TABLE), _READ),
            (tags.get(MLFLOW_EXPERIMENT_DATABRICKS_TRACE_LOG_STORAGE_TABLE), _READ),
        ]
        if (table := _parse_table(name))
    ]
    # Unity Catalog authorizes answers on traces stored there, so viewing the experiment is enough.
    level = "CAN_READ" if trace_tables else "CAN_EDIT"
    is_notebook = tags.get(_EXPERIMENT_TYPE_TAG) == _NOTEBOOK_EXPERIMENT_TYPE
    object_type = "notebooks" if is_notebook else "experiments"
    endpoint = f"/api/2.0/permissions/{object_type}/{experiment_id}"
    try:
        # PATCH replaces a listed user's own entry, so leave out anyone already at or above
        # `level` rather than lower their access.
        ranks = _permission_ranks(_client.call_api("GET", endpoint).get("access_control_list", []))
        if to_grant := [u for u in users if ranks.get(u.lower(), 0) < _LEVEL_RANK[level]]:
            _client.call_api(
                "PATCH",
                endpoint,
                json={
                    "access_control_list": [
                        {"user_name": u, "permission_level": level} for u in to_grant
                    ]
                },
            )
    except MlflowException as e:
        failures.append(_Failure(f"{level} on experiment {experiment_id!r}", e))
    return trace_tables


def _queue_dataset_tables(
    experiment_id: str, queue_id: str, failures: list[_Failure]
) -> list[_Table]:
    dataset_ids: dict[str, None] = {}
    page_token = None
    try:
        while True:
            resp = _client.call(
                "GET",
                f"experiments/{experiment_id}/reviewQueues/{queue_id}/items",
                params={"page_token": page_token},
            )
            for item in resp.get("review_queue_items", []):
                if item.get("kind") == ItemKind.DATASET_RECORD.value:
                    if dataset_id := item.get("dataset_record", {}).get("dataset_id"):
                        dataset_ids[dataset_id] = None
            if not (page_token := resp.get("next_page_token")):
                break
    except MlflowException as e:
        failures.append(_Failure(f"access to the datasets in review queue {queue_id!r}", e))

    tables: list[_Table] = []
    for dataset_id in dataset_ids:
        try:
            name = _client.call("GET", f"datasets/{quote(dataset_id, safe='')}").get("name")
        except MlflowException as e:
            failures.append(_Failure(f"SELECT, MODIFY on the table of dataset {dataset_id!r}", e))
            continue
        if table := _parse_table(name):
            tables.append(table)
    return tables


def _grant_table_access(
    users: list[str], grants: list[tuple[_Table, tuple[str, ...]]], failures: list[_Failure]
) -> None:
    if not grants:
        return
    # Reading or writing a table also needs USE CATALOG and USE SCHEMA on the objects above it.
    # The tables usually share one catalog and schema, so grant each of those once.
    privileges_by_table: dict[str, dict[str, None]] = {}
    for table, privileges in grants:
        privileges_by_table.setdefault(table.full_name, {}).update(dict.fromkeys(privileges))
    securables = [
        *(("catalog", c, ("USE_CATALOG",)) for c in dict.fromkeys(t.catalog for t, _ in grants)),
        *(
            ("schema", s, ("USE_SCHEMA",))
            for s in dict.fromkeys(f"{t.catalog}.{t.schema}" for t, _ in grants)
        ),
        *(("table", name, tuple(p)) for name, p in privileges_by_table.items()),
    ]
    # Each object has its own MANAGE, so one failing says nothing about the rest: try them all.
    for securable_type, name, privileges in securables:
        try:
            _client.call_api(
                "PATCH",
                f"/api/2.1/unity-catalog/permissions/{securable_type}/{quote(name, safe='.')}",
                json={"changes": [{"principal": u, "add": list(privileges)} for u in users]},
            )
        except MlflowException as e:
            readable = ", ".join(p.replace("_", " ") for p in privileges)
            failures.append(_Failure(f"{readable} on {securable_type} {name!r}", e))


def _warn(users: list[str], failures: list[_Failure]) -> None:
    _logger.warning(
        "Couldn't give %s all the access they need to review this queue. Still needed: %s. "
        "Someone with CAN_MANAGE on the experiment, or MANAGE on those Unity Catalog objects, can "
        "grant it. First error: %s",
        ", ".join(users),
        "; ".join(f.grant for f in failures),
        failures[0].error.message,
    )
