# Conditions on a native FastAPI route.
#
# Every route wired so far is served by Flask, so the condition gate has only ever been
# reached through `_before_request`. The MCP server API is served natively by FastAPI: its
# validators are async, they take a Starlette request rather than reading Flask's globals,
# and they run with no Flask request context at all.
#
# `POST <server>/tags` is the pilot. It is the smallest route on that surface that names a
# value a condition can judge, and the MCP family mirrors the registry one almost exactly --
# entry tags, version tags, and aliases that live on the entry rather than the version --
# so what works here is the pattern for the rest of it.
#
# These drive the validator directly rather than over HTTP. The async seam and the body read
# are what is new and what could break; a live server would add a great deal of setup
# without exercising anything more of it.

import asyncio
import json
from types import SimpleNamespace

import pytest
from starlette.requests import Request as StarletteRequest

from mlflow.server import auth as auth_module
from mlflow.server.auth import resources as auth_resources
from mlflow.server.auth.conditions import MutationConditionSpec
from mlflow.server.auth.permissions import EDIT, READ
from mlflow.server.mcp_server_api import get_mcp_server_api_route_prefixes

_PREFIX = get_mcp_server_api_route_prefixes()[1]
_SERVER = "acme/search"


@pytest.fixture(autouse=True)
def clean_cache():
    auth_resources.clear_cache()
    yield
    auth_resources.clear_cache()


def _request(path: str, method: str, body=None) -> StarletteRequest:
    """A Starlette request whose body can be awaited once, as a real one can."""
    payload = b"" if body is None else json.dumps(body).encode()
    sent = {"done": False}

    async def receive():
        if sent["done"]:
            return {"type": "http.disconnect"}
        sent["done"] = True
        return {"type": "http.request", "body": payload, "more_body": False}

    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "root_path": "",
    }
    return StarletteRequest(scope, receive)


def _configure(monkeypatch, *, value_condition=None, target_condition=None, permission=EDIT):
    """Grant `permission` on the server and attach the given conditions to the caller."""

    class Store:
        def get_user(self, username):
            return SimpleNamespace(id=1, username=username, is_admin=False)

        def list_mutation_conditions_for_user(self, user_id, workspace, resource_types):
            if value_condition is None and target_condition is None:
                return []
            return [
                MutationConditionSpec(
                    "mcp_server",
                    value_condition=value_condition,
                    target_condition=target_condition,
                )
            ]

    monkeypatch.setattr(auth_module, "store", Store())
    monkeypatch.setattr(auth_module, "_get_mcp_server_permission", lambda name, user: permission)
    monkeypatch.setattr(auth_module, "get_anchor_workspace", lambda rt, rid: "ws")


def _server_with_tags(monkeypatch, **tags):
    monkeypatch.setattr(
        auth_resources,
        "attrs_for_bulk",
        lambda rt, ids: {
            i: auth_resources.values_for_entity(rt, i, SimpleNamespace(tags=dict(tags), aliases={}))
            for i in ids
        },
    )


def _run(path, method, body):
    """Drive the async validator to completion.

    `asyncio.run` rather than `pytest.mark.asyncio`: this repo has no pytest-asyncio (the
    async tests in `test_typesafe_gateway_auth.py` are skipped for exactly that reason), and
    a single awaited call needs no event-loop fixture.
    """

    async def go():
        validator = auth_module._get_mcp_server_validator(path)
        return await validator("alice", _request(path, method, body))

    return asyncio.run(go())


# ---- The request condition ---------------------------------------------------


def test_a_request_condition_denies_a_disallowed_tag(monkeypatch):
    """The proof the feature reaches this surface at all: a grant that permits the write,
    narrowed by a condition that does not.
    """
    _configure(monkeypatch, value_condition="tag_key != 'owner'")
    allowed = _run(f"{_PREFIX}/{_SERVER}/tags", "POST", {"key": "owner", "value": "alice"})
    assert allowed is False


def test_a_request_condition_permits_an_allowed_tag(monkeypatch):
    """The other direction, which is what distinguishes a working condition from a route
    that denies everything.
    """
    _configure(monkeypatch, value_condition="tag_key != 'owner'")
    allowed = _run(f"{_PREFIX}/{_SERVER}/tags", "POST", {"key": "notes", "value": "x"})
    assert allowed is True


def test_the_value_is_judged_and_not_only_the_key(monkeypatch):
    """`tag_value` has to arrive too -- extracting the key alone would leave every
    value-constraining condition vacuous while looking wired.
    """
    _configure(monkeypatch, value_condition="tag_value != 'secret'")
    assert (_run(f"{_PREFIX}/{_SERVER}/tags", "POST", {"key": "notes", "value": "secret"})) is False
    assert (_run(f"{_PREFIX}/{_SERVER}/tags", "POST", {"key": "notes", "value": "fine"})) is True


def test_no_condition_leaves_the_route_as_it_was(monkeypatch):
    """The empty-table case, on this surface as on the Flask one."""
    _configure(monkeypatch)
    allowed = _run(f"{_PREFIX}/{_SERVER}/tags", "POST", {"key": "owner", "value": "alice"})
    assert allowed is True


def test_a_grant_that_denies_is_not_widened_by_a_satisfied_condition(monkeypatch):
    """Grants add, conditions subtract. A condition the request satisfies must not turn a
    READ grant into permission to write.
    """
    _configure(monkeypatch, value_condition="tag_key != 'owner'", permission=READ)
    allowed = _run(f"{_PREFIX}/{_SERVER}/tags", "POST", {"key": "notes", "value": "x"})
    assert allowed is False


# ---- The resource condition --------------------------------------------------


def test_a_resource_condition_reads_the_server_state(monkeypatch):
    """The resource half needs a fetch, which on this surface happens with no Flask context
    and so through the ContextVar cache rather than `g`.
    """
    _configure(monkeypatch, target_condition="tags.stage = 'dev'")
    _server_with_tags(monkeypatch, stage="prod")
    assert (_run(f"{_PREFIX}/{_SERVER}/tags", "POST", {"key": "notes", "value": "x"})) is False

    _server_with_tags(monkeypatch, stage="dev")
    auth_resources.clear_cache()
    assert (_run(f"{_PREFIX}/{_SERVER}/tags", "POST", {"key": "notes", "value": "x"})) is True


def test_both_conditions_must_pass(monkeypatch):
    """Configured together they AND, so satisfying one is not enough."""
    _configure(
        monkeypatch, value_condition="tag_key != 'owner'", target_condition="tags.stage = 'dev'"
    )
    _server_with_tags(monkeypatch, stage="dev")
    assert (_run(f"{_PREFIX}/{_SERVER}/tags", "POST", {"key": "owner", "value": "a"})) is False
    assert (_run(f"{_PREFIX}/{_SERVER}/tags", "POST", {"key": "notes", "value": "a"})) is True


# ---- What must stay unconditioned -------------------------------------------


def test_reading_the_server_is_never_gated(monkeypatch):
    """Conditions gate mutations only. An unsatisfiable one must not make a GET fail."""
    _configure(monkeypatch, value_condition="tag_key = 'impossible'", permission=READ)
    assert (_run(f"{_PREFIX}/{_SERVER}", "GET", None)) is True


def test_an_unrelated_mutation_is_not_gated_by_the_tag_condition(monkeypatch):
    """Only the wired route consults conditions. A server update carries no tag, and
    conditioning it on one would deny a write the admin never restricted.
    """
    _configure(monkeypatch, value_condition="tag_key != 'owner'")
    assert (_run(f"{_PREFIX}/{_SERVER}", "PATCH", {"description": "x"})) is True


def test_a_body_naming_no_tag_is_vacuous(monkeypatch):
    """A malformed body has nothing for a condition to judge, and must not be handed a
    placeholder that gets judged instead. Asserted with a POSITIVE clause: `!=` would pass
    whether the values are absent or invented, so it cannot tell the two apart.
    """
    _configure(monkeypatch, value_condition="tag_key = 'notes'")
    assert (_run(f"{_PREFIX}/{_SERVER}/tags", "POST", {})) is True
