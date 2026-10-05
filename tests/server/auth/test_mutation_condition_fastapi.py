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
from mlflow.server.auth.permissions import EDIT, MANAGE, READ
from mlflow.server.mcp_server_api import get_mcp_server_api_route_prefixes
from mlflow.store.condition_pushdown import DECLINED

_PREFIX = get_mcp_server_api_route_prefixes()[1]
_SERVER = "acme/search"


@pytest.fixture(autouse=True)
def _pushdown_declines(monkeypatch):
    """Make the stores decline pushdown, so these tests pin the fallback path.

    The cases in this module stub the *resource layer* -- enumerators, bulk
    loaders, counting shims -- rather than the store, so once the gate started
    asking the store first they reached the real default store and failed on a
    missing database. Declining here keeps them exercising the enumerate-and-judge
    path, which is still what every non-SQL backend uses, and is therefore a path
    that needs its own coverage rather than being an accident of the stubs.

    Autouse but not binding: a test that wants the pushdown consulted can
    monkeypatch the store again, and its own patch wins.
    """
    from types import SimpleNamespace

    from mlflow.server import auth as auth_module

    declining = SimpleNamespace(
        find_failing_resource=lambda *a, **k: DECLINED,
    )
    monkeypatch.setattr(auth_module, "_get_tracking_store", lambda: declining)
    monkeypatch.setattr(auth_module, "_get_model_registry_store", lambda: declining, raising=False)


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

        def is_workspace_admin(self, user_id, workspace):
            return False

        def list_mutation_conditions_for_user(
            self, user_id, workspace, resource_types, parents=None
        ):
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
    # The alias and delete branches consult the version tier before reaching conditions. It has
    # its own tests; stubbing it to allow isolates what these assert. That the grant half still
    # governs is covered by `test_a_grant_that_denies_is_not_widened_by_a_satisfied_condition`,
    # which goes through the real permission check.
    monkeypatch.setattr(auth_module, "_mcp_server_version_action_allowed", lambda u, n, a: True)


def _server_with_tags(monkeypatch, **tags):
    monkeypatch.setattr(
        auth_resources,
        "attrs_for_bulk",
        lambda rt, ids: {
            i: auth_resources.values_for_entity(rt, i, SimpleNamespace(tags=dict(tags), aliases={}))
            for i in ids
        },
    )


async def _run(path, method, body):
    """Resolve the route's validator and await it, as the middleware does."""
    validator = auth_module._get_mcp_server_validator(path)
    return await validator("alice", _request(path, method, body))


# ---- The request condition ---------------------------------------------------


@pytest.mark.asyncio
async def test_a_request_condition_denies_a_disallowed_tag(monkeypatch):
    """The proof the feature reaches this surface at all: a grant that permits the write,
    narrowed by a condition that does not.
    """
    _configure(monkeypatch, value_condition="tag_key != 'owner'")
    allowed = await _run(f"{_PREFIX}/{_SERVER}/tags", "POST", {"key": "owner", "value": "alice"})
    assert allowed is False


@pytest.mark.asyncio
async def test_a_request_condition_permits_an_allowed_tag(monkeypatch):
    """The other direction, which is what distinguishes a working condition from a route
    that denies everything.
    """
    _configure(monkeypatch, value_condition="tag_key != 'owner'")
    allowed = await _run(f"{_PREFIX}/{_SERVER}/tags", "POST", {"key": "notes", "value": "x"})
    assert allowed is True


@pytest.mark.asyncio
async def test_the_value_is_judged_and_not_only_the_key(monkeypatch):
    """`tag_value` has to arrive too -- extracting the key alone would leave every
    value-constraining condition vacuous while looking wired.
    """
    _configure(monkeypatch, value_condition="tag_value != 'secret'")
    assert (
        await _run(f"{_PREFIX}/{_SERVER}/tags", "POST", {"key": "notes", "value": "secret"})
    ) is False
    assert (
        await _run(f"{_PREFIX}/{_SERVER}/tags", "POST", {"key": "notes", "value": "fine"})
    ) is True


@pytest.mark.asyncio
async def test_no_condition_leaves_the_route_as_it_was(monkeypatch):
    """The empty-table case, on this surface as on the Flask one."""
    _configure(monkeypatch)
    allowed = await _run(f"{_PREFIX}/{_SERVER}/tags", "POST", {"key": "owner", "value": "alice"})
    assert allowed is True


@pytest.mark.asyncio
async def test_a_grant_that_denies_is_not_widened_by_a_satisfied_condition(monkeypatch):
    """Grants add, conditions subtract. A condition the request satisfies must not turn a
    READ grant into permission to write.
    """
    _configure(monkeypatch, value_condition="tag_key != 'owner'", permission=READ)
    allowed = await _run(f"{_PREFIX}/{_SERVER}/tags", "POST", {"key": "notes", "value": "x"})
    assert allowed is False


# ---- The resource condition --------------------------------------------------


@pytest.mark.asyncio
async def test_a_resource_condition_reads_the_server_state(monkeypatch):
    """The resource half needs a fetch, which on this surface happens with no Flask context
    and so through the ContextVar cache rather than `g`.
    """
    _configure(monkeypatch, target_condition="tags.stage = 'dev'")
    _server_with_tags(monkeypatch, stage="prod")
    assert (
        await _run(f"{_PREFIX}/{_SERVER}/tags", "POST", {"key": "notes", "value": "x"})
    ) is False

    _server_with_tags(monkeypatch, stage="dev")
    auth_resources.clear_cache()
    assert (await _run(f"{_PREFIX}/{_SERVER}/tags", "POST", {"key": "notes", "value": "x"})) is True


@pytest.mark.asyncio
async def test_both_conditions_must_pass(monkeypatch):
    """Configured together they AND, so satisfying one is not enough."""
    _configure(
        monkeypatch, value_condition="tag_key != 'owner'", target_condition="tags.stage = 'dev'"
    )
    _server_with_tags(monkeypatch, stage="dev")
    assert (
        await _run(f"{_PREFIX}/{_SERVER}/tags", "POST", {"key": "owner", "value": "a"})
    ) is False
    assert (await _run(f"{_PREFIX}/{_SERVER}/tags", "POST", {"key": "notes", "value": "a"})) is True


# ---- What must stay unconditioned -------------------------------------------


@pytest.mark.asyncio
async def test_reading_the_server_is_never_gated(monkeypatch):
    """Conditions gate mutations only. An unsatisfiable one must not make a GET fail."""
    _configure(monkeypatch, value_condition="tag_key = 'impossible'", permission=READ)
    assert (await _run(f"{_PREFIX}/{_SERVER}", "GET", None)) is True


@pytest.mark.asyncio
async def test_an_unrelated_mutation_is_not_gated_by_the_tag_condition(monkeypatch):
    """Only the wired route consults conditions. A server update carries no tag, and
    conditioning it on one would deny a write the admin never restricted.
    """
    _configure(monkeypatch, value_condition="tag_key != 'owner'")
    assert (await _run(f"{_PREFIX}/{_SERVER}", "PATCH", {"description": "x"})) is True


@pytest.mark.asyncio
async def test_a_body_naming_no_tag_is_vacuous(monkeypatch):
    """A malformed body has nothing for a condition to judge, and must not be handed a
    placeholder that gets judged instead. Asserted with a POSITIVE clause: `!=` would pass
    whether the values are absent or invented, so it cannot tell the two apart.
    """
    _configure(monkeypatch, value_condition="tag_key = 'notes'")
    assert (await _run(f"{_PREFIX}/{_SERVER}/tags", "POST", {})) is True


# ---- The rest of the MCP mutation surface ------------------------------------
#
# Six routes carry a value a condition can judge: the server's own tags (set and delete), its
# aliases (set and delete), and a version's tags (set and delete). The grant checks for these
# return from several different branches of the validator, so conditions are applied once
# after the grant decision rather than at each of them.


def _version_configured(monkeypatch, *, value_condition=None, target_condition=None):
    """Conditions on `mcp_server_version` rather than on the server."""

    class Store:
        def get_user(self, username):
            return SimpleNamespace(id=1, username=username, is_admin=False)

        def is_workspace_admin(self, user_id, workspace):
            return False

        def list_mutation_conditions_for_user(
            self, user_id, workspace, resource_types, parents=None
        ):
            return [
                MutationConditionSpec(
                    "mcp_server_version",
                    value_condition=value_condition,
                    target_condition=target_condition,
                )
            ]

    monkeypatch.setattr(auth_module, "store", Store())
    monkeypatch.setattr(auth_module, "_get_mcp_server_permission", lambda name, user: MANAGE)
    monkeypatch.setattr(auth_module, "_mcp_server_version_action_allowed", lambda u, n, a: True)
    monkeypatch.setattr(auth_module, "get_anchor_workspace", lambda rt, rid: "ws")


# ---- Server tag delete: the key is a path segment ----------------------------


@pytest.mark.asyncio
async def test_deleting_a_server_tag_is_conditioned(monkeypatch):
    """The DELETE form takes its key from the path, and its grant check returns from a
    different branch than the POST form -- so wiring one says nothing about the other.
    """
    _configure(monkeypatch, value_condition="tag_key != 'owner'", permission=MANAGE)
    assert (await _run(f"{_PREFIX}/{_SERVER}/tags/owner", "DELETE", None)) is False
    assert (await _run(f"{_PREFIX}/{_SERVER}/tags/notes", "DELETE", None)) is True


@pytest.mark.asyncio
async def test_a_tag_delete_carries_a_none_value(monkeypatch):
    """A delete names a key but no value, and that absence must stay an absence rather than
    becoming an empty string.

    `!=` is the clause that can tell the two apart here, which is the mirror image of the usual
    trap. An absent value makes the clause vacuous, so it permits; an empty-string placeholder
    would make `tag_value != ''` FAIL and deny a delete the admin never restricted. With
    `tag_value = ''` both cases permit, so it proves nothing.
    """
    _configure(monkeypatch, value_condition="tag_value != ''", permission=MANAGE)
    assert (await _run(f"{_PREFIX}/{_SERVER}/tags/notes", "DELETE", None)) is True


# ---- Aliases: conditioned on the server, not the version (D18) --------------


@pytest.mark.asyncio
async def test_setting_an_alias_is_conditioned(monkeypatch):
    _configure(monkeypatch, value_condition="alias != 'production'")
    assert (
        await _run(
            f"{_PREFIX}/{_SERVER}/aliases", "POST", {"alias": "production", "version": "1.0.0"}
        )
    ) is False
    assert (
        await _run(f"{_PREFIX}/{_SERVER}/aliases", "POST", {"alias": "staging", "version": "1.0.0"})
    ) is True


@pytest.mark.asyncio
async def test_deleting_an_alias_is_conditioned(monkeypatch):
    """Gated as well as the set: leaving delete open would let a restricted alias be removed
    and then recreated elsewhere (D12).
    """
    _configure(monkeypatch, value_condition="alias != 'production'", permission=MANAGE)
    assert (await _run(f"{_PREFIX}/{_SERVER}/aliases/production", "DELETE", None)) is False
    assert (await _run(f"{_PREFIX}/{_SERVER}/aliases/staging", "DELETE", None)) is True


@pytest.mark.asyncio
async def test_an_alias_condition_does_not_gate_a_tag_route(monkeypatch):
    """D20 in the permitting direction. Asserted with a positive clause, since `!=` would pass
    whether the alias is absent or invented.
    """
    _configure(monkeypatch, value_condition="alias = 'production'")
    assert (await _run(f"{_PREFIX}/{_SERVER}/tags", "POST", {"key": "notes", "value": "x"})) is True


@pytest.mark.asyncio
async def test_a_tag_condition_does_not_gate_an_alias_route(monkeypatch):
    """And the other way around."""
    _configure(monkeypatch, value_condition="tag_key = 'notes'")
    assert (
        await _run(f"{_PREFIX}/{_SERVER}/aliases", "POST", {"alias": "staging", "version": "1.0.0"})
    ) is True


# ---- Version tags: conditioned on the version's own id ----------------------


@pytest.mark.asyncio
async def test_setting_a_version_tag_is_conditioned(monkeypatch):
    _version_configured(monkeypatch, value_condition="tag_key != 'approved'")
    base = f"{_PREFIX}/{_SERVER}/versions/1.2.0/tags"
    assert (await _run(base, "POST", {"key": "approved", "value": "yes"})) is False
    assert (await _run(base, "POST", {"key": "notes", "value": "yes"})) is True


@pytest.mark.asyncio
async def test_deleting_a_version_tag_is_conditioned(monkeypatch):
    _version_configured(monkeypatch, value_condition="tag_key != 'approved'")
    base = f"{_PREFIX}/{_SERVER}/versions/1.2.0/tags"
    assert (await _run(f"{base}/approved", "DELETE", None)) is False
    assert (await _run(f"{base}/notes", "DELETE", None)) is True


@pytest.mark.asyncio
async def test_a_version_condition_reads_the_versions_own_state(monkeypatch):
    """The resource condition must be evaluated against the VERSION, not its server.

    Reading the server's tags here would widen every version condition to its parent: a
    condition meant to protect one version would be satisfied by a tag on the server.
    """
    _version_configured(monkeypatch, target_condition="tags.stage = 'dev'")
    seen = []

    def attrs_for_bulk(resource_type, ids):
        seen.append((resource_type, list(ids)))
        return {
            i: auth_resources.values_for_entity(
                resource_type, i, SimpleNamespace(tags={"stage": "prod"}, aliases={})
            )
            for i in ids
        }

    monkeypatch.setattr(auth_resources, "attrs_for_bulk", attrs_for_bulk)
    allowed = await _run(
        f"{_PREFIX}/{_SERVER}/versions/1.2.0/tags", "POST", {"key": "notes", "value": "x"}
    )
    assert allowed is False
    assert seen == [("mcp_server_version", ["acme%2Fsearch/1.2.0"])], (
        f"the version's own id must be read, not the server's; got {seen}"
    )


@pytest.mark.asyncio
async def test_a_server_condition_does_not_gate_a_version_tag(monkeypatch):
    """A condition on `mcp_server` must not reach a version route. The two are separate types
    precisely so an admin can restrict one without the other.
    """
    _configure(monkeypatch, value_condition="tag_key = 'nothing_matches'", permission=MANAGE)
    monkeypatch.setattr(auth_module, "_mcp_server_version_action_allowed", lambda u, n, a: True)
    allowed = await _run(
        f"{_PREFIX}/{_SERVER}/versions/1.2.0/tags", "POST", {"key": "notes", "value": "x"}
    )
    assert allowed is True


# ---- Creates carry no tags --------------------------------------------------


@pytest.mark.asyncio
async def test_creating_a_server_is_not_conditioned(monkeypatch):
    """`CreateMCPServerRequest` has no tags field, so there is nothing for a request condition
    to judge and no context is declared. Asserted so the absence is a decision on the record
    rather than an oversight.
    """
    _configure(monkeypatch, value_condition="tag_key = 'nothing_matches'")
    monkeypatch.setattr(auth_module, "validate_can_create_mcp_server", lambda u, n=None: True)
    monkeypatch.setattr(auth_module, "_mcp_auto_create_not_denied", lambda u, n: True)
    assert (await _run(f"{_PREFIX}", "POST", {"name": "acme/search"})) is True


# ---- Lifecycle routes: no tag named, but a target condition still governs ----


@pytest.mark.asyncio
async def test_updating_a_server_is_gated_by_a_target_condition(monkeypatch):
    """A PATCH names no tag, so a request condition is vacuous -- but a target condition must
    still apply. "Do not touch prod" that still permits renaming prod is not a restriction.
    """
    _configure(monkeypatch, target_condition="tags.env != 'prod'", permission=MANAGE)
    monkeypatch.setattr(
        auth_resources,
        "attrs_for_bulk",
        lambda rt, ids: {
            i: auth_resources.values_for_entity(
                rt, i, SimpleNamespace(tags={"env": "prod"}, aliases={})
            )
            for i in ids
        },
    )
    assert (await _run(f"{_PREFIX}/{_SERVER}", "PATCH", {"description": "x"})) is False


@pytest.mark.asyncio
async def test_deleting_a_server_is_gated_by_a_target_condition(monkeypatch):
    _configure(monkeypatch, target_condition="tags.env != 'prod'", permission=MANAGE)
    monkeypatch.setattr(
        auth_resources,
        "attrs_for_bulk",
        lambda rt, ids: {
            i: auth_resources.values_for_entity(
                rt, i, SimpleNamespace(tags={"env": "prod"}, aliases={})
            )
            for i in ids
        },
    )
    assert (await _run(f"{_PREFIX}/{_SERVER}", "DELETE", None)) is False


def _version_restricted(monkeypatch, children, target_condition="tags.keep != 'y'"):
    """A condition on the VERSION tier only, with the server itself unrestricted.

    `children` is what enumerating the server's versions returns: a tuple of ids, or None for
    "could not enumerate".
    """

    class Store:
        def get_user(self, username):
            return SimpleNamespace(id=1, username=username, is_admin=False)

        def is_workspace_admin(self, user_id, workspace):
            return False

        def list_mutation_conditions_for_user(
            self, user_id, workspace, resource_types, parents=None
        ):
            return [
                MutationConditionSpec(
                    "mcp_server_version", value_condition=None, target_condition=target_condition
                )
            ]

    monkeypatch.setattr(auth_module, "store", Store())
    monkeypatch.setattr(auth_module, "_get_mcp_server_permission", lambda n, u: MANAGE)
    monkeypatch.setattr(auth_module, "_mcp_server_version_action_allowed", lambda u, n, a: True)
    monkeypatch.setattr(auth_module, "get_anchor_workspace", lambda rt, rid: "ws")
    monkeypatch.setattr(auth_resources, "versions_of_mcp_server", lambda name: children)


@pytest.mark.asyncio
async def test_deleting_a_server_denies_when_a_child_version_fails_its_condition(monkeypatch):
    """The cascade's children are judged individually, and the child transition has to succeed
    for the parent's to. One failing version therefore fails the whole server delete.
    """
    _version_restricted(monkeypatch, ("acme%2Fsearch/1.0.0", "acme%2Fsearch/2.0.0"))
    monkeypatch.setattr(
        auth_resources,
        "attrs_for_bulk",
        lambda rt, ids: {
            i: auth_resources.values_for_entity(
                rt,
                i,
                # Only the second version is protected; one is enough to refuse.
                SimpleNamespace(tags={"keep": "y"} if i.endswith("2.0.0") else {}, aliases={}),
            )
            for i in ids
        },
    )
    assert (await _run(f"{_PREFIX}/{_SERVER}", "DELETE", None)) is False


@pytest.mark.asyncio
async def test_deleting_a_server_permits_when_every_child_passes(monkeypatch):
    """The point of enumerating rather than refusing outright: a condition on the child tier
    must not block a cascade whose children all satisfy it.
    """
    _version_restricted(monkeypatch, ("acme%2Fsearch/1.0.0", "acme%2Fsearch/2.0.0"))
    monkeypatch.setattr(
        auth_resources,
        "attrs_for_bulk",
        lambda rt, ids: {
            i: auth_resources.values_for_entity(
                rt, i, SimpleNamespace(tags={"keep": "n"}, aliases={})
            )
            for i in ids
        },
    )
    assert (await _run(f"{_PREFIX}/{_SERVER}", "DELETE", None)) is True


@pytest.mark.asyncio
async def test_deleting_a_childless_server_is_permitted(monkeypatch):
    """No children means nothing for the child condition to forbid. This must be distinct from
    "could not enumerate", which denies.
    """
    _version_restricted(monkeypatch, ())
    reads = []
    monkeypatch.setattr(
        auth_resources, "attrs_for_bulk", lambda rt, ids: reads.append((rt, list(ids))) or {}
    )
    assert (await _run(f"{_PREFIX}/{_SERVER}", "DELETE", None)) is True
    assert reads == [], f"nothing to read when there are no children; read {reads}"


@pytest.mark.asyncio
async def test_deleting_a_server_denies_when_children_cannot_be_enumerated(monkeypatch):
    """`None` from the enumerator means the child set could not be established -- too many to
    bound, or the search failed. The condition cannot be evaluated, so the cascade is refused
    rather than allowed through unjudged.
    """
    _version_restricted(monkeypatch, None)
    assert (await _run(f"{_PREFIX}/{_SERVER}", "DELETE", None)) is False


@pytest.mark.asyncio
async def test_an_unconditioned_cascade_never_enumerates(monkeypatch):
    """The enumeration is lazy, and that is what keeps this affordable: with no condition on the
    child tier the server delete must not list the server's versions at all.
    """
    _configure(monkeypatch, value_condition="tag_key != 'nope'", permission=MANAGE)
    calls = []
    monkeypatch.setattr(
        auth_resources, "versions_of_mcp_server", lambda name: calls.append(name) or ()
    )
    assert (await _run(f"{_PREFIX}/{_SERVER}", "DELETE", None)) is True
    assert calls == [], f"enumerated children with no child condition configured; {calls}"


@pytest.mark.asyncio
async def test_deleting_a_server_is_unaffected_when_no_version_condition_exists(monkeypatch):
    """The conservative refusal must not cost anything when nothing is configured for the child
    tier -- otherwise every cascade delete would break the moment conditions were enabled.
    """
    _configure(monkeypatch, value_condition="tag_key != 'nope'", permission=MANAGE)
    assert (await _run(f"{_PREFIX}/{_SERVER}", "DELETE", None)) is True


@pytest.mark.parametrize("method", ["PATCH", "DELETE"])
def test_mutating_a_version_itself_is_gated_by_its_target_condition(monkeypatch, method):
    """`versions/<version>` PATCH and DELETE name no tag, but they mutate that version, so its
    target condition governs -- and it must be read against the VERSION's id, not the server's.
    """

    async def go():
        _version_configured(monkeypatch, target_condition="tags.keep != 'y'")
        seen = []

        def attrs_for_bulk(rt, ids):
            seen.append((rt, list(ids)))
            return {
                i: auth_resources.values_for_entity(
                    rt, i, SimpleNamespace(tags={"keep": "y"}, aliases={})
                )
                for i in ids
            }

        monkeypatch.setattr(auth_resources, "attrs_for_bulk", attrs_for_bulk)
        allowed = await _run(f"{_PREFIX}/{_SERVER}/versions/1.2.0", method, None)
        assert allowed is False
        assert seen == [("mcp_server_version", ["acme%2Fsearch/1.2.0"])], (
            f"the version's own id must be read; got {seen}"
        )

    asyncio.run(go())
