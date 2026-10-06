"""What a 403 tells a caller when a mutation condition refused the request.

A condition denial used to say only that one happened. That is a dead end: the caller
holds a permission level that allows the operation, the thing that refused is a filter
string on a role they may not have written, and nothing in the response says which
filter or -- for a cascade -- which of a parent's children tripped it. Deleting an
experiment with three thousand runs reported "a condition refused" and left the caller
to guess which run.

So the denial names the clause, and for a target condition the resource that broke it.

Disclosing that leaks nothing. Conditions are consulted only AFTER a base grant has
passed, and every grant that permits a mutation also permits a read -- ``EDIT`` and
``MANAGE`` both carry ``can_read`` -- so a caller who reaches a condition check can
already fetch the state being quoted back at them. The condition rows themselves are
readable too: neither ``roles/list`` nor the condition listing is scoped to the caller.

Attribution is best-effort by construction. The store reports *which resource* failed,
which it can do over an unbounded population; naming the *clause* needs one more read of
that one resource, and a resource that vanished in between leaves the class of refusal
stated without the clause. That degrades to the old message rather than to a wrong one.
"""

import json
from types import SimpleNamespace

import pytest

from mlflow.server import auth as auth_module
from mlflow.server.auth import resources as auth_resources
from mlflow.server.auth.conditions import (
    ConditionContext,
    ConditionScope,
    RunRequestValues,
)
from mlflow.store.condition_pushdown import DECLINED

TAG_KEY = "lifecycle"


def _row(*, value_condition=None, target_condition=None, resource_type="run", pattern="*"):
    return SimpleNamespace(
        resource_type=resource_type,
        value_condition=value_condition,
        target_condition=target_condition,
        resource_pattern=pattern,
    )


@pytest.fixture
def gate(monkeypatch):
    """Run the condition gate with given rows, returning ``(allowed, message)``.

    The message comes from ``denial_message()`` rather than from an assertion on the
    detail directly, because the body a caller actually receives is the contract.
    """

    def run(contexts, rows, *, store_answer=DECLINED, values=None):
        class Store:
            def get_user(self, username):
                return SimpleNamespace(id=1, username=username, is_admin=False)

            def is_workspace_admin(self, user_id, workspace):
                return False

            def list_mutation_conditions_for_user(self, user_id, workspace, types, parents=None):
                return list(rows)

        monkeypatch.setattr(auth_module, "store", Store())
        fake = SimpleNamespace(find_failing_resource=lambda *a, **k: store_answer)
        monkeypatch.setattr(auth_module, "_get_tracking_store", lambda: fake)
        monkeypatch.setattr(auth_module, "_get_model_registry_store", lambda: fake, raising=False)

        supplied = values or {}
        monkeypatch.setattr(
            auth_resources, "attrs_for", lambda t, i: supplied.get((t, i)), raising=False
        )
        monkeypatch.setattr(
            auth_resources,
            "attrs_for_bulk",
            lambda t, ids: {i: supplied.get((t, i)) for i in ids},
            raising=False,
        )
        auth_resources.clear_cache()
        allowed = auth_module.authorize_on_conditions("alice", "w", list(contexts))
        return allowed, auth_module.denial_message()

    return run


def _mutate(resource_type="run", ids=("r-1",), request=None, parent=None):
    return ConditionContext(
        resource_type=resource_type,
        scope=ConditionScope.MUTATE,
        request=request if request is not None else RunRequestValues(),
        resource_ids=tuple(ids),
        parent_resource_id=parent,
    )


class TestAValueConditionNamesTheIdentifier:
    """The request-side half, which is pure and so always attributable."""

    def test_the_refusing_identifier_is_named(self, gate):
        allowed, message = gate(
            [_mutate(request=RunRequestValues(tags=((TAG_KEY, "prod"),)))],
            [_row(value_condition="tag_value != 'prod'")],
        )
        assert allowed is False
        assert "tag_value" in message, message

    def test_the_identifier_is_the_one_that_failed_not_the_first(self, gate):
        """A conjunction must attribute to the clause that actually refused.

        Reporting clause one regardless would be worse than saying nothing: it points
        the caller at a filter that permitted their request.
        """
        allowed, message = gate(
            [_mutate(request=RunRequestValues(tags=(("team", "ml"),)))],
            [_row(value_condition="tag_value != 'nothing' AND tag_key != 'team'")],
        )
        assert allowed is False
        assert "tag_key" in message, message
        assert "tag_value" not in message, message

    def test_a_permitted_request_says_nothing(self, gate):
        allowed, message = gate(
            [_mutate(request=RunRequestValues(tags=((TAG_KEY, "dev"),)))],
            [_row(value_condition="tag_value != 'prod'")],
        )
        assert allowed is True
        assert message == "Permission denied", (
            "a permitted request must leave the generic message, since no 403 is sent "
            "and a stale detail would mislabel the next denial"
        )


class TestATargetConditionNamesTheClauseAndTheResource:
    """The resource-side half, where the store names the resource and a read names the clause."""

    def test_both_the_clause_and_the_resource_are_named(self, gate):
        allowed, message = gate(
            [_mutate(ids=("r-7",))],
            [_row(target_condition=f"tags.{TAG_KEY} = 'dev'")],
            store_answer="r-7",
            values={("run", "r-7"): SimpleNamespace(tags={TAG_KEY: "prod"}, aliases={})},
        )
        assert allowed is False
        assert f"tags.{TAG_KEY}" in message, message
        assert "r-7" in message, message

    def test_the_prefixed_identifier_is_quoted_as_written(self, gate):
        """``tags.lifecycle``, not ``lifecycle`` -- the form the admin typed.

        A bare key would be ambiguous between the two namespaces, and would not
        round-trip into the condition string the caller has to go and read.
        """
        _, message = gate(
            [_mutate(ids=("r-1",))],
            [_row(target_condition=f"tags.{TAG_KEY} = 'dev'")],
            store_answer="r-1",
            values={("run", "r-1"): SimpleNamespace(tags={}, aliases={})},
        )
        assert f"'tags.{TAG_KEY}'" in message, message

    def test_an_unattributable_denial_degrades_to_the_class_of_refusal(self, gate):
        """The resource the store named is gone by the time attribution reads it.

        The denial stands -- it was already decided -- and the message says a condition
        refused without inventing a clause.
        """
        allowed, message = gate(
            [_mutate(ids=("r-9",))],
            [_row(target_condition=f"tags.{TAG_KEY} = 'dev'")],
            store_answer="r-9",
            values={},
        )
        assert allowed is False
        assert "access condition" in message
        assert "tags." not in message, message

    def test_attribution_never_turns_a_denial_into_an_error(self, gate, monkeypatch):
        """A read that raises must not convert a clean 403 into a 500.

        Attribution is a courtesy on a path that has already decided to deny, so it is
        the one place here allowed to swallow an exception.
        """

        def _boom(_type, _id):
            raise RuntimeError("backend down")

        monkeypatch.setattr(auth_resources, "attrs_for", _boom, raising=False)
        allowed, message = gate(
            [_mutate(ids=("r-1",))],
            [_row(target_condition=f"tags.{TAG_KEY} = 'dev'")],
            store_answer="r-1",
        )
        assert allowed is False
        assert "access condition" in message


class TestACascadeNamesTheChildThatBlockedIt:
    """The case with no workaround before: the caller could not learn which child."""

    def test_the_offending_child_is_named(self, gate):
        allowed, message = gate(
            [
                ConditionContext(
                    resource_type="run",
                    scope=ConditionScope.MUTATE,
                    request=RunRequestValues(),
                    resource_ids=(),
                    parent_resource_id="exp-1",
                    resource_id_resolver=lambda: ["child-3"],
                )
            ],
            [_row(target_condition=f"tags.{TAG_KEY} = 'dev'")],
            store_answer="child-3",
            values={("run", "child-3"): SimpleNamespace(tags={TAG_KEY: "prod"}, aliases={})},
        )
        assert allowed is False
        assert "child-3" in message, (
            "a cascade denial must name the child that blocked it; the store already "
            "selects that id, so withholding it tells the caller nothing actionable"
        )
        assert f"tags.{TAG_KEY}" in message, message


class TestAVersionDenialIsAttributableToo:
    """A version's id is composite, and the two layers spell it differently.

    The authorization layer addresses a version as the single opaque string
    ``name/version`` with the name percent-encoded; the store is handed the decomposed
    parts, because that format is the auth layer's invention and a store that split on
    ``/`` would cut a reverse-DNS name in the wrong place. So the store ANSWERS in parts
    too -- and attribution reads the resource by auth-layer id. Without converting back,
    the lookup misses, attribution returns ``None``, and a denial that knows exactly
    which version refused reports only that something did.

    Both selectors are covered because both receive the store's key shape: the named
    path converts its ids on the way in, so its answer comes back decomposed as well.
    """

    ROW = [
        _row(target_condition=f"tags.{TAG_KEY} = 'dev'", resource_type="registered_model_version")
    ]

    def test_a_named_version_is_identified_in_the_denial(self, gate):
        allowed, message = gate(
            [_mutate(resource_type="registered_model_version", ids=("m-prod/2",))],
            self.ROW,
            store_answer=("m-prod", "2"),
            values={
                ("registered_model_version", "m-prod/2"): SimpleNamespace(
                    tags={TAG_KEY: "prod"}, aliases={}
                )
            },
        )
        assert allowed is False
        assert f"tags.{TAG_KEY}" in message, message
        assert "m-prod/2" in message, (
            f"the denial must name the version in the form the auth layer uses: {message!r}"
        )

    def test_a_cascaded_version_is_identified_in_the_denial(self, gate):
        allowed, message = gate(
            [
                ConditionContext(
                    resource_type="registered_model_version",
                    scope=ConditionScope.MUTATE,
                    request=RunRequestValues(),
                    resource_ids=(),
                    parent_resource_id="m-prod",
                    # A resolver must be present for the cascade branch to be taken at all;
                    # it is never called when the store answers, which is the point.
                    resource_id_resolver=lambda: ["m-prod/1", "m-prod/2"],
                )
            ],
            self.ROW,
            store_answer=("m-prod", "2"),
            values={
                ("registered_model_version", "m-prod/2"): SimpleNamespace(
                    tags={TAG_KEY: "prod"}, aliases={}
                )
            },
        )
        assert allowed is False
        assert "m-prod/2" in message, (
            f"a cascade denial must name the version that blocked it: {message!r}"
        )

    def test_a_name_containing_a_slash_round_trips(self, gate):
        """An MCP name is reverse-DNS, so it always contains ``/``.

        This is why the conversion must go through the same percent-encoding helper
        rather than joining with ``/``: a naive join produces an id that no lookup
        matches, silently costing the attribution rather than failing loudly.
        """
        allowed, message = gate(
            [
                ConditionContext(
                    resource_type="mcp_server_version",
                    scope=ConditionScope.MUTATE,
                    request=RunRequestValues(),
                    resource_ids=(),
                    parent_resource_id="demo/gateway",
                    resource_id_resolver=lambda: ["demo%2Fgateway/1.0.0"],
                )
            ],
            [
                _row(
                    target_condition=f"tags.{TAG_KEY} = 'dev'",
                    resource_type="mcp_server_version",
                )
            ],
            store_answer=("demo/gateway", "1.0.0"),
            values={
                ("mcp_server_version", "demo%2Fgateway/1.0.0"): SimpleNamespace(
                    tags={TAG_KEY: "prod"}, aliases={}
                )
            },
        )
        assert allowed is False
        assert "demo%2Fgateway/1.0.0" in message, (
            f"the name must be percent-encoded exactly as version_resource_id writes it: "
            f"{message!r}"
        )


class TestTheDetailDoesNotLeakBetweenRequests:
    """Same lifetime and same hazard as the resource memo it lives beside."""

    def test_clear_cache_drops_the_detail(self, gate):
        allowed, message = gate(
            [_mutate(request=RunRequestValues(tags=((TAG_KEY, "prod"),)))],
            [_row(value_condition="tag_value != 'prod'")],
        )
        assert allowed is False
        assert "tag_value" in message
        auth_resources.clear_cache()
        assert auth_resources.condition_denial_detail() is None
        assert auth_module.denial_message() == "Permission denied", (
            "left set, the next request's GRANT denial would be labelled with this "
            "request's condition clause"
        )

    def test_the_first_detail_wins(self, gate):
        """Authorization can refuse several things while evaluating a disjunction.

        The first refusal is the one the caller hit; a later branch being tried and also
        refused must not overwrite it.
        """
        auth_resources.clear_cache()
        auth_resources.note_condition_denial("the first reason")
        auth_resources.note_condition_denial("a later branch")
        assert auth_resources.condition_denial_detail() == "the first reason"

    def test_a_detail_without_a_denial_is_not_reported(self, gate):
        """``denial_message`` keys on the denial flag, not on the detail being present."""
        auth_resources.clear_cache()
        assert auth_module.denial_message() == "Permission denied"


class TestTheForbiddenBodyIsAParseableEnvelope:
    """A 403 the UI can actually read.

    The body carried the reason all along, but as a bare ``text/html`` string. The client's
    ``ErrorWrapper`` does ``JSON.parse`` on it, stores ``null`` on failure, and then
    ``getUserVisibleError()`` returns the literal string ``'INTERNAL_SERVER_ERROR'`` -- so a
    correct authorization decision reached the user as an internal server error, and every
    denial detail this module exists to produce was invisible outside devtools.

    ``error_code`` AND ``message`` must both be present: ``renderHttpError`` checks for both
    before using either, and falls back to the generic text if either is missing.
    """

    def _body(self):
        with auth_module.app.test_request_context("/"):
            return auth_module.make_forbidden_response()

    def test_the_body_is_json_not_html(self):
        res = self._body()
        assert res.status_code == 403
        assert res.mimetype == "application/json", (
            "a text/html body is what makes the client fall back to INTERNAL_SERVER_ERROR"
        )

    def test_the_envelope_carries_the_code_and_the_message(self):
        payload = json.loads(self._body().get_data(as_text=True))
        assert payload["error_code"] == "PERMISSION_DENIED"
        assert payload["message"] == "Permission denied"

    def test_a_condition_denial_detail_reaches_the_message_field(self):
        with auth_module.app.test_request_context("/"):
            auth_resources.note_condition_denial("'tags.a' on run 'r1' does not satisfy it")
            res = auth_module.make_forbidden_response()
        payload = json.loads(res.get_data(as_text=True))
        assert payload["error_code"] == "PERMISSION_DENIED"
        assert "condition" in payload["message"]
        assert "'tags.a' on run 'r1'" in payload["message"], (
            "the detail is the whole point of the envelope: it is what the user sees instead "
            "of a generic failure"
        )

    def test_the_status_still_reads_as_a_denial_to_a_text_matcher(self):
        """The 60-odd existing substring assertions, and any client grepping the body, keep
        working -- a JSON body still contains the phrase.
        """
        assert "Permission denied" in self._body().get_data(as_text=True)
