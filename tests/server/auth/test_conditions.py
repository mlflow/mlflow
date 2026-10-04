# Unit tests for ``mlflow.server.auth.conditions`` -- the pure half of
# condition-based access control.
#
# No store, no request, no Flask. These tests are the security core: every other
# phase trusts that a parsed condition means what it says here.

import pytest

from mlflow.exceptions import MlflowException
from mlflow.server.auth.conditions import (
    ALIAS_OWNING_RESOURCE_TYPES,
    ALLOWED_COMPARATORS,
    MAX_CLAUSES,
    NAMESPACE_REQUEST,
    NAMESPACE_RESOURCE,
    PARENT_RESOURCE_TYPES,
    PARENTLESS_RESOURCE_TYPES,
    REQUEST_IDENTIFIERS,
    REQUEST_VALUES_SHAPES,
    RESOURCE_PREFIXES,
    RESOURCE_VALUES_SHAPES,
    SUPPORTED_RESOURCE_TYPES,
    Clause,
    ConditionContext,
    ConditionScope,
    RegisteredModelRequestValues,
    RegisteredModelResourceValues,
    RunRequestValues,
    RunResourceValues,
    combine,
    condition_load_types,
    context_for,
    evaluate_request,
    evaluate_resource,
    needs_resource_values,
    parse_condition,
    request_values_shape,
    resource_values_shape,
    validate_condition,
    validate_condition_parent_scope,
)

# ---- Parsing: request namespace --------------------------------------------


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
        filter_ids_by_clauses=lambda *a, **k: None,
        any_child_failing_clauses=lambda *a, **k: None,
    )
    monkeypatch.setattr(auth_module, "_get_tracking_store", lambda: declining)
    monkeypatch.setattr(auth_module, "_get_model_registry_store", lambda: declining, raising=False)


@pytest.mark.parametrize(
    "filter_string",
    [
        "tag_key = 'lifecycle'",
        "tag_key != 'lifecycle'",
        "tag_value = 'dev'",
        "tag_key LIKE 'team-%'",
        "tag_key ILIKE 'TEAM-%'",
        "tag_key IN ('a', 'b')",
        "tag_key NOT IN ('a', 'b')",
        "tag_key != 'lifecycle' AND tag_value != 'prod'",
        # `alias` is a SQL keyword, so sqlparse does not form a Comparison for it.
        # All four shapes must still parse -- an admin will write the unquoted form.
        "alias = 'champion'",
        "alias != 'champion'",
        "alias LIKE 'dev-%'",
        "alias IN ('a')",
        "alias NOT IN ('champion', 'prod')",
        "`alias` = 'champion'",
        "alias = 'x' AND tag_key = 'y'",
    ],
)
def test_parse_request_accepts(filter_string):
    assert parse_condition(filter_string, NAMESPACE_REQUEST)


@pytest.mark.parametrize(
    "filter_string",
    [
        "tags.lifecycle = 'dev'",
        "tags.lifecycle != 'prod'",
        "tags.`my.dotted.key` = 'true'",
        "aliases.champion = '3'",
        "tags.a = '1' AND aliases.b = '2'",
        "tags.x IN ('a', 'b')",
        "tags.x NOT IN ('a', 'b')",
        "tags.x LIKE 'y%'",
    ],
)
def test_parse_resource_accepts(filter_string):
    assert parse_condition(filter_string, NAMESPACE_RESOURCE)


@pytest.mark.parametrize("empty", [None, "", "   "])
@pytest.mark.parametrize("namespace", [NAMESPACE_REQUEST, NAMESPACE_RESOURCE])
def test_parse_empty_is_unconstrained(empty, namespace):
    # "No condition" and "a condition constraining nothing" are the same thing.
    assert parse_condition(empty, namespace) == ()


# ---- Parsing: rejections ---------------------------------------------------


@pytest.mark.parametrize(
    ("filter_string", "namespace", "expected"),
    [
        # No OR. A condition is a restriction; OR-ing restrictions weakens them.
        ("tag_key = 'a' OR tag_key = 'b'", NAMESPACE_REQUEST, "OR is not supported"),
        ("alias = 'a' OR alias = 'b'", NAMESPACE_REQUEST, "OR is not supported"),
        ("tags.a = '1' OR tags.b = '2'", NAMESPACE_RESOURCE, "OR is not supported"),
        # Cross-namespace identifiers, in both directions. Each must name the other
        # namespace explicitly -- an admin who mixes them has written a restriction
        # that would otherwise never fire.
        ("tags.x = 'a'", NAMESPACE_REQUEST, "resource-condition identifier"),
        ("aliases.x = 'a'", NAMESPACE_REQUEST, "resource-condition identifier"),
        ("tag_key = 'a'", NAMESPACE_RESOURCE, "request-condition identifier"),
        ("tag_value = 'a'", NAMESPACE_RESOURCE, "request-condition identifier"),
        ("alias = 'a'", NAMESPACE_RESOURCE, "request-condition identifier"),
        # Unknown identifiers.
        ("bogus = 'a'", NAMESPACE_REQUEST, "Invalid request-condition identifier"),
        ("bogus.x = 'a'", NAMESPACE_RESOURCE, "Invalid resource-condition identifier"),
        # A resource identifier needs a key.
        ("tags. = 'a'", NAMESPACE_RESOURCE, "missing a key"),
        # Malformed.
        ("garbage", NAMESPACE_REQUEST, "Invalid clause"),
        ("tag_key = unquoted", NAMESPACE_REQUEST, "quoted"),
    ],
)
def test_parse_rejects(filter_string, namespace, expected):
    with pytest.raises(MlflowException, match=expected):
        parse_condition(filter_string, namespace)


@pytest.mark.parametrize("comparator", [">", "<", ">=", "<="])
def test_parse_rejects_ordering_comparators(comparator):
    """Every condition value is a string, so ordering would compare
    lexicographically and mislead an admin into thinking they expressed a range.
    """
    with pytest.raises(MlflowException, match="not supported in conditions"):
        parse_condition(f"tag_key {comparator} 'a'", NAMESPACE_REQUEST)


def test_allowed_comparators_are_exclusion_capable():
    # ``!=`` and ``NOT IN`` are what let the RFC omit a separate deny form.
    assert {"!=", "NOT IN"} <= ALLOWED_COMPARATORS


def test_parse_rejects_too_many_clauses():
    over = " AND ".join(f"tag_key != 'k{i}'" for i in range(MAX_CLAUSES + 1))
    with pytest.raises(MlflowException, match=f"exceeds the maximum of {MAX_CLAUSES}"):
        parse_condition(over, NAMESPACE_REQUEST)

    at_limit = " AND ".join(f"tag_key != 'k{i}'" for i in range(MAX_CLAUSES))
    assert len(parse_condition(at_limit, NAMESPACE_REQUEST)) == MAX_CLAUSES


def test_parse_rejects_unknown_namespace():
    with pytest.raises(MlflowException, match="Unknown condition namespace"):
        parse_condition("tag_key = 'a'", "nonsense")


# ---- D4: the reserved-tag split --------------------------------------------


@pytest.mark.parametrize(
    "filter_string",
    [
        "tag_key = 'mlflow.runName'",
        "tag_key != 'mlflow.source.name'",
        "tag_key IN ('ok', 'mlflow.runName')",
    ],
)
def test_request_condition_rejects_reserved_keys(filter_string):
    """D4: MLflow writes ``mlflow.*`` itself, so a request condition constraining
    them would let an admin break ordinary logging -- and it would fail in ways
    that look like MLflow bugs rather than policy.
    """
    with pytest.raises(MlflowException, match="reserved tag keys"):
        parse_condition(filter_string, NAMESPACE_REQUEST)


def test_resource_condition_also_rejects_reserved_keys():
    """D4 applies to BOTH namespaces, for one shared reason.

    A reserved tag is user-writable with an arbitrary value -- ``set_tag`` accepts
    ``mlflow.runName``, ``mlflow.user`` and ``mlflow.source.type`` unchallenged -- and the
    request side can never gate that write, because naming a reserved key there is
    rejected. So a resource condition on a reserved tag restricts nothing: the holder sets
    the tag to whatever makes the condition pass. ``tags.mlflow.user = 'alice'`` reads as
    "only alice's runs" and is forgeable by anyone who can set a tag.

    That is the same phantom-restriction hazard the request-side ban exists to prevent, so
    it is refused in the same place, at authoring, where the admin finds out immediately.
    """
    for filter_string in (
        "tags.`mlflow.prompt.is_prompt` = 'true'",
        "tags.`mlflow.prompt.is_prompt` != 'true'",
        "tags.`mlflow.runName` != 'secret'",
        "tags.`mlflow.user` = 'alice'",
    ):
        with pytest.raises(MlflowException, match="reserved tag keys"):
            parse_condition(filter_string, NAMESPACE_RESOURCE)


def test_resource_condition_permits_an_ordinary_key_that_merely_contains_mlflow():
    """The ban is a prefix test on the key, not a substring search: a user-owned
    ``mlflow_stage`` or ``team.mlflow.note`` is not MLflow-written and stays authorable.
    """
    assert parse_condition("tags.mlflow_stage = 'dev'", NAMESPACE_RESOURCE)[0].key == "mlflow_stage"
    assert parse_condition("tags.`team.mlflow.note` = 'x'", NAMESPACE_RESOURCE)[0].key == (
        "team.mlflow.note"
    )


# ---- Evaluation: request, and the D13/D20 vacuity rule ---------------------


def test_request_absent_identifier_is_vacuous():
    """D13/D20. Not leniency -- necessity. Routes carry different subsets of the
    namespace, so without this rule no single condition could apply to more than
    one route.
    """
    clauses = parse_condition("tag_key != 'lifecycle'", NAMESPACE_REQUEST)
    assert evaluate_request(clauses, RunRequestValues()) is True

    alias_clauses = parse_condition("alias LIKE 'dev-%'", NAMESPACE_REQUEST)
    assert evaluate_request(alias_clauses, RunRequestValues(tags=(("x", "1"),))) is True


def test_request_metrics_only_log_batch_is_allowed():
    """§7.1 case 5b, the case most likely to catch a real bug: a metrics-only
    ``LogBatch`` carries no tags, so a ``tag_key`` condition must not deny it.
    """
    clauses = parse_condition("tag_key != 'lifecycle'", NAMESPACE_REQUEST)
    assert evaluate_request(clauses, RunRequestValues()) is True


def test_request_delete_shape_gates_on_key_but_not_value():
    """D12 plus D13. A deletion names a key with no value.

    The ``tag_key`` clause still bites -- otherwise a governed key would be
    writable in one direction, which is the delete-side bypass of the RFC's own
    use case 3. The ``tag_value`` clause is vacuous, because there is no value to
    test; constraining what a delete removes is the resource condition's job.
    """
    delete = RunRequestValues(tags=(("lifecycle", None),))

    key_clauses = parse_condition("tag_key != 'lifecycle'", NAMESPACE_REQUEST)
    assert evaluate_request(key_clauses, delete) is False

    value_clauses = parse_condition("tag_value = 'dev'", NAMESPACE_REQUEST)
    assert evaluate_request(value_clauses, delete) is True


def test_request_batch_any_failure_denies():
    # A bulk request must not be a way around a restriction that holds for one.
    clauses = parse_condition("tag_key != 'lifecycle'", NAMESPACE_REQUEST)
    assert evaluate_request(clauses, RunRequestValues(tags=(("ok", "1"),))) is True
    assert (
        evaluate_request(clauses, RunRequestValues(tags=(("ok", "1"), ("lifecycle", "2")))) is False
    )


@pytest.mark.parametrize(
    ("filter_string", "values", "expected"),
    [
        ("tag_key = 'a'", RunRequestValues(tags=(("a", "1"),)), True),
        ("tag_key = 'a'", RunRequestValues(tags=(("b", "1"),)), False),
        ("tag_key IN ('a', 'b')", RunRequestValues(tags=(("b", "1"),)), True),
        ("tag_key NOT IN ('a', 'b')", RunRequestValues(tags=(("c", "1"),)), True),
        ("tag_key NOT IN ('a', 'b')", RunRequestValues(tags=(("a", "1"),)), False),
        ("tag_value = 'dev'", RunRequestValues(tags=(("x", "dev"),)), True),
        ("tag_value = 'dev'", RunRequestValues(tags=(("x", "prod"),)), False),
        ("tag_key LIKE 'team-%'", RunRequestValues(tags=(("team-a", "1"),)), True),
        ("tag_key LIKE 'team-%'", RunRequestValues(tags=(("other", "1"),)), False),
        ("alias = 'champion'", RegisteredModelRequestValues(aliases=("champion",)), True),
        ("alias != 'champion'", RegisteredModelRequestValues(aliases=("champion",)), False),
        ("alias LIKE 'dev-%'", RegisteredModelRequestValues(aliases=("dev-1",)), True),
        ("alias NOT IN ('champion',)", RegisteredModelRequestValues(aliases=("dev-1",)), True),
    ],
)
def test_request_truth_table(filter_string, values, expected):
    assert evaluate_request(parse_condition(filter_string, NAMESPACE_REQUEST), values) is expected


# ---- Evaluation: resource, and the opposite absence rule -------------------


def test_resource_absent_tag_fails():
    """D20, the inverse of the request side. A resource lacking the tag does not
    have the state the condition describes, so it does not satisfy it. This is
    MLflow's own search semantics (``lhs is None`` -> ``False``), so a resource
    condition selects exactly what the same filter string would in a search box.
    """
    clauses = parse_condition("tags.lifecycle = 'dev'", NAMESPACE_RESOURCE)
    assert evaluate_resource(clauses, RunResourceValues("r1")) is False


def test_resource_absence_footgun_is_documented_behaviour():
    """The surprise worth documenting: ``tags.lifecycle != 'prod'`` *denies* an
    untagged resource, because the tag is absent rather than not-'prod'. Errs in
    the strict direction, but admins do not expect it.
    """
    clauses = parse_condition("tags.lifecycle != 'prod'", NAMESPACE_RESOURCE)
    assert evaluate_resource(clauses, RunResourceValues("r1")) is False
    assert evaluate_resource(clauses, RunResourceValues("r1", tags={"lifecycle": "dev"})) is True


def test_absence_semantics_are_opposite_per_namespace():
    """The asymmetry is the design. Implementing both the same way opens a hole in
    one direction or the other: vacuous-on-absence for resources would let an
    untagged resource slip past every ``tags.*`` restriction, and fail-on-absence
    for requests would deny ordinary value-free writes.
    """
    assert (
        evaluate_request(
            parse_condition("tag_key = 'lifecycle'", NAMESPACE_REQUEST), RunRequestValues()
        )
        is True
    )
    assert (
        evaluate_resource(
            parse_condition("tags.lifecycle = 'dev'", NAMESPACE_RESOURCE), RunResourceValues("r")
        )
        is False
    )


def test_resource_absence_fails_for_every_comparator_with_no_carve_out():
    """Absence denies uniformly, including for ``!=``, and there is no reserved-key
    exception any more.

    The prompt marker used to be special-cased here so that absence read as "not a
    prompt". D4 now rejects a ``tags.mlflow.*`` clause at authoring in both namespaces,
    so no such clause can be stored and the carve-out had nothing left to apply to. The
    model/prompt distinction is carried by ``resource_type`` instead, since ``prompt`` is
    a first-class RBAC type.

    What remains is the one rule, which is also what makes deleting a governing tag lose
    access rather than escape the restriction.
    """
    for filter_string in (
        "tags.lifecycle = 'dev'",
        "tags.lifecycle != 'prod'",
        "tags.lifecycle LIKE 'd%'",
        "tags.lifecycle IN ('dev', 'staging')",
        "tags.lifecycle NOT IN ('prod',)",
    ):
        clauses = parse_condition(filter_string, NAMESPACE_RESOURCE)
        assert evaluate_resource(clauses, RunResourceValues("m")) is False, filter_string

    # And a reserved key cannot reach evaluation at all.
    with pytest.raises(MlflowException, match="reserved tag keys"):
        parse_condition("tags.`mlflow.prompt.is_prompt` != 'true'", NAMESPACE_RESOURCE)


@pytest.mark.parametrize(
    ("filter_string", "values", "expected"),
    [
        ("tags.a = '1'", RunResourceValues("r", tags={"a": "1"}), True),
        ("tags.a = '1'", RunResourceValues("r", tags={"a": "2"}), False),
        ("tags.a IN ('1', '2')", RunResourceValues("r", tags={"a": "2"}), True),
        ("tags.a NOT IN ('1', '2')", RunResourceValues("r", tags={"a": "3"}), True),
        ("tags.a LIKE 'pre%'", RunResourceValues("r", tags={"a": "prefix"}), True),
        (
            "aliases.champion = '3'",
            RegisteredModelResourceValues("r", aliases={"champion": "3"}),
            True,
        ),
        (
            "aliases.champion = '3'",
            RegisteredModelResourceValues("r", aliases={"champion": "4"}),
            False,
        ),
        ("aliases.champion = '3'", RunResourceValues("r"), False),
        (
            "tags.a = '1' AND tags.b = '2'",
            RunResourceValues("r", tags={"a": "1", "b": "2"}),
            True,
        ),
        (
            "tags.a = '1' AND tags.b = '2'",
            RunResourceValues("r", tags={"a": "1"}),
            False,
        ),
    ],
)
def test_resource_truth_table(filter_string, values, expected):
    assert evaluate_resource(parse_condition(filter_string, NAMESPACE_RESOURCE), values) is expected


def test_resource_aliases_empty_for_version_types_denies_alias_clause():
    """D18 consequence: a version's ``ResourceValues`` never carries aliases, since
    aliases are owned by the registry entry. An alias clause therefore cannot be
    satisfied by a version -- which is correct, because the alias condition is
    keyed on the owning type.
    """
    clauses = parse_condition("aliases.champion = '3'", NAMESPACE_RESOURCE)
    version = RunResourceValues("m/3", tags={"a": "1"})
    assert evaluate_resource(clauses, version) is False


# ---- Combination -----------------------------------------------------------


def test_combine_is_plain_and():
    assert combine([]) is True
    assert combine([True, True]) is True
    assert combine([True, False]) is False


def test_permissive_condition_never_lifts_another_restriction():
    """The property that makes conditions safe to reason about: adding a role
    cannot widen what another role's condition restricts. Conditions only ever
    subtract, so there is nothing to resolve between two of them.
    """
    restrictive = evaluate_request(
        parse_condition("tag_key != 'lifecycle'", NAMESPACE_REQUEST),
        RunRequestValues(tags=(("lifecycle", "x"),)),
    )
    permissive = evaluate_request(
        parse_condition("tag_key != 'unrelated'", NAMESPACE_REQUEST),
        RunRequestValues(tags=(("lifecycle", "x"),)),
    )
    assert restrictive is False
    assert permissive is True
    assert combine([restrictive, permissive]) is False
    # And an absent condition (no clauses -> vacuously True) likewise.
    assert combine([restrictive, True]) is False


def test_contradictory_conditions_fail_closed():
    values = RunRequestValues(tags=(("lifecycle", "x"),))
    a = evaluate_request(parse_condition("tag_key = 'lifecycle'", NAMESPACE_REQUEST), values)
    b = evaluate_request(parse_condition("tag_key != 'lifecycle'", NAMESPACE_REQUEST), values)
    assert combine([a, b]) is False


# ---- Framework types -------------------------------------------------------


def test_condition_load_types_dedups():
    contexts = [
        ConditionContext("run", ConditionScope.MUTATE, RunRequestValues()),
        ConditionContext("run", ConditionScope.CREATE, RunRequestValues()),
        ConditionContext("trace", ConditionScope.MUTATE, RunRequestValues()),
    ]
    assert condition_load_types(contexts) == ("run", "trace")


def test_condition_load_types_empty():
    assert condition_load_types([]) == ()


def test_context_for_treats_wildcard_id_as_no_resource():
    """Sub-resource grants are wildcard-only grain, so a child requirement's id is
    literally ``"*"``. Passing that through would send the framework off to fetch a
    resource named ``*``.
    """
    assert (
        context_for(
            "logged_model", "*", ConditionScope.MUTATE, parent_resource_id="e1"
        ).resource_ids
        == ()
    )
    assert (
        context_for("run", None, ConditionScope.CREATE, parent_resource_id="e1").resource_ids == ()
    )
    assert context_for(
        "run", "r1", ConditionScope.MUTATE, parent_resource_id="e1"
    ).resource_ids == ("r1",)


def test_context_for_mirrors_request_values():
    values = RunRequestValues(tags=(("a", "1"),))
    context = context_for("run", "r1", ConditionScope.MUTATE, values, parent_resource_id="e1")
    assert context.resource_type == "run"
    assert context.scope is ConditionScope.MUTATE
    assert context.request == values


def test_needs_resource_values_short_circuits():
    # Two common ways to answer no, and each avoids reading a resource at all.
    create = context_for("run", None, ConditionScope.CREATE, parent_resource_id="e1")
    mutate = context_for("run", "r1", ConditionScope.MUTATE, parent_resource_id="e1")

    # A create has no prior state.
    assert needs_resource_values([create], {"run"}) is False
    # Configured, but no role has a target condition on this type.
    assert needs_resource_values([mutate], set()) is False
    # Both present -> the case that reads.
    assert needs_resource_values([mutate], {"run"}) is True

    # A MUTATE context naming NO resource must still reach the gate, even though there is
    # nothing to fetch. The gate refuses it (D21): an operation that cannot say which
    # resources it will change cannot be checked against a condition on them. Answering
    # False here would return *allow* instead, making a predicate-mode bulk delete a way
    # around every resource condition -- so this deliberately does not short-circuit.
    assert (
        needs_resource_values(
            [context_for("run", "*", ConditionScope.MUTATE, parent_resource_id="e1")], {"run"}
        )
        is True
    )
    assert (
        needs_resource_values(
            [context_for("run", None, ConditionScope.MUTATE, parent_resource_id="e1")], {"run"}
        )
        is True
    )


def test_no_none_scope():
    """A type no condition could govern simply has no context, which is cheaper and
    less error-prone than a scope that must be remembered and checked.
    """
    assert {s.name for s in ConditionScope} == {"CREATE", "MUTATE"}


# ---- Write-time validation hook --------------------------------------------


def test_validate_condition_rejects_malformed():
    """The store calls this so a malformed condition can never be persisted. A
    condition that failed to parse at evaluation time would have to either fail
    open (unsafe) or deny every mutation (an outage).
    """
    with pytest.raises(MlflowException, match="OR is not supported"):
        validate_condition("tag_key = 'a' OR tag_key = 'b'", NAMESPACE_REQUEST, "run")
    validate_condition("tag_key != 'a'", NAMESPACE_REQUEST, "run")
    validate_condition(None, NAMESPACE_RESOURCE, "run")


@pytest.mark.parametrize(
    ("resource_type", "condition", "namespace"),
    [
        ("run", "alias = 'champion'", NAMESPACE_REQUEST),
        ("experiment", "alias LIKE 'dev-%'", NAMESPACE_REQUEST),
        ("trace", "aliases.champion = '3'", NAMESPACE_RESOURCE),
        ("logged_model", "aliases.x = '1'", NAMESPACE_RESOURCE),
    ],
)
def test_validate_condition_rejects_alias_on_a_type_without_aliases(
    resource_type, condition, namespace
):
    """An alias clause parses anywhere, but only two types carry aliases -- and the
    two namespaces then fail in opposite directions. A request clause would be
    vacuous under D20, so the restriction silently never fires and the admin
    believes a protection is in force that is not. A resource clause would find
    nothing, fail, and deny every mutation of the type. Refuse both on the way in.
    """
    with pytest.raises(MlflowException, match="does not carry aliases"):
        validate_condition(condition, namespace, resource_type)


@pytest.mark.parametrize(
    ("resource_type", "owner"),
    [("registered_model_version", "registered_model"), ("prompt_version", "prompt")],
)
def test_validate_condition_points_a_version_alias_at_its_registry_entry(resource_type, owner):
    """A version is the one case an admin plausibly expects to work, since the route
    that sets an alias names a version (D18). The error has to say where the
    condition belongs, not merely that it is wrong.
    """
    with pytest.raises(MlflowException, match=f"condition the '{owner}' resource type instead"):
        validate_condition("alias = 'champion'", NAMESPACE_REQUEST, resource_type)
    with pytest.raises(MlflowException, match=f"condition the '{owner}' resource type instead"):
        validate_condition("aliases.champion = '3'", NAMESPACE_RESOURCE, resource_type)


@pytest.mark.parametrize("resource_type", sorted(ALIAS_OWNING_RESOURCE_TYPES))
def test_validate_condition_allows_alias_on_the_types_that_own_one(resource_type):
    validate_condition("alias = 'champion'", NAMESPACE_REQUEST, resource_type)
    validate_condition("aliases.champion = '3'", NAMESPACE_RESOURCE, resource_type)


@pytest.mark.parametrize("resource_type", sorted(SUPPORTED_RESOURCE_TYPES))
def test_validate_condition_allows_tags_on_every_supported_type(resource_type):
    """Only aliases need the cross-check: every supported type carries tags, so a
    tag condition must never be refused on type grounds.
    """
    validate_condition("tag_key != 'lifecycle'", NAMESPACE_REQUEST, resource_type)
    validate_condition("tags.lifecycle = 'dev'", NAMESPACE_RESOURCE, resource_type)


def test_validate_condition_requires_a_resource_type():
    """The cross-check is not optional. A caller that could omit the resource type
    would persist a condition that silently never fires, which is the failure this
    validation exists to prevent -- so there is deliberately no default.
    """
    with pytest.raises(TypeError, match="resource_type"):
        validate_condition("alias = 'champion'", NAMESPACE_REQUEST)


# ---- Parent scope ----------------------------------------------------------


def test_every_supported_type_declares_whether_it_has_a_parent():
    """The map must be total over the supported types.

    A type missing from it is indistinguishable from a parentless one, so a child
    type accidentally omitted would silently accept no parent scope -- the admin
    would be told their scoped condition is invalid for a type that should support
    it. Completeness is the guard.
    """
    classified = set(PARENT_RESOURCE_TYPES) | set(PARENTLESS_RESOURCE_TYPES)
    assert classified == set(SUPPORTED_RESOURCE_TYPES)
    assert not (set(PARENT_RESOURCE_TYPES) & set(PARENTLESS_RESOURCE_TYPES))


def test_parent_map_matches_the_rfc_table():
    """Pinned literally rather than derived. The RFC publishes this table as launch
    scope, so a change here is a change to the published design and should have to
    edit a test that says so.
    """
    assert PARENT_RESOURCE_TYPES == {
        "run": "experiment",
        "trace": "experiment",
        "logged_model": "experiment",
        "registered_model_version": "registered_model",
        "prompt_version": "prompt",
        "mcp_server_version": "mcp_server",
    }


@pytest.mark.parametrize(
    ("resource_type", "parent"),
    [
        ("registered_model_version", "registered_model"),
        ("prompt_version", "prompt"),
        ("mcp_server_version", "mcp_server"),
    ],
)
def test_every_version_type_parents_to_its_registry_entry(resource_type, parent):
    """Mirrors the alias rule: a version's aliases live on its registry entry, and so
    does its parent scope. The two encode the same containment.
    """
    assert PARENT_RESOURCE_TYPES[resource_type] == parent
    assert parent in ALIAS_OWNING_RESOURCE_TYPES


@pytest.mark.parametrize("resource_type", sorted(PARENTLESS_RESOURCE_TYPES))
def test_parent_scope_is_rejected_for_a_parentless_type(resource_type):
    """An experiment has no direct parent a condition could scope to. Accepting one
    would store a filter that can never match, which is the same phantom-restriction
    failure `validate_condition_resource_type` exists to prevent.
    """
    with pytest.raises(MlflowException, match="no direct parent"):
        validate_condition_parent_scope(resource_type, "workspace", "ws-1")


def test_parent_scope_rejects_a_parent_type_that_is_not_the_declared_one():
    """A run's parent is an experiment. Scoping it to a registered model would be
    accepted by any check that only asked "is this a supported type?", so the check
    is against the *declared* parent of this child, not the type vocabulary.
    """
    with pytest.raises(MlflowException, match="parent of 'run' is 'experiment'"):
        validate_condition_parent_scope("run", "registered_model", "m-1")


@pytest.mark.parametrize(
    ("parent_type", "parent_id"),
    [("experiment", None), (None, "123")],
)
def test_parent_scope_must_be_a_complete_pair(parent_type, parent_id):
    """Half a pair is ambiguous: a type with no ID names every parent of that type,
    and an ID with no type names nothing. Both readings are restrictions the admin
    did not write, so neither is inferred.
    """
    with pytest.raises(MlflowException, match="both.*or neither"):
        validate_condition_parent_scope("run", parent_type, parent_id)


def test_unscoped_is_accepted_for_every_supported_type():
    """Both null is the unscoped case -- the only shape available before parent scope
    existed, so it must stay valid for every type including the parentless ones.
    """
    for resource_type in sorted(SUPPORTED_RESOURCE_TYPES):
        validate_condition_parent_scope(resource_type, None, None)


@pytest.mark.parametrize("blank", ["", "   "])
def test_parent_scope_rejects_a_blank_parent_id(blank):
    """An empty ID is not a wildcard. Stored, it would match no parent while reading
    as a scope, so it is refused rather than normalised to unscoped -- the admin
    asked for a narrowing and must not silently get a broadening.
    """
    with pytest.raises(MlflowException, match="parent resource ID"):
        validate_condition_parent_scope("run", "experiment", blank)


@pytest.mark.parametrize(
    ("resource_type", "parent_type"),
    sorted(PARENT_RESOURCE_TYPES.items()),
)
def test_each_child_type_accepts_its_own_parent(resource_type, parent_type):
    validate_condition_parent_scope(resource_type, parent_type, "parent-1")


def test_context_for_refuses_a_child_type_without_a_parent():
    """A child-type context with no parent is a wiring bug, and must raise.

    This is the one place the fail-open direction is reachable: a parent-scoped
    condition is selected by matching the target's resolved parent, so a validator
    that omits it produces a context no scoped condition matches. The admin's
    restriction silently stops biting -- and unlike a denial, nothing surfaces.

    Raising mirrors the request-values shape check: a mis-wired validator fails in
    the tests that exercise its route rather than in production, and fails closed if
    one slips through.
    """
    for resource_type in sorted(PARENT_RESOURCE_TYPES):
        with pytest.raises(MlflowException, match="no parent was supplied"):
            context_for(resource_type, "child-1", ConditionScope.MUTATE)


def test_context_for_accepts_a_child_type_with_a_parent():
    for resource_type, parent_type in sorted(PARENT_RESOURCE_TYPES.items()):
        context = context_for(
            resource_type, "child-1", ConditionScope.MUTATE, parent_resource_id="parent-1"
        )
        assert context.parent_resource_id == "parent-1"
        assert PARENT_RESOURCE_TYPES[resource_type] == parent_type


@pytest.mark.parametrize("resource_type", sorted(PARENTLESS_RESOURCE_TYPES))
def test_context_for_needs_no_parent_for_a_parentless_type(resource_type):
    assert context_for(resource_type, "r-1", ConditionScope.MUTATE).parent_resource_id is None


def test_context_for_allows_an_unresolved_parent_only_when_explicit():
    """A cascade tier names no specific child, and its parent is the resource being
    cascaded -- so the parent is always known there. The escape hatch exists for the
    reverse case: an enumeration whose parent genuinely is not a conditionable
    resource. It has to be asked for, so that forgetting to pass a parent cannot
    silently take it.
    """
    context = context_for(
        "run",
        None,
        ConditionScope.MUTATE,
        parent_resource_id=None,
        allow_unscoped_parent=True,
    )
    assert context.parent_resource_id is None


def test_clause_describe_round_trips_readably():
    (clause,) = parse_condition("tags.lifecycle != 'prod'", NAMESPACE_RESOURCE)
    assert clause.describe() == "tags.lifecycle != 'prod'"
    (in_clause,) = parse_condition("tag_key IN ('a', 'b')", NAMESPACE_REQUEST)
    assert in_clause.describe() == "tag_key IN ('a', 'b')"


# ---- The validator wiring contract -----------------------------------------


@pytest.mark.parametrize(
    "resource_type", ["run", "experiment", "trace", "logged_model", "registered_model_version"]
)
def test_context_rejects_aliases_for_a_type_that_owns_none(resource_type):
    """`RequestValues` is one shape shared by every type, so it cannot express that a run
    has no alias to set. A mis-wired validator must therefore fail here rather than
    populate a field that silently goes unread.

    A raise, not a denial: this is a wiring bug and should surface in the tests that
    exercise the route.
    """
    with pytest.raises(MlflowException, match="wiring error"):
        context_for(
            resource_type,
            "x",
            ConditionScope.MUTATE,
            RegisteredModelRequestValues(aliases=("champion",)),
        )


@pytest.mark.parametrize("resource_type", sorted(ALIAS_OWNING_RESOURCE_TYPES))
def test_context_allows_aliases_for_the_types_that_own_them(resource_type):
    """Built through the table, because each alias-owning type declares its own shape --
    a prompt is not a registered model even though both carry aliases.
    """
    shape = request_values_shape(resource_type)
    context = context_for(resource_type, "x", ConditionScope.MUTATE, shape(aliases=("champion",)))
    assert context.request.aliases == ("champion",)


@pytest.mark.parametrize("resource_type", sorted(SUPPORTED_RESOURCE_TYPES))
def test_context_allows_tags_for_every_supported_type(resource_type):
    """Every supported type carries tags, whichever shape it declares. Built through the
    table so a type whose shape changes does not silently stop being covered here.
    """
    shape = request_values_shape(resource_type)
    context = context_for(
        resource_type,
        "x",
        ConditionScope.MUTATE,
        shape(tags=(("k", "v"),)),
        # A child type must declare its parent, exactly as its validator does.
        parent_resource_id="p1" if resource_type in PARENT_RESOURCE_TYPES else None,
    )
    assert context.request.tags == (("k", "v"),)


def test_every_supported_type_declares_both_shapes():
    """A type without an entry would raise KeyError on its first request. Asserted here so
    adding a supported type fails at once rather than on the route that first uses it.
    """
    assert set(REQUEST_VALUES_SHAPES) == set(SUPPORTED_RESOURCE_TYPES)
    assert set(RESOURCE_VALUES_SHAPES) == set(SUPPORTED_RESOURCE_TYPES)


def test_each_type_declares_a_distinct_shape():
    """One shape per type, not shared between types. Sharing would make a mis-wiring
    between two types that happen to carry the same fields undetectable.
    """
    assert len(set(REQUEST_VALUES_SHAPES.values())) == len(SUPPORTED_RESOURCE_TYPES)
    assert len(set(RESOURCE_VALUES_SHAPES.values())) == len(SUPPORTED_RESOURCE_TYPES)


@pytest.mark.parametrize("resource_type", sorted(SUPPORTED_RESOURCE_TYPES))
def test_only_alias_owning_types_declare_an_alias_field(resource_type):
    """The shape is the contract, so it must agree with ALIAS_OWNING_RESOURCE_TYPES (D18)
    -- otherwise a type could carry a field its conditions can never name, or be unable to
    carry one they can.
    """
    owns = resource_type in ALIAS_OWNING_RESOURCE_TYPES
    assert ("aliases" in request_values_shape(resource_type)._fields) is owns
    assert ("aliases" in resource_values_shape(resource_type)._fields) is owns


def test_context_rejects_an_unsupported_resource_type():
    """The same check the store applies on the way in, applied again at the point a
    validator declares a type -- a typo there would otherwise load conditions for a type
    no condition can exist for, and pass vacuously.
    """
    with pytest.raises(MlflowException, match="not supported for resource type"):
        context_for("assessment", "x", ConditionScope.MUTATE)


def test_clause_is_hashable_and_comparable():
    # ``Clause`` is a NamedTuple so loaded conditions can be cached and compared.
    a = Clause("tag_key", None, "=", "x")
    b = Clause("tag_key", None, "=", "x")
    assert a == b
    assert len({a, b}) == 1


# ---- Vocabulary/projection consistency -------------------------------------
#
# Both projections fall back to ``None`` for an identifier they do not recognise, and the
# two namespaces then read that fallback in opposite directions: vacuous on the request
# side, failing on the resource side. So an identifier added to the parser's vocabulary
# but not to its projection is silently FAIL-OPEN for requests -- the condition parses,
# stores, and then never restricts anything.
#
# These guards are what make the value shapes safe to extend: adding an identifier
# without projecting it fails here rather than in production.

#: identifier -> (values that populate it, a condition those values must VIOLATE)
_REQUEST_PROBES = {
    "tag_key": (RunRequestValues(tags=(("k", "v"),)), "tag_key != 'k'"),
    "tag_value": (RunRequestValues(tags=(("k", "v"),)), "tag_value != 'v'"),
    "alias": (RegisteredModelRequestValues(aliases=("a",)), "alias != 'a'"),
}

_RESOURCE_PROBES = {
    "tags": (RunResourceValues("r", tags={"k": "v"}), "tags.k != 'v'"),
    "aliases": (RegisteredModelResourceValues("r", aliases={"a": "1"}), "aliases.a != '1'"),
}


def test_every_request_identifier_has_a_probe():
    """If a new identifier is added, it must be enrolled below rather than skipped."""
    assert set(_REQUEST_PROBES) == set(REQUEST_IDENTIFIERS)


def test_every_resource_prefix_has_a_probe():
    assert set(_RESOURCE_PROBES) == set(RESOURCE_PREFIXES)


@pytest.mark.parametrize("identifier", sorted(_REQUEST_PROBES))
def test_every_request_identifier_is_projected(identifier):
    """A populated value must be able to violate a clause on its own identifier.

    Passing here would mean the projection never saw the value, so the clause was
    treated as vacuous -- the fail-open case.
    """
    values, condition = _REQUEST_PROBES[identifier]
    assert evaluate_request(parse_condition(condition, NAMESPACE_REQUEST), values) is False


@pytest.mark.parametrize("identifier", sorted(_RESOURCE_PROBES))
def test_every_resource_prefix_is_projected(identifier):
    values, condition = _RESOURCE_PROBES[identifier]
    assert evaluate_resource(parse_condition(condition, NAMESPACE_RESOURCE), values) is False


# ---- Reserved tags must not decide an unrelated clause ----------------------


def test_a_managed_tag_does_not_fail_an_unrelated_positive_clause():
    """MLflow writes its own `mlflow.*` tags on a create, and a request condition can never
    name one (authoring rejects it). So a reserved key must not participate in request
    evaluation either: with `all()` over every pair, `tag_key = 'team'` would otherwise be
    failed by the `mlflow.user` that MLflow itself added, denying a create the admin never
    restricted.

    Rejecting a condition that NAMES a reserved key is a different guarantee from exempting
    reserved values from evaluation; only the first was implemented.
    """
    clauses = parse_condition("tag_key = 'team'", NAMESPACE_REQUEST)
    values = RunRequestValues(tags=(("team", "ml"), ("mlflow.user", "alice")))
    assert evaluate_request(clauses, values) is True


def test_a_managed_tag_value_does_not_fail_an_unrelated_value_clause():
    """The same for `tag_value`, which the authoring check does not cover at all."""
    clauses = parse_condition("tag_value = 'ml'", NAMESPACE_REQUEST)
    values = RunRequestValues(tags=(("team", "ml"), ("mlflow.source.name", "train.py")))
    assert evaluate_request(clauses, values) is True


def test_a_user_tag_is_still_judged_alongside_a_managed_one():
    """The filter must not become a way through: a disallowed USER tag in the same request
    still denies, even when a managed tag is present.
    """
    clauses = parse_condition("tag_key != 'secret'", NAMESPACE_REQUEST)
    values = RunRequestValues(tags=(("secret", "x"), ("mlflow.user", "alice")))
    assert evaluate_request(clauses, values) is False
