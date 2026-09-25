"""Unit tests for ``mlflow.server.auth.conditions`` -- the pure half of
condition-based access control.

No store, no request, no Flask. These tests are the security core: every other
phase trusts that a parsed condition means what it says here.
"""

import pytest

from mlflow.exceptions import MlflowException
from mlflow.server.auth.conditions import (
    ALLOWED_COMPARATORS,
    MAX_CLAUSES,
    NAMESPACE_REQUEST,
    NAMESPACE_RESOURCE,
    Clause,
    ConditionContext,
    ConditionScope,
    RequestValues,
    ResourceValues,
    combine,
    condition_load_types,
    context_for,
    evaluate_request,
    evaluate_resource,
    needs_resource_values,
    parse_condition,
    validate_condition,
)

# ---- Parsing: request namespace --------------------------------------------


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
        "tags.`mlflow.prompt.is_prompt` = 'true'",
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
    """ "No condition" and "a condition constraining nothing" are the same thing."""
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
    """``!=`` and ``NOT IN`` are what let the RFC omit a separate deny form."""
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


def test_resource_condition_permits_reserved_keys():
    """D4, the other direction: testing current ``mlflow.*`` state is exactly how
    an admin expresses "only prompts" or "only runs named X".
    """
    clauses = parse_condition("tags.`mlflow.prompt.is_prompt` = 'true'", NAMESPACE_RESOURCE)
    assert clauses[0].key == "mlflow.prompt.is_prompt"


# ---- Evaluation: request, and the D13/D20 vacuity rule ---------------------


def test_request_absent_identifier_is_vacuous():
    """D13/D20. Not leniency -- necessity. Routes carry different subsets of the
    namespace, so without this rule no single condition could apply to more than
    one route.
    """
    clauses = parse_condition("tag_key != 'lifecycle'", NAMESPACE_REQUEST)
    assert evaluate_request(clauses, RequestValues()) is True

    alias_clauses = parse_condition("alias LIKE 'dev-%'", NAMESPACE_REQUEST)
    assert evaluate_request(alias_clauses, RequestValues(tags=(("x", "1"),))) is True


def test_request_metrics_only_log_batch_is_allowed():
    """§7.1 case 5b, the case most likely to catch a real bug: a metrics-only
    ``LogBatch`` carries no tags, so a ``tag_key`` condition must not deny it.
    """
    clauses = parse_condition("tag_key != 'lifecycle'", NAMESPACE_REQUEST)
    assert evaluate_request(clauses, RequestValues()) is True


def test_request_delete_shape_gates_on_key_but_not_value():
    """D12 plus D13. A deletion names a key with no value.

    The ``tag_key`` clause still bites -- otherwise a governed key would be
    writable in one direction, which is the delete-side bypass of the RFC's own
    use case 3. The ``tag_value`` clause is vacuous, because there is no value to
    test; constraining what a delete removes is the resource condition's job.
    """
    delete = RequestValues(tags=(("lifecycle", None),))

    key_clauses = parse_condition("tag_key != 'lifecycle'", NAMESPACE_REQUEST)
    assert evaluate_request(key_clauses, delete) is False

    value_clauses = parse_condition("tag_value = 'dev'", NAMESPACE_REQUEST)
    assert evaluate_request(value_clauses, delete) is True


def test_request_batch_any_failure_denies():
    """A bulk request must not be a way around a restriction that holds for one."""
    clauses = parse_condition("tag_key != 'lifecycle'", NAMESPACE_REQUEST)
    assert evaluate_request(clauses, RequestValues(tags=(("ok", "1"),))) is True
    assert evaluate_request(clauses, RequestValues(tags=(("ok", "1"), ("lifecycle", "2")))) is False


@pytest.mark.parametrize(
    ("filter_string", "values", "expected"),
    [
        ("tag_key = 'a'", RequestValues(tags=(("a", "1"),)), True),
        ("tag_key = 'a'", RequestValues(tags=(("b", "1"),)), False),
        ("tag_key IN ('a', 'b')", RequestValues(tags=(("b", "1"),)), True),
        ("tag_key NOT IN ('a', 'b')", RequestValues(tags=(("c", "1"),)), True),
        ("tag_key NOT IN ('a', 'b')", RequestValues(tags=(("a", "1"),)), False),
        ("tag_value = 'dev'", RequestValues(tags=(("x", "dev"),)), True),
        ("tag_value = 'dev'", RequestValues(tags=(("x", "prod"),)), False),
        ("tag_key LIKE 'team-%'", RequestValues(tags=(("team-a", "1"),)), True),
        ("tag_key LIKE 'team-%'", RequestValues(tags=(("other", "1"),)), False),
        ("alias = 'champion'", RequestValues(aliases=("champion",)), True),
        ("alias != 'champion'", RequestValues(aliases=("champion",)), False),
        ("alias LIKE 'dev-%'", RequestValues(aliases=("dev-1",)), True),
        ("alias NOT IN ('champion',)", RequestValues(aliases=("dev-1",)), True),
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
    assert evaluate_resource(clauses, ResourceValues("r1")) is False


def test_resource_absence_footgun_is_documented_behaviour():
    """The surprise worth documenting: ``tags.lifecycle != 'prod'`` *denies* an
    untagged resource, because the tag is absent rather than not-'prod'. Errs in
    the strict direction, but admins do not expect it.
    """
    clauses = parse_condition("tags.lifecycle != 'prod'", NAMESPACE_RESOURCE)
    assert evaluate_resource(clauses, ResourceValues("r1")) is False
    assert evaluate_resource(clauses, ResourceValues("r1", tags={"lifecycle": "dev"})) is True


def test_absence_semantics_are_opposite_per_namespace():
    """The asymmetry is the design. Implementing both the same way opens a hole in
    one direction or the other: vacuous-on-absence for resources would let an
    untagged resource slip past every ``tags.*`` restriction, and fail-on-absence
    for requests would deny ordinary value-free writes.
    """
    assert (
        evaluate_request(
            parse_condition("tag_key = 'lifecycle'", NAMESPACE_REQUEST), RequestValues()
        )
        is True
    )
    assert (
        evaluate_resource(
            parse_condition("tags.lifecycle = 'dev'", NAMESPACE_RESOURCE), ResourceValues("r")
        )
        is False
    )


def test_resource_prompt_marker_absence_means_not_a_prompt():
    """An ordinary registered model carries no ``mlflow.prompt.is_prompt`` tag, so
    absence means "not a prompt" rather than "unknown" -- which is why MLflow's own
    registered-model search special-cases it too.

    Without this, ``tags.mlflow.prompt.is_prompt != 'true'`` -- the natural way to
    write "models, not prompts" -- would deny every ordinary registered model.
    """
    not_prompt = parse_condition("tags.`mlflow.prompt.is_prompt` != 'true'", NAMESPACE_RESOURCE)
    assert evaluate_resource(not_prompt, ResourceValues("m")) is True
    assert (
        evaluate_resource(not_prompt, ResourceValues("m", tags={"mlflow.prompt.is_prompt": "true"}))
        is False
    )

    is_false = parse_condition("tags.`mlflow.prompt.is_prompt` = 'false'", NAMESPACE_RESOURCE)
    assert evaluate_resource(is_false, ResourceValues("m")) is True

    is_true = parse_condition("tags.`mlflow.prompt.is_prompt` = 'true'", NAMESPACE_RESOURCE)
    assert evaluate_resource(is_true, ResourceValues("m")) is False


@pytest.mark.parametrize(
    ("filter_string", "values", "expected"),
    [
        ("tags.a = '1'", ResourceValues("r", tags={"a": "1"}), True),
        ("tags.a = '1'", ResourceValues("r", tags={"a": "2"}), False),
        ("tags.a IN ('1', '2')", ResourceValues("r", tags={"a": "2"}), True),
        ("tags.a NOT IN ('1', '2')", ResourceValues("r", tags={"a": "3"}), True),
        ("tags.a LIKE 'pre%'", ResourceValues("r", tags={"a": "prefix"}), True),
        ("aliases.champion = '3'", ResourceValues("r", aliases={"champion": "3"}), True),
        ("aliases.champion = '3'", ResourceValues("r", aliases={"champion": "4"}), False),
        ("aliases.champion = '3'", ResourceValues("r"), False),
        (
            "tags.a = '1' AND tags.b = '2'",
            ResourceValues("r", tags={"a": "1", "b": "2"}),
            True,
        ),
        (
            "tags.a = '1' AND tags.b = '2'",
            ResourceValues("r", tags={"a": "1"}),
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
    version = ResourceValues("m/3", tags={"a": "1"})
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
        RequestValues(tags=(("lifecycle", "x"),)),
    )
    permissive = evaluate_request(
        parse_condition("tag_key != 'unrelated'", NAMESPACE_REQUEST),
        RequestValues(tags=(("lifecycle", "x"),)),
    )
    assert restrictive is False
    assert permissive is True
    assert combine([restrictive, permissive]) is False
    # And an absent condition (no clauses -> vacuously True) likewise.
    assert combine([restrictive, True]) is False


def test_contradictory_conditions_fail_closed():
    values = RequestValues(tags=(("lifecycle", "x"),))
    a = evaluate_request(parse_condition("tag_key = 'lifecycle'", NAMESPACE_REQUEST), values)
    b = evaluate_request(parse_condition("tag_key != 'lifecycle'", NAMESPACE_REQUEST), values)
    assert combine([a, b]) is False


# ---- Framework types -------------------------------------------------------


def test_condition_load_types_dedups():
    contexts = [
        ConditionContext("run", ConditionScope.MUTATE),
        ConditionContext("run", ConditionScope.CREATE),
        ConditionContext("trace", ConditionScope.MUTATE),
    ]
    assert condition_load_types(contexts) == ("run", "trace")


def test_condition_load_types_empty():
    assert condition_load_types([]) == ()


def test_context_for_treats_wildcard_id_as_no_resource():
    """Sub-resource grants are wildcard-only grain, so a child requirement's id is
    literally ``"*"``. Passing that through would send the framework off to fetch a
    resource named ``*``.
    """
    assert context_for("logged_model", "*", ConditionScope.MUTATE).resource_ids == ()
    assert context_for("run", None, ConditionScope.CREATE).resource_ids == ()
    assert context_for("run", "r1", ConditionScope.MUTATE).resource_ids == ("r1",)


def test_context_for_mirrors_request_values():
    values = RequestValues(tags=(("a", "1"),))
    context = context_for("run", "r1", ConditionScope.MUTATE, values)
    assert context.resource_type == "run"
    assert context.scope is ConditionScope.MUTATE
    assert context.request == values


def test_needs_resource_values_short_circuits():
    """Two common ways to answer no, and each avoids reading a resource at all."""
    create = context_for("run", None, ConditionScope.CREATE)
    mutate = context_for("run", "r1", ConditionScope.MUTATE)

    # A create has no prior state.
    assert needs_resource_values([create], {"run"}) is False
    # Configured, but no role has a target condition on this type.
    assert needs_resource_values([mutate], set()) is False
    # Both present -> the only case that reads.
    assert needs_resource_values([mutate], {"run"}) is True
    # A wildcard id means no identified resource to read.
    assert needs_resource_values([context_for("run", "*", ConditionScope.MUTATE)], {"run"}) is False


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
        validate_condition("tag_key = 'a' OR tag_key = 'b'", NAMESPACE_REQUEST)
    validate_condition("tag_key != 'a'", NAMESPACE_REQUEST)
    validate_condition(None, NAMESPACE_RESOURCE)


def test_clause_describe_round_trips_readably():
    (clause,) = parse_condition("tags.lifecycle != 'prod'", NAMESPACE_RESOURCE)
    assert clause.describe() == "tags.lifecycle != 'prod'"
    (in_clause,) = parse_condition("tag_key IN ('a', 'b')", NAMESPACE_REQUEST)
    assert in_clause.describe() == "tag_key IN ('a', 'b')"


def test_clause_is_hashable_and_comparable():
    """``Clause`` is a NamedTuple so loaded conditions can be cached and compared."""
    a = Clause("tag_key", None, "=", "x")
    b = Clause("tag_key", None, "=", "x")
    assert a == b
    assert len({a, b}) == 1
