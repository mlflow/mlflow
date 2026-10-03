"""Condition-based access control: the pure half.

Two optional filters per ``(role, resource_type)``, layered on top of existing RBAC
grants, gating **create/mutation only** -- never reads:

- **request condition** (the RFC calls it *value*) constrains *what values* may be
  set. Its vocabulary is ``tag_key``, ``tag_value``, ``alias``, and it is evaluated
  against the request body. It applies on create.
- **resource condition** (the RFC calls it *target*) constrains *which existing
  resources* may be mutated. Its vocabulary is ``tags.<key>`` and
  ``aliases.<name>``, and it is evaluated against the resource's current state. It
  is vacuous on create, because there is no prior state to test.

The governing invariant is **grants add, conditions subtract**. Capability is still
the union of the user's roles' grants; conditions can only remove from that union.
Every applicable condition across *all* of a user's roles must pass. A condition
never confers access, is never consulted on a read, and an empty table reproduces
the pre-conditions behaviour exactly.

This module is deliberately pure: no store, no request, no Flask. It knows how to
parse a filter string and how to decide whether values satisfy it. Loading rows,
resolving resource attributes, and wiring into validators all live elsewhere
(``sqlalchemy_store``, ``resources.py``, ``__init__.py``) so that the security core
can be reviewed and tested in isolation.

Why not extend ``SearchUtils``: its ``parse_search_filter`` validates identifiers
against the *run* vocabulary, and these two namespaces are neither a subset nor a
superset of it. We reuse its tokenizer and comparison primitives -- which are the
fiddly, well-tested parts -- and own the vocabulary here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from enum import Enum, auto
from typing import NamedTuple

import sqlparse
from sqlparse.sql import Comparison, Parenthesis, Statement, TokenList
from sqlparse.tokens import Token as TokenType

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.search_utils import SearchUtils, _join_in_comparison_tokens

# ---------------------------------------------------------------------------
# Namespaces and vocabulary
# ---------------------------------------------------------------------------

#: Request-condition namespace: constrains the values a mutation may set.
NAMESPACE_REQUEST = "request"
#: Resource-condition namespace: constrains which resources may be mutated.
NAMESPACE_RESOURCE = "resource"

NAMESPACES = frozenset({NAMESPACE_REQUEST, NAMESPACE_RESOURCE})

#: Request identifiers. Flat names, because a request carries values, not state.
REQUEST_IDENTIFIER_TAG_KEY = "tag_key"
REQUEST_IDENTIFIER_TAG_VALUE = "tag_value"
REQUEST_IDENTIFIER_ALIAS = "alias"

REQUEST_IDENTIFIERS = frozenset({
    REQUEST_IDENTIFIER_TAG_KEY,
    REQUEST_IDENTIFIER_TAG_VALUE,
    REQUEST_IDENTIFIER_ALIAS,
})

#: Resource identifier prefixes. Keyed, because resource state is a map:
#: ``tags.lifecycle``, ``aliases.champion``.
RESOURCE_PREFIX_TAGS = "tags"
RESOURCE_PREFIX_ALIASES = "aliases"

RESOURCE_PREFIXES = frozenset({RESOURCE_PREFIX_TAGS, RESOURCE_PREFIX_ALIASES})

#: Comparators permitted in a condition. A deliberate subset of what
#: ``SearchUtils.get_comparison_func`` supports: every value here is a string, so
#: the numeric orderings (``>``, ``<``, ``>=``, ``<=``) would compare
#: lexicographically and mislead an admin into thinking they had expressed a range.
#: ``!=`` and ``NOT IN`` are what supply exclusion, which is why the RFC needs no
#: separate deny form.
ALLOWED_COMPARATORS = frozenset({"=", "!=", "LIKE", "ILIKE", "IN", "NOT IN"})

#: Per the RFC. A cap rather than a performance bound: a condition an admin cannot
#: read at a glance is one they cannot reason about, and every clause is an AND.
MAX_CLAUSES = 5

#: Per the RFC: how many condition objects a role may hold for one resource type.
#:
#: A storage bound, not an evaluation bound. Scope selects before combination, so a
#: request only evaluates the unscoped conditions plus those whose parent matches the
#: target's resolved parent -- the other objects are filtered out in SQL and never
#: reach the evaluator. The bound is high because parent scope makes conditions
#: per-parent: governing N models individually costs N objects, so a small limit would
#: be exhausted by a modest workspace.
#:
#: Enforced by the slot column's range CHECK plus its UNIQUE, which together bound the
#: count without a count-then-insert race.
MAX_CONDITIONS_PER_ROLE_TYPE = 100

#: Tags MLflow writes itself. Exempt from request conditions (D4) -- an admin must
#: not be able to block MLflow's own bookkeeping, e.g. ``mlflow.runName`` -- but
#: permitted in resource conditions, where testing current ``mlflow.*`` state is
#: both safe and useful.
RESERVED_TAG_PREFIX = "mlflow."

#: Resource types a condition may be authored for.
#:
#: Narrower than the grant vocabulary, and deliberately so: a condition is only
#: meaningful for a type that carries tags or aliases and has a mutating route that
#: names them. Authoring a condition for anything else would be advertised
#: protection that never fires, so it is rejected at write time.
#:
#: ``prompt`` and ``prompt_version`` are present at full parity with their
#: registered-model counterparts (D2). A prompt *is* a registered model carrying the
#: prompt marker tag, so the request-extraction map already fires for both -- parity
#: costs two entries here rather than a second code path.
#:
#: Absent: ``assessment`` (the RFC excludes it), and the deprecated stage-transition
#: surface, whose ``stage`` field is neither a tag nor an alias (D15/D16).
SUPPORTED_RESOURCE_TYPES = frozenset({
    "experiment",
    "run",
    "trace",
    "logged_model",
    "registered_model",
    "registered_model_version",
    "prompt",
    "prompt_version",
    "mcp_server",
    "mcp_server_version",
})

#: Types that own aliases, and so supply the ``alias`` clause (D18). A version's
#: alias list names aliases stored on its *parent*, so gating the version on them
#: would let one alias be governed under two different resource types.
ALIAS_OWNING_RESOURCE_TYPES = frozenset({"registered_model", "prompt", "mcp_server"})

#: The version types, named so an alias condition mistakenly placed on one can say
#: which registry entry to condition instead. Kept beside the constant above because
#: the two encode the same fact about where an alias lives.
_VERSION_RESOURCE_TYPES = frozenset({
    "registered_model_version",
    "prompt_version",
    "mcp_server_version",
})

#: Each child type's single direct parent. Parent scope is *exact*: a condition scoped
#: to experiment 42 governs that experiment's children and nothing else. There is no
#: inheritance across types -- a scoped ``run`` condition does not reach traces or
#: logged models -- so this map is the whole containment vocabulary a condition can name.
#:
#: The version rows mirror ``ALIAS_OWNING_RESOURCE_TYPES``: a version's aliases live on
#: its registry entry, and so does its parent scope. Both encode the same containment.
PARENT_RESOURCE_TYPES: "dict[str, str]" = {
    "run": "experiment",
    "trace": "experiment",
    "logged_model": "experiment",
    "registered_model_version": "registered_model",
    "prompt_version": "prompt",
    "mcp_server_version": "mcp_server",
}

#: Types with no direct parent, and so no parent scope.
#:
#: Declared literally rather than derived as "supported minus parented". A derived set
#: would make the completeness test tautological: a new child type added to
#: ``SUPPORTED_RESOURCE_TYPES`` but forgotten here would silently become parentless and
#: reject the scope it should accept. Spelling both sets out means that omission fails a
#: test instead.
PARENTLESS_RESOURCE_TYPES = frozenset({
    "experiment",
    "registered_model",
    "prompt",
    "mcp_server",
})


def validate_condition_resource_type(resource_type: str) -> None:
    """Reject a resource type that no condition could govern.

    Fails at authoring time rather than silently never matching, because a condition
    on an unsupported type is worse than no condition: the admin believes a
    restriction is in force.
    """
    if resource_type not in SUPPORTED_RESOURCE_TYPES:
        raise MlflowException(
            f"Mutation conditions are not supported for resource type '{resource_type}'. "
            f"Supported types are {sorted(SUPPORTED_RESOURCE_TYPES)} -- a condition is only "
            f"meaningful for a type that carries tags or aliases and has a mutating route "
            f"naming them.",
            error_code=INVALID_PARAMETER_VALUE,
        )


def validate_condition_parent_scope(
    resource_type: str,
    parent_resource_type: "str | None",
    parent_resource_id: "str | None",
) -> None:
    """Reject a parent scope that could never match the target type.

    Same reasoning as :func:`validate_condition_resource_type`, applied to scope rather
    than type: a scope that cannot match stores a restriction the admin believes is in
    force, which is worse than no restriction at all. So every rejection here is a thing
    that would otherwise have been persisted and never fired.

    Unscoped -- both arguments ``None`` -- is the shape that predates parent scope and
    stays valid for every supported type, including the parentless ones.
    """
    validate_condition_resource_type(resource_type)
    if parent_resource_type is None and parent_resource_id is None:
        return
    if parent_resource_type is None or parent_resource_id is None:
        raise MlflowException(
            "A parent scope needs both 'parent_resource_type' and 'parent_resource_id', "
            "or neither. A type without an ID would name every parent of that type, and "
            "an ID without a type names nothing -- both are restrictions that were not "
            "written, so neither is inferred.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if resource_type in PARENTLESS_RESOURCE_TYPES:
        raise MlflowException(
            f"Resource type '{resource_type}' has no direct parent, so a condition on it "
            f"cannot be parent-scoped. Scope it to the workspace instead by omitting both "
            f"parent fields. Parent-scopable types are "
            f"{sorted(PARENT_RESOURCE_TYPES)}.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    expected = PARENT_RESOURCE_TYPES[resource_type]
    if parent_resource_type != expected:
        raise MlflowException(
            f"The direct parent of '{resource_type}' is '{expected}', not "
            f"'{parent_resource_type}'. Parent scope is exact and does not inherit across "
            f"resource types, so only the declared parent can be named.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if not parent_resource_id.strip():
        raise MlflowException(
            "A parent resource ID must be a non-empty string. An empty ID is not a "
            "wildcard: stored, it would match no parent while reading as a scope. It is "
            "refused rather than treated as unscoped, because an admin who asked for a "
            "narrowing must not silently receive a broadening.",
            error_code=INVALID_PARAMETER_VALUE,
        )


class ConditionScope(Enum):
    """Which conditions apply to an operation.

    There is no ``NONE``: an operation that no condition could govern simply has no
    :class:`ConditionContext`, which is a cheaper and less error-prone way to say
    "not applicable" than a scope that must be checked for.
    """

    #: Creating a resource. Request conditions apply; resource conditions are
    #: vacuous, because the resource does not exist yet.
    CREATE = auto()
    #: Mutating an existing resource. Both kinds apply.
    MUTATE = auto()


class Clause(NamedTuple):
    """One parsed comparison.

    ``identifier`` is the flat name for a request clause (``tag_key``) and the
    prefix for a resource clause (``tags``); ``key`` is populated only for resource
    clauses, naming which tag or alias is being tested.
    """

    identifier: str
    key: str | None
    comparator: str
    value: str | tuple[str, ...]

    def describe(self) -> str:
        """Render the clause approximately as authored, for error messages."""
        lhs = self.identifier if self.key is None else f"{self.identifier}.{self.key}"
        if isinstance(self.value, tuple):
            rendered = "(" + ", ".join(f"'{v}'" for v in self.value) + ")"
        else:
            rendered = f"'{self.value}'"
        return f"{lhs} {self.comparator} {rendered}"


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


#: The condition-relevant values a request carries, one shape per resource type.
#:
#: One per type rather than one per capability so a type that later diverges is a local
#: edit -- a field added to the shape that needs it -- instead of a new shared class and a
#: re-pointing of every type that happened to use the old one. Several are identical
#: today; that is the cost of keeping them independent, and it is paid once.
#:
#: The RFC's vocabulary is tags and aliases only, so what varies today is alias ownership:
#: the registry entry owns aliases, its versions do not (D18). A type whose shape omits a
#: field cannot carry it -- the attempt is a ``TypeError`` at the line that made it, not a
#: value silently ignored later.
#:
#: ``tags`` is a sequence of ``(key, value)`` pairs rather than a mapping because a batch
#: request may set the same key twice, and because a *deletion* names a key with no value,
#: represented as ``value=None``. That ``None`` is what lets a delete be gated on the key
#: it removes (D12) while a ``tag_value`` clause stays vacuous over it (D13).

_TagPairs = tuple[tuple[str, str | None], ...]


class ExperimentRequestValues(NamedTuple):
    tags: _TagPairs = ()


class RunRequestValues(NamedTuple):
    tags: _TagPairs = ()


class TraceRequestValues(NamedTuple):
    tags: _TagPairs = ()


class LoggedModelRequestValues(NamedTuple):
    tags: _TagPairs = ()


class RegisteredModelRequestValues(NamedTuple):
    """Owns aliases as well as tags (D18)."""

    tags: _TagPairs = ()
    aliases: tuple[str, ...] = ()


class RegisteredModelVersionRequestValues(NamedTuple):
    """No ``aliases``: an alias set on a version belongs to its registry entry (D18)."""

    tags: _TagPairs = ()


class McpServerVersionRequestValues(NamedTuple):
    """No ``aliases``: an MCP server's aliases live on the server, not the version (D18)."""

    tags: _TagPairs = ()


class McpServerRequestValues(NamedTuple):
    """An MCP server carries the same entry/version/alias shape as a registry entry, and
    like one it owns the aliases its versions are named by (D18).
    """

    tags: _TagPairs = ()
    aliases: tuple[str, ...] = ()


class PromptRequestValues(NamedTuple):
    """Owns aliases as well as tags (D18)."""

    tags: _TagPairs = ()
    aliases: tuple[str, ...] = ()


class PromptVersionRequestValues(NamedTuple):
    """No ``aliases``, for the same reason as a model version (D18)."""

    tags: _TagPairs = ()


#: The condition-relevant current state of one existing resource, one shape per type.
#:
#: ``aliases`` is present only on the types that *own* aliases. A missing field is a
#: stronger statement than an empty mapping: empty says "none right now", missing says the
#: type can never have them. Both deny an alias clause, but only the second makes
#: projecting one a programming error rather than a silent no-op.


class ExperimentResourceValues(NamedTuple):
    resource_id: str
    tags: Mapping[str, str] = {}


class RunResourceValues(NamedTuple):
    resource_id: str
    tags: Mapping[str, str] = {}


class TraceResourceValues(NamedTuple):
    resource_id: str
    tags: Mapping[str, str] = {}


class LoggedModelResourceValues(NamedTuple):
    resource_id: str
    tags: Mapping[str, str] = {}


class RegisteredModelResourceValues(NamedTuple):
    resource_id: str
    tags: Mapping[str, str] = {}
    aliases: Mapping[str, str] = {}


class RegisteredModelVersionResourceValues(NamedTuple):
    resource_id: str
    tags: Mapping[str, str] = {}


class McpServerVersionResourceValues(NamedTuple):
    resource_id: str
    tags: Mapping[str, str] = {}


class McpServerResourceValues(NamedTuple):
    resource_id: str
    tags: Mapping[str, str] = {}
    aliases: Mapping[str, str] = {}


class PromptResourceValues(NamedTuple):
    resource_id: str
    tags: Mapping[str, str] = {}
    aliases: Mapping[str, str] = {}


class PromptVersionResourceValues(NamedTuple):
    resource_id: str
    tags: Mapping[str, str] = {}


#: Which shape each resource type declares. The contract a validator is held to, stated
#: once: the class it must construct, and by omission the fields it cannot set. Every
#: supported type appears -- a missing entry is a ``KeyError`` at the first request, which
#: is how a new type is forced to declare its shape rather than silently borrowing another.
REQUEST_VALUES_SHAPES: "dict[str, type]" = {
    "experiment": ExperimentRequestValues,
    "run": RunRequestValues,
    "trace": TraceRequestValues,
    "logged_model": LoggedModelRequestValues,
    "registered_model": RegisteredModelRequestValues,
    "registered_model_version": RegisteredModelVersionRequestValues,
    "prompt": PromptRequestValues,
    "prompt_version": PromptVersionRequestValues,
    "mcp_server": McpServerRequestValues,
    "mcp_server_version": McpServerVersionRequestValues,
}

RESOURCE_VALUES_SHAPES: "dict[str, type]" = {
    "experiment": ExperimentResourceValues,
    "run": RunResourceValues,
    "trace": TraceResourceValues,
    "logged_model": LoggedModelResourceValues,
    "registered_model": RegisteredModelResourceValues,
    "registered_model_version": RegisteredModelVersionResourceValues,
    "prompt": PromptResourceValues,
    "prompt_version": PromptVersionResourceValues,
    "mcp_server": McpServerResourceValues,
    "mcp_server_version": McpServerVersionResourceValues,
}

#: Annotation-only unions. There is no single constructible ``RequestValues``: a caller
#: builds the shape its resource type declares, which is what makes a field the type
#: cannot carry a ``TypeError`` rather than a silently unused value.
RequestValues = (
    ExperimentRequestValues
    | RunRequestValues
    | TraceRequestValues
    | LoggedModelRequestValues
    | RegisteredModelRequestValues
    | RegisteredModelVersionRequestValues
    | PromptRequestValues
    | PromptVersionRequestValues
)
ResourceValues = (
    ExperimentResourceValues
    | RunResourceValues
    | TraceResourceValues
    | LoggedModelResourceValues
    | RegisteredModelResourceValues
    | RegisteredModelVersionResourceValues
    | PromptResourceValues
    | PromptVersionResourceValues
)


def request_values_shape(resource_type: str) -> "type":
    """The request-values class a validator for this type must construct."""
    return REQUEST_VALUES_SHAPES[resource_type]


def resource_values_shape(resource_type: str) -> "type":
    """The resource-values class this type's state is projected into."""
    return RESOURCE_VALUES_SHAPES[resource_type]


class ConditionContext(NamedTuple):
    """A validator's declaration: "this operation targets resources of this type".

    Deliberately **not** derived from the permission ``Requirement``. Inference from
    it fails in both directions on the create-in-experiment path: the
    ``experiment``/``update`` requirement has a mutating action but names the
    *container* (conditioning it would deny wrongly), while the requirement that
    does name the target carries a non-capability action and so would be skipped.
    The validator knows which resource it is about to change; nothing else reliably
    does.

    ``resource_ids`` holds **ids, never loaded entities**. The framework pulls
    attributes from them, and only if a resource condition actually exists -- which
    is what keeps the common paths free of extra queries.

    ``resource_id_resolver`` is for a cascade, whose children the request never names. It is
    called ONLY when a target condition actually exists for this type, which is what keeps an
    unconditioned cascade from paying for an enumeration it would discard. It returns the
    child ids, or ``None`` when they could not be enumerated -- and those are different
    answers: no children lets the cascade proceed, while "could not enumerate" must deny,
    because a condition that cannot be evaluated must never pass vacuously.

    ``parent_resource_id`` is the target's resolved direct parent, and is what lets a
    parent-scoped condition be selected. Validators that mutate a child type already
    hold it -- sub-resource routes resolve the experiment as their grant anchor, and
    version routes carry the registry name in the request -- so supplying it costs no
    extra fetch. ``None`` for a parentless type is correct and expected; ``None`` for a
    child type means the parent could not be resolved, which the gate must treat as a
    refusal rather than as "no scoped condition applies", since a request outside every
    scope would otherwise escape exactly the conditions written to govern it.
    """

    resource_type: str
    scope: ConditionScope
    request: RequestValues
    resource_ids: tuple[str, ...] = ()
    resource_id_resolver: "Callable[[], tuple[str, ...] | None] | None" = None
    parent_resource_id: str | None = None


class MutationConditionSpec(NamedTuple):
    """One role's conditions for one resource type, as loaded from the store."""

    resource_type: str
    value_condition: str | None = None
    target_condition: str | None = None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _identifier_head(raw: str) -> str:
    """Lowercased, backtick-trimmed leading segment of a left-hand side."""
    return SearchUtils._trim_backticks(raw.strip()).partition(".")[0].strip().lower()


def _is_condition_identifier(token, namespace: str) -> bool:
    """Is this bare token an identifier from *either* namespace?

    Deliberately not namespace-scoped. Joining only the current namespace's
    identifiers would make ``alias = 'x'`` in a *resource* condition fail as a
    generic "invalid clause", when the useful message is the cross-namespace one
    telling the admin that ``alias`` belongs in the request condition. Joining both
    lets :func:`_split_identifier` produce that message. An unknown bare keyword is
    still left loose and still reported as invalid.
    """
    head = _identifier_head(token.value)
    return head in REQUEST_IDENTIFIERS or head in RESOURCE_PREFIXES


def _join_keyword_identifier_tokens(tokens, namespace: str):
    """Rejoin clauses whose identifier sqlparse classified as a SQL keyword.

    ``alias`` is a SQL keyword, so ``alias = 'champion'`` tokenizes as three loose
    tokens rather than a ``Comparison`` and would be rejected as an invalid clause.
    The RFC names the identifier ``alias``, and requiring admins to write
    ``` `alias` ``` instead would be a trap -- the unquoted form looks correct and
    is what anyone would try first.

    So we stitch those runs back into a ``Comparison`` ourselves, the same way
    ``SearchUtils._join_in_comparison_tokens`` does for ``IN`` and for the trace
    ``timestamp`` builtin. Only identifiers in the namespace's own vocabulary are
    joined, so this cannot resurrect a clause that should have been rejected: an
    unknown bare keyword stays loose and is still reported as invalid.
    """
    joined = []
    pending = [t for t in tokens if not t.is_whitespace]
    index = 0
    while index < len(pending):
        token = pending[index]
        remaining = pending[index + 1 :]
        if (
            token.ttype in (TokenType.Keyword, TokenType.Name, TokenType.Name.Builtin)
            and _is_condition_identifier(token, namespace)
            and remaining
        ):
            # <identifier> <comparison-operator> <value>
            if remaining[0].ttype is TokenType.Operator.Comparison and len(remaining) >= 2:
                joined.append(Comparison(TokenList([token, remaining[0], remaining[1]])))
                index += 3
                continue
            # <identifier> IN|NOT IN ( ... )
            if (
                remaining[0].ttype is TokenType.Keyword
                and remaining[0].value.upper().replace(" ", " ").strip() in ("IN", "NOT IN")
                and len(remaining) >= 2
                and isinstance(remaining[1], Parenthesis)
            ):
                joined.append(Comparison(TokenList([token, remaining[0], remaining[1]])))
                index += 3
                continue
            # <identifier> NOT IN ( ... ) tokenized as two keywords
            if (
                remaining[0].ttype is TokenType.Keyword
                and remaining[0].value.upper().strip() == "NOT"
                and len(remaining) >= 3
                and remaining[1].ttype is TokenType.Keyword
                and remaining[1].value.upper().strip() == "IN"
                and isinstance(remaining[2], Parenthesis)
            ):
                joined.append(
                    Comparison(TokenList([token, remaining[0], remaining[1], remaining[2]]))
                )
                index += 4
                continue
        joined.append(token)
        index += 1
    return joined


def _invalid_statement_token(token) -> bool:
    """Mirror of ``SearchUtils._invalid_statement_token_search_runs``.

    Anything that is not a comparison, whitespace, or ``AND`` is rejected -- which
    is what gives us the RFC's no-``OR`` rule structurally, rather than as a
    separate check that could be forgotten.
    """
    if (
        isinstance(token, Comparison)
        or token.is_whitespace
        or token.match(ttype=TokenType.Keyword, values=["AND"])
    ):
        return False
    return True


def _split_identifier(raw: str, namespace: str) -> tuple[str, str | None]:
    """Split the left-hand side into ``(identifier, key)`` and validate it.

    Rejects cross-namespace identifiers explicitly: a ``tags.x`` clause in a request
    condition and a ``tag_key`` clause in a resource condition are both silently
    meaningless otherwise, and an admin who mixes them would believe they had
    written a restriction that never fires.
    """
    stripped = SearchUtils._trim_backticks(raw.strip())

    if namespace == NAMESPACE_REQUEST:
        identifier = stripped.lower()
        if identifier in REQUEST_IDENTIFIERS:
            return identifier, None
        if identifier.split(".", 1)[0] in RESOURCE_PREFIXES:
            raise MlflowException(
                f"'{raw}' is a resource-condition identifier and cannot be used in a request "
                f"condition. Request conditions constrain the values being set, so they use "
                f"{sorted(REQUEST_IDENTIFIERS)}. To constrain which resources may be mutated, "
                f"put '{raw}' in the resource condition instead.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        raise MlflowException(
            f"Invalid request-condition identifier '{raw}'. Valid identifiers are "
            f"{sorted(REQUEST_IDENTIFIERS)}.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    # Resource namespace: must be prefix.key
    head, _, tail = stripped.partition(".")
    prefix = head.strip().lower()
    if prefix not in RESOURCE_PREFIXES:
        if prefix in REQUEST_IDENTIFIERS:
            raise MlflowException(
                f"'{raw}' is a request-condition identifier and cannot be used in a resource "
                f"condition. Resource conditions constrain which resources may be mutated, so "
                f"they use 'tags.<key>' or 'aliases.<name>'. To constrain the values being set, "
                f"put '{raw}' in the request condition instead.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        raise MlflowException(
            f"Invalid resource-condition identifier '{raw}'. Valid identifiers are "
            f"'tags.<key>' and 'aliases.<name>'.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    key = SearchUtils._trim_backticks(tail.strip())
    key = SearchUtils._strip_quotes(key)
    if not key:
        raise MlflowException(
            f"Resource-condition identifier '{raw}' is missing a key. Use "
            f"'{prefix}.<name>', for example 'tags.lifecycle'.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    return prefix, key


def _parse_value(token, comparator: str, raw_lhs: str) -> str | tuple[str, ...]:
    """Extract the right-hand side. ``IN``/``NOT IN`` yield a tuple, others a str."""
    if comparator in ("IN", "NOT IN"):
        if not isinstance(token, Parenthesis):
            raise MlflowException(
                f"'{comparator}' for '{raw_lhs}' requires a parenthesised list of quoted "
                f"strings, for example \"{raw_lhs} {comparator} ('a', 'b')\".",
                error_code=INVALID_PARAMETER_VALUE,
            )
        values = SearchUtils._parse_list_from_sql_token(token)
        if not values:
            raise MlflowException(
                f"'{comparator}' for '{raw_lhs}' requires at least one value.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        return tuple(str(v) for v in values)
    return str(SearchUtils._strip_quotes(token.value, expect_quoted_value=True))


def _parse_comparison(comparison: Comparison, namespace: str) -> Clause:
    tokens = [t for t in comparison.tokens if not t.is_whitespace]
    # `NOT IN` may arrive as one keyword token or as two, depending on how sqlparse
    # split it; collapse the two-token form so the shape below is uniform.
    if (
        len(tokens) == 4
        and tokens[1].ttype is TokenType.Keyword
        and tokens[1].value.upper().strip() == "NOT"
        and tokens[2].ttype is TokenType.Keyword
        and tokens[2].value.upper().strip() == "IN"
    ):
        comparator_value = "NOT IN"
        raw_lhs = tokens[0].value
        value_token = tokens[3]
    elif len(tokens) == 3:
        comparator_value = tokens[1].value.upper().strip()
        raw_lhs = tokens[0].value
        value_token = tokens[2]
    else:
        raise MlflowException(
            f"Invalid clause '{comparison}' in condition. Expected the form "
            f"<identifier> <comparator> <value>.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    identifier, key = _split_identifier(raw_lhs, namespace)

    comparator = " ".join(comparator_value.split())
    if comparator not in ALLOWED_COMPARATORS:
        raise MlflowException(
            f"Comparator '{comparator_value}' is not supported in conditions. Supported "
            f"comparators are {sorted(ALLOWED_COMPARATORS)}. Condition values are always "
            f"strings, so ordering comparators would compare lexicographically rather than "
            f"numerically.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    value = _parse_value(value_token, comparator, raw_lhs)

    if namespace == NAMESPACE_REQUEST and identifier == REQUEST_IDENTIFIER_TAG_KEY:
        # Request side: the reserved name is the clause's VALUE (``tag_key = 'mlflow.x'``).
        _reject_reserved_keys(value)
    elif namespace == NAMESPACE_RESOURCE and identifier == RESOURCE_PREFIX_TAGS:
        # Resource side: it is the clause's KEY (``tags.mlflow.x = '...'``).
        _reject_reserved_keys(key)

    return Clause(identifier=identifier, key=key, comparator=comparator, value=value)


def _reject_reserved_keys(value: str | tuple[str, ...]) -> None:
    """D4: ``mlflow.*`` keys may not appear in a condition, in EITHER namespace.

    MLflow writes these itself -- ``mlflow.runName`` on a run, the prompt marker on a
    registered model -- so a *request* condition constraining them would make an admin
    able to break ordinary logging, and would fail in ways that look like MLflow bugs
    rather than policy.

    A *resource* condition on one is refused for a different reason that lands in the same
    place: it restricts nothing. A reserved tag is user-writable with an arbitrary value --
    ``set_tag`` accepts ``mlflow.runName``, ``mlflow.user`` and ``mlflow.source.type``
    unchallenged -- and by the rule above no request condition can ever gate that write. So
    the holder of the grant simply sets the tag to whatever makes the condition pass:
    ``tags.mlflow.runName != 'secret'`` is escaped by renaming the run, and
    ``tags.mlflow.user = 'alice'`` reads as "only alice's runs" while being forgeable by
    anyone who can set a tag.

    Both directions are therefore the same hazard -- a protection the admin believes is in
    force that is not -- and rejecting at authoring time is clearer than discovering it when
    the restriction fails to bite.

    The test is a prefix test on the tag key, so a user-owned ``mlflow_stage`` or
    ``team.mlflow.note`` is unaffected.
    """
    candidates = value if isinstance(value, tuple) else (value,)
    if reserved := [v for v in candidates if v.startswith(RESERVED_TAG_PREFIX)]:
        raise MlflowException(
            f"Conditions may not reference reserved tag keys {sorted(reserved)} (the "
            f"'{RESERVED_TAG_PREFIX}' prefix is written by MLflow itself, so constraining it "
            f"would block MLflow's own tag writes, and testing it restricts nothing because "
            f"the same keys are freely settable and no request condition may gate them).",
            error_code=INVALID_PARAMETER_VALUE,
        )


def parse_condition(filter_string: str | None, namespace: str) -> tuple[Clause, ...]:
    """Parse a condition filter string into clauses.

    Returns an empty tuple for an empty or absent string: "no condition" and "a
    condition that constrains nothing" are the same thing, and both mean
    unconstrained.

    Every clause is combined with ``AND``. ``OR`` is rejected, because a condition is a restriction
    and a disjunction of restrictions is a weaker restriction -- which reads as
    though it tightened something while loosening it.
    """
    if namespace not in NAMESPACES:
        raise MlflowException(
            f"Unknown condition namespace '{namespace}'. Expected one of {sorted(NAMESPACES)}.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if not filter_string or not filter_string.strip():
        return ()

    try:
        parsed = sqlparse.parse(filter_string)
    except Exception:
        raise MlflowException(
            f"Error parsing condition '{filter_string}'.", error_code=INVALID_PARAMETER_VALUE
        )
    if len(parsed) == 0 or not isinstance(parsed[0], Statement):
        raise MlflowException(
            f"Invalid condition '{filter_string}'. Could not be parsed.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if len(parsed) > 1:
        raise MlflowException(
            f"Invalid condition '{filter_string}'. Expected a single filter expression.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    tokens = _join_keyword_identifier_tokens(
        _join_in_comparison_tokens(parsed[0].tokens), namespace
    )
    if invalids := list(filter(_invalid_statement_token, tokens)):
        rendered = ", ".join(f"'{t}'" for t in invalids)
        raise MlflowException(
            f"Invalid clause(s) in condition: {rendered}. Conditions support comparisons "
            f"joined by AND only -- OR is not supported, because a condition is a restriction "
            f"and OR-ing restrictions weakens rather than tightens them.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    clauses = tuple(_parse_comparison(t, namespace) for t in tokens if isinstance(t, Comparison))
    if not clauses:
        raise MlflowException(
            f"Condition '{filter_string}' contains no comparisons.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if len(clauses) > MAX_CLAUSES:
        raise MlflowException(
            f"Condition has {len(clauses)} clauses, which exceeds the maximum of {MAX_CLAUSES}.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    return clauses


def _validate_clauses_for_resource_type(
    clauses: tuple[Clause, ...], namespace: str, resource_type: str
) -> None:
    """Reject a clause whose identifier the resource type cannot carry.

    Parsing alone is not enough: ``alias = 'champion'`` is well-formed but
    meaningless on a run, and the two namespaces then fail in *opposite*
    directions. A request clause on an absent value is vacuous (D20), so the
    restriction silently never fires and the admin believes a protection is in
    force that is not -- the same fail-open hazard the supported-type check exists
    to prevent. A resource clause on an absent value fails, so it denies every
    mutation of that type instead. Neither is a condition anyone would author on
    purpose, so both are refused here rather than surfacing later as a phantom
    restriction or an outage.

    Only aliases need checking: every supported type carries tags.
    """
    if resource_type in ALIAS_OWNING_RESOURCE_TYPES:
        return

    alias_identifier = (
        REQUEST_IDENTIFIER_ALIAS if namespace == NAMESPACE_REQUEST else RESOURCE_PREFIX_ALIASES
    )
    if not any(clause.identifier == alias_identifier for clause in clauses):
        return

    # A version is the one case an admin plausibly expects to work: the route that
    # sets an alias names a version, so say where the condition belongs instead.
    if resource_type in _VERSION_RESOURCE_TYPES:
        owner = "prompt" if resource_type == "prompt_version" else "registered_model"
        raise MlflowException(
            f"Condition identifier '{alias_identifier}' is not valid for resource type "
            f"'{resource_type}'. An alias is stored on the registry entry rather than on a "
            f"version, so condition the '{owner}' resource type instead -- a version's alias "
            f"list names aliases owned by its parent, and gating the version on them would "
            f"let one alias be governed under two different resource types.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    raise MlflowException(
        f"Condition identifier '{alias_identifier}' is not valid for resource type "
        f"'{resource_type}', which does not carry aliases. Only "
        f"{sorted(ALIAS_OWNING_RESOURCE_TYPES)} do. Use a tag condition instead: an alias "
        f"condition here would never restrict anything, or would deny every mutation of "
        f"this type.",
        error_code=INVALID_PARAMETER_VALUE,
    )


def validate_condition(filter_string: str | None, namespace: str, resource_type: str) -> None:
    """Write-time validation hook. Raises if the condition could not govern.

    Called by the store so a malformed condition can never be persisted -- a
    condition that fails to parse at *evaluation* time would have to either
    fail open (unsafe) or deny every mutation (an outage), so the only good place
    to catch it is on the way in.

    ``resource_type`` is required rather than defaulted: a caller that forgets it
    would persist a condition that silently never fires, so there is deliberately
    no way to ask for the parse without the cross-check.
    """
    clauses = parse_condition(filter_string, namespace)
    _validate_clauses_for_resource_type(clauses, namespace, resource_type)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _compare(clause: Clause, lhs: str) -> bool:
    return SearchUtils.get_comparison_func(clause.comparator)(lhs, clause.value)


def _request_lhs_values(clause: Clause, values: RequestValues) -> tuple[str, ...] | None:
    """Project the request values a clause applies to.

    Returns ``None`` when the request carries nothing of that kind -- which the
    caller reads as "vacuous", not "failed".
    """
    # D4: `mlflow.*` pairs are excluded from BOTH request projections. A request condition can
    # never name a reserved key -- authoring rejects it -- so a reserved key must not decide one
    # either. Because every value must satisfy the clause, leaving them in means MLflow's own
    # tag writes fail an unrelated positive clause: `tag_key = 'team'` would be failed by the
    # `mlflow.user` that MLflow itself attaches, denying a create the admin never restricted.
    # Rejecting a clause that NAMES a reserved key and exempting reserved values from
    # evaluation are different guarantees, and only the first is enforced at authoring time.
    #
    # This removes no expressible policy, precisely because such a clause cannot be written.
    # The resource side is deliberately unfiltered: reading current `mlflow.*` state is how an
    # admin says "only prompts".
    user_tags = tuple(
        (key, value) for key, value in values.tags if not key.startswith(RESERVED_TAG_PREFIX)
    )
    if clause.identifier == REQUEST_IDENTIFIER_TAG_KEY:
        return tuple(key for key, _ in user_tags) or None
    if clause.identifier == REQUEST_IDENTIFIER_TAG_VALUE:
        # A deletion names a key with no value, so its value is None and a
        # tag_value clause has nothing to test (D13). Constraining what a delete
        # removes is the resource condition's job.
        present = tuple(v for _, v in user_tags if v is not None)
        return present or None
    if clause.identifier == REQUEST_IDENTIFIER_ALIAS:
        # A tags-only shape has no ``aliases`` field at all, which is a stronger statement
        # than an empty one: the type cannot set an alias, so a request to it never does,
        # so the clause never applies. Vacuous is therefore the correct reading and not a
        # hole -- and such a condition cannot be stored for such a type anyway, because the
        # store rejects it on the way in.
        return getattr(values, "aliases", ()) or None
    return None


def evaluate_request(clauses: Sequence[Clause], values: RequestValues) -> bool:
    """Do the values this request sets satisfy every clause?

    **Absence is vacuous here** (D13/D20): a clause whose identifier the request
    does not carry passes. This is not leniency, it is necessary -- routes carry
    different subsets of the namespace (``CreateExperiment`` has no alias,
    ``SetRegisteredModelAlias`` has no tag key), so without this rule a single
    condition could never be written that applied to more than one route, and every
    value-free request (a metrics-only ``LogBatch``, a rename) would be denied by a
    ``tag_key`` clause that has nothing to say about it.

    Note this is the **opposite** of the resource side, deliberately. See
    :func:`evaluate_resource`.

    Every value must satisfy the clause: setting ten tags where one is disallowed is
    denied, because the alternative is that a bulk request is a way around a
    restriction that holds for a single one.
    """
    for clause in clauses:
        lhs_values = _request_lhs_values(clause, values)
        if lhs_values is None:
            continue  # vacuous: nothing of this kind in the request
        if not all(_compare(clause, lhs) for lhs in lhs_values):
            return False
    return True


def _resource_lhs(clause: Clause, values: ResourceValues) -> str | None:
    if clause.identifier == RESOURCE_PREFIX_TAGS:
        return values.tags.get(clause.key)
    if clause.identifier == RESOURCE_PREFIX_ALIASES:
        # Missing field -> no alias state -> ``None``, which fails (D20). Fail-closed, the
        # safe direction on this side, and unreachable for the same reason as above.
        return getattr(values, "aliases", {}).get(clause.key)
    return None


def evaluate_resource(clauses: Sequence[Clause], values: ResourceValues) -> bool:
    """Does this resource's current state satisfy every clause?

    **Absence fails here** -- the inverse of :func:`evaluate_request`, and the
    asymmetry is the point. A resource that lacks the tag does not have the state
    the condition describes, so it does not satisfy it. This reuses MLflow's search
    semantics verbatim (``lhs is None`` -> ``False``), so a resource condition
    selects exactly the resources the same filter string would return in a search
    box, and it errs closed.

    Implementing both namespaces the same way would open a hole in one direction or
    the other: vacuous-on-absence here would let an untagged resource slip past
    every ``tags.*`` restriction, and fail-on-absence on the request side would deny
    ordinary value-free writes.

    Absence fails for EVERY comparator, with no exception -- including ``!=``, where
    ``tags.lifecycle != 'prod'`` *denies* a resource carrying no ``lifecycle`` tag,
    because the tag is absent rather than not-'prod'. That surprises admins, but the
    surprise is in the strict direction, and it is what makes deleting a governing tag
    lose access rather than escape the restriction.

    The rule has no reserved-key carve-out because it needs none: D4 rejects a
    ``tags.mlflow.*`` clause at authoring in both namespaces, so one cannot be stored.
    """
    for clause in clauses:
        lhs = _resource_lhs(clause, values)
        if lhs is None:
            return False
        if not _compare(clause, lhs):
            return False
    return True


# ---------------------------------------------------------------------------
# Combination
# ---------------------------------------------------------------------------


def combine(results: Sequence[bool]) -> bool:
    """AND over every applicable condition. That is the whole rule.

    No precedence, no load keys, no tier override, no fold -- deliberately unlike
    the grants path, which needs all of those because grants *add* and a more
    specific grant must be able to override a broader one. Conditions only ever
    subtract, so two conditions can never conflict in a way that needs resolving:
    if either says no, the answer is no.

    The consequence worth stating plainly, because it is the property that makes
    conditions safe to reason about: **adding a role can never lift another role's
    restriction.** A permissive condition, or no condition at all, on a second role
    does not widen the first.
    """
    return all(results)


def condition_load_types(contexts: Sequence[ConditionContext]) -> tuple[str, ...]:
    """The distinct resource types a set of contexts needs conditions for.

    Deduplicated so the loader issues one query regardless of how many contexts a
    validator declares.
    """
    seen: dict[str, None] = {}
    for context in contexts:
        seen.setdefault(context.resource_type, None)
    return tuple(seen)


def condition_load_parents(
    contexts: Sequence[ConditionContext],
) -> "dict[str, tuple[str, ...]]":
    """The resolved direct parents in play, per resource type.

    Handed to the store so the scope predicate runs in SQL: a role holding the full
    ``MAX_CONDITIONS_PER_ROLE_TYPE`` for a type transfers only its unscoped conditions
    plus those naming a parent actually in play.

    Grouped per type rather than flattened into one set of ids, because an id is only
    meaningful against its own type. A request touching runs of experiment 7 and
    versions of model 7 must not let either 7 satisfy the other's scope.
    """
    parents: dict[str, dict[str, None]] = {}
    for context in contexts:
        if context.parent_resource_id is None:
            continue
        parents.setdefault(context.resource_type, {}).setdefault(context.parent_resource_id, None)
    return {resource_type: tuple(ids) for resource_type, ids in parents.items()}


def needs_resource_values(
    contexts: Sequence[ConditionContext], types_with_target: frozenset[str] | set[str]
) -> bool:
    """Would evaluating these contexts require reading resource state?

    The gate calls this before touching a store. Two ways to answer no, and both are
    common: no context is at ``MUTATE`` scope (a create has no prior state), or no
    role has a target condition on any type in play (the configured-but-request-only
    case). Either way the resource is never read.

    Deliberately does **not** also require ``resource_ids``. A ``MUTATE`` context that
    names none still has to reach the gate's target loop, which refuses it (D21) -- an
    operation that cannot say which resources it will change cannot be checked against a
    condition on them. Answering "no reads needed" here would return *allow* instead,
    making a predicate-mode bulk delete a way around every resource condition. The loop
    denies before fetching anything, so this costs no query.
    """
    return any(
        context.scope is ConditionScope.MUTATE and context.resource_type in types_with_target
        for context in contexts
    )


def _validate_request_values_for_resource_type(resource_type: str, request: RequestValues) -> None:
    """Reject request values whose shape the resource type does not declare.

    The shape itself prevents the common mistake: a validator for a tags-only type
    cannot pass an alias, because :class:`TagRequestValues` has no such field and the
    attempt is a ``TypeError`` at the line that made it. What the shape cannot prevent is
    passing the *wrong shape* -- an alias-owning one where a tags-only one belongs, which
    would carry a field the type's conditions can never name. That is what this catches.

    Raises rather than denies: it is a wiring bug, not a user error. A raise surfaces it
    in the tests that exercise the route, and is fail-closed if one reaches production.
    """
    expected = REQUEST_VALUES_SHAPES[resource_type]
    if type(request) is not expected:
        raise MlflowException(
            f"Validator wiring error: resource type '{resource_type}' declares "
            f"{expected.__name__}, but got {type(request).__name__}. The shape states which "
            f"values the type can carry, so passing another means the operation is wired to "
            f"the wrong type or is trying to set something this type does not have.",
            error_code=INVALID_PARAMETER_VALUE,
        )


def context_for(
    resource_type: str,
    resource_id: str | None,
    scope: ConditionScope,
    request: RequestValues | None = None,
    resource_id_resolver: "Callable[[], tuple[str, ...] | None] | None" = None,
    parent_resource_id: str | None = None,
) -> ConditionContext:
    """Build a context, treating a wildcard id as "no specific resource".

    Sub-resource grants are wildcard-only grain, so a child requirement's id is
    literally ``"*"``. Passing that through as a resource id would send the
    framework off to fetch a resource named ``*``. A wildcard means the operation is
    not scoped to one identified resource, so there is nothing for a resource
    condition to read -- request conditions still apply.

    ``parent_resource_id`` is the target's resolved direct parent, needed to select a
    parent-scoped condition. A wildcard is normalised to ``None`` for the same reason
    as the resource id: it names no particular parent.

    Also the one place a validator's request values are checked against the type it
    declared, so the mis-wiring the shared shape cannot prevent fails loudly here
    instead of silently going unread.
    """
    validate_condition_resource_type(resource_type)
    request = request if request is not None else REQUEST_VALUES_SHAPES[resource_type]()
    _validate_request_values_for_resource_type(resource_type, request)
    ids: tuple[str, ...] = ()
    if resource_id is not None and resource_id != "*":
        ids = (resource_id,)
    if parent_resource_id == "*":
        parent_resource_id = None
    return ConditionContext(
        resource_type=resource_type,
        scope=scope,
        request=request,
        resource_ids=ids,
        resource_id_resolver=resource_id_resolver,
        parent_resource_id=parent_resource_id,
    )
