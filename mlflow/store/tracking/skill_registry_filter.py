"""Shared filter application, ordering, and pagination helpers for the skill registry.

Translates parsed filter dicts (from ``SearchUtils.parse_search_filter``) into
SQLAlchemy WHERE clauses and provides a reusable order-by parser.  Both Skill
and Agent Plugin stores consume these helpers through a ``column_map`` that
maps filter keys to the appropriate SQLAlchemy column or expression.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sqlalchemy as sa

from mlflow.entities import SkillStatus
from mlflow.exceptions import MlflowException
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking.dbmodels.models import (
    SqlAgentPlugin,
    SqlAgentPluginVersion,
    SqlAgentPluginVersionMember,
    _agent_plugin_version_has_deleted_member,
)
from mlflow.store.tracking.skill_registry_pagination import (
    SkillRegistryPaginationToken,
)
from mlflow.utils.search_utils import (
    _SKILL_REGISTRY_TAG_COMPARATORS,
    SearchUtils,
    _quote_keyword_fields,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Query
    from sqlalchemy.sql.elements import ClauseElement, ColumnElement

# Parsing already validates operators per field type; these checks guard
# callers that build filter dicts by hand.
_VALID_FILTER_COMPARATORS = {"=", "!=", ">", ">=", "<", "<=", "LIKE", "ILIKE", "IN", "NOT IN"}


def _get_comparison_func(comparator: str, dialect: str, col):
    """Return a comparison function for a mapped column or a computed expression.

    ``SearchUtils.get_sql_comparison_func`` builds MySQL's case-sensitive SQL
    from ``column.class_``, which computed expressions (e.g. a resolved status
    subquery) do not have. Those use the expression-safe variant, which keeps
    MySQL and MSSQL comparisons case-sensitive, matching the column path.
    """
    if hasattr(col, "class_"):
        return SearchUtils.get_sql_comparison_func(comparator, dialect)
    return SearchUtils.get_sql_expression_comparison_func(comparator, dialect)


# ---------------------------------------------------------------------------
# Filter application
# ---------------------------------------------------------------------------


def apply_skill_registry_filters(
    query: Query,
    parsed_filters: list[dict[str, Any]],
    column_map: dict[str, ColumnElement],
    parent_model_class: type,
    tag_model_class: type,
    tag_join_keys: list[str],
    dialect: str,
) -> Query:
    """Apply parsed filter dicts as SQLAlchemy WHERE clauses.

    ``column_map`` maps filter keys to SQLAlchemy columns or expressions.  For
    computed fields (e.g. ``resolved_status``) the caller passes the
    appropriate expression.

    ``parent_model_class`` is the ORM model being queried (e.g. ``SqlSkill``).

    ``tag_join_keys`` lists the column names on the tag model that form the
    join to the parent (e.g. ``["workspace", "organization", "name"]``).

    Filters that require cross-table joins (e.g. ``member_name`` for agent
    plugins) must be extracted from ``parsed_filters`` by the caller and
    handled separately via ``apply_member_name_filter``.  Any attribute key
    not present in ``column_map`` is rejected.
    """
    attribute_filters: list[ClauseElement] = []
    tag_filters: dict[str, list[ClauseElement]] = {}

    for f in parsed_filters:
        type_ = f["type"]
        key = f["key"]
        comparator = f["comparator"].upper()
        value = f["value"]

        if type_ == "attribute":
            if comparator not in _VALID_FILTER_COMPARATORS:
                raise MlflowException.invalid_parameter_value(
                    f"Invalid comparator '{comparator}' for attribute '{key}'."
                )
            if key not in column_map:
                raise MlflowException.invalid_parameter_value(f"Invalid filter attribute '{key}'.")
            col = column_map[key]
            attribute_filters.append(_get_comparison_func(comparator, dialect, col)(col, value))
        elif type_ == "tag":
            if comparator not in _SKILL_REGISTRY_TAG_COMPARATORS:
                raise MlflowException.invalid_parameter_value(
                    f"Invalid comparator '{comparator}' for tag '{key}'."
                )
            if key not in tag_filters:
                key_filter = SearchUtils.get_sql_comparison_func("=", dialect)(
                    tag_model_class.key, key
                )
                tag_filters[key] = [key_filter]
            val_filter = SearchUtils.get_sql_comparison_func(comparator, dialect)(
                tag_model_class.value, value
            )
            tag_filters[key].append(val_filter)
        else:
            raise MlflowException.invalid_parameter_value(f"Invalid filter type '{type_}'.")

    if attribute_filters:
        query = query.filter(*attribute_filters)

    if tag_filters:
        # Multi-tag intersection: each tag key's clauses are OR'd, then HAVING
        # COUNT ensures every requested key matched (same pattern as MCP search).
        sql_tag_filters = (sa.and_(*clauses) for clauses in tag_filters.values())
        join_cols = [getattr(tag_model_class, k) for k in tag_join_keys]
        tag_subquery = (
            sa
            .select(*join_cols)
            .filter(sa.or_(*sql_tag_filters))
            .group_by(*join_cols)
            .having(sa.func.count(sa.literal(1)) == len(tag_filters))
            .subquery()
        )
        join_conds = [
            getattr(parent_model_class, k) == getattr(tag_subquery.c, k) for k in tag_join_keys
        ]
        query = query.join(tag_subquery, sa.and_(*join_conds))

    return query


def apply_member_name_filter(
    query: Query,
    member_name_value: str,
    dialect: str,
) -> Query:
    """Filter agent plugins to those with an eligible version referencing the given skill.

    A version is eligible when it is not deleted and not withdrawn, using the
    same withdrawal rule as latest-version resolution: a version that bundles
    any deleted skill version is withdrawn (per RFC-0008), while deprecated
    members do not withdraw it. Older eligible versions still contribute
    matches.
    """
    # Correlate only the plugin version, so the check covers every member of
    # that version rather than just the member row matched by name.
    is_withdrawn = _agent_plugin_version_has_deleted_member().correlate(SqlAgentPluginVersion)

    member_subquery = (
        sa
        .select(
            SqlAgentPluginVersionMember.plugin_workspace,
            SqlAgentPluginVersionMember.plugin_organization,
            SqlAgentPluginVersionMember.plugin_name,
        )
        .join(
            SqlAgentPluginVersion,
            sa.and_(
                SqlAgentPluginVersionMember.plugin_workspace == SqlAgentPluginVersion.workspace,
                SqlAgentPluginVersionMember.plugin_organization
                == SqlAgentPluginVersion.organization,
                SqlAgentPluginVersionMember.plugin_name == SqlAgentPluginVersion.name,
                SqlAgentPluginVersionMember.plugin_version == SqlAgentPluginVersion.version,
            ),
        )
        .where(
            SearchUtils.get_sql_comparison_func("=", dialect)(
                SqlAgentPluginVersionMember.member_name, member_name_value
            ),
            SqlAgentPluginVersion.status != SkillStatus.DELETED.value,
            ~is_withdrawn,
        )
        .distinct()
        .subquery()
    )
    return query.join(
        member_subquery,
        sa.and_(
            SqlAgentPlugin.workspace == member_subquery.c.plugin_workspace,
            SqlAgentPlugin.organization == member_subquery.c.plugin_organization,
            SqlAgentPlugin.name == member_subquery.c.plugin_name,
        ),
    )


# ---------------------------------------------------------------------------
# Order-by parsing
# ---------------------------------------------------------------------------


def parse_skill_registry_order_by(
    order_by_list: list[str] | None,
    valid_keys: set[str],
    column_map: dict[str, ColumnElement | tuple[ColumnElement, ...]],
    default_tiebreakers: list[ClauseElement],
) -> list[ClauseElement]:
    """Parse an ``order_by`` list into SQLAlchemy clauses.

    A key may map to a tuple of expressions ordered in sequence, each taking
    the requested direction. Agent plugin ``version`` needs this: mapped to
    its materialized SemVer columns it sorts by precedence, whereas the raw
    version string would sort ``10.0.0`` before ``9.1.0``.

    ``default_tiebreakers`` are appended when their expression has not been
    explicitly requested, ensuring deterministic pagination order.
    Deduplication compares expressions structurally, so it covers plain
    columns, columns mapped under a different key, and computed expressions.
    """
    clauses: list[ClauseElement] = []
    observed: set[str] = set()
    ordered_expressions: list[ClauseElement] = []

    if order_by_list:
        for order_by_clause in order_by_list:
            # Backtick-quote sqlparse keywords so they parse as identifiers.
            quoted = _quote_keyword_fields(order_by_clause)
            token_value, is_ascending = SearchUtils._parse_order_by_string(quoted)
            # The base parser strips double quotes only when a direction
            # follows, so normalize the key as _get_identifier does for filters.
            key = SearchUtils._trim_backticks(SearchUtils._strip_quotes(token_value.strip()))
            if key not in valid_keys:
                raise MlflowException.invalid_parameter_value(
                    f"Invalid order_by key '{key}'. Valid keys: {sorted(valid_keys)}"
                )
            if key in observed:
                raise MlflowException.invalid_parameter_value(f"Duplicate order_by field: '{key}'")
            observed.add(key)
            columns = column_map[key]
            if not isinstance(columns, tuple):
                columns = (columns,)
            for col in columns:
                ordered_expressions.append(_as_expression(col))
                clauses.append(col.asc() if is_ascending else col.desc())

    for clause in default_tiebreakers:
        tiebreaker = _as_expression(getattr(clause, "element", clause))
        if not any(tiebreaker.compare(expr) for expr in ordered_expressions):
            clauses.append(clause)

    return clauses


def _as_expression(value):
    # Mapped attributes (e.g. SqlSkill.name) are not ClauseElements themselves.
    return value.__clause_element__() if hasattr(value, "__clause_element__") else value


# ---------------------------------------------------------------------------
# Result pagination
# ---------------------------------------------------------------------------


def paginate_results(
    results: list[Any],
    max_results: int,
    offset: int,
    filter_string: str | None,
    order_by: list[str] | None,
    query_scope: str,
) -> PagedList:
    """Trim results to ``max_results`` and compute the next page token.

    The caller queries for ``max_results + 1`` rows; if the extra row is
    present, a next-page token is returned.
    """
    next_token = None
    if len(results) > max_results:
        next_token = SkillRegistryPaginationToken(
            filter_string=filter_string,
            order_by=order_by,
            offset=offset + max_results,
            query_scope=query_scope,
        ).encode()
    return PagedList(results[:max_results], next_token)
