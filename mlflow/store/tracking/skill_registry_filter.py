"""Shared filter application, ordering, and pagination helpers for the skill registry.

Translates parsed filter dicts (from ``SearchUtils.parse_search_filter``) into
SQLAlchemy WHERE clauses and provides a reusable order-by parser.  Both Skill
and Agent Plugin stores consume these helpers through a ``column_map`` that
maps filter keys to the appropriate SQLAlchemy column or expression.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sqlalchemy as sa

from mlflow.exceptions import MlflowException
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking.skill_registry_pagination import (
    SkillRegistryPaginationToken,
)
from mlflow.utils.search_utils import SearchUtils

if TYPE_CHECKING:
    from sqlalchemy.orm import Query
    from sqlalchemy.sql.elements import ClauseElement, ColumnElement

_VALID_FILTER_COMPARATORS = {"=", "!=", ">", ">=", "<", "<=", "LIKE", "ILIKE", "IN"}


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
    """
    attribute_filters: list[ClauseElement] = []
    tag_filters: dict[str, list[ClauseElement]] = {}

    for f in parsed_filters:
        type_ = f["type"]
        key = f["key"]
        comparator = f["comparator"]
        value = f["value"]

        if type_ == "attribute":
            if comparator not in _VALID_FILTER_COMPARATORS:
                raise MlflowException.invalid_parameter_value(
                    f"Invalid comparator '{comparator}' for attribute '{key}'."
                )
            if key not in column_map:
                raise MlflowException.invalid_parameter_value(f"Invalid filter attribute '{key}'.")
            col = column_map[key]
            attribute_filters.append(
                SearchUtils.get_sql_comparison_func(comparator, dialect)(col, value)
            )
        elif type_ == "tag":
            if comparator not in _VALID_FILTER_COMPARATORS:
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
    """Filter agent plugins to those with a version referencing the given skill.

    Matches any version (not only the latest-resolved one), so this finds
    plugins pinned to older versions as well.
    """
    # Lazy import to avoid circular dependency with dbmodels.models.
    from mlflow.store.tracking.dbmodels.models import (
        SqlAgentPlugin,
        SqlAgentPluginVersionMember,
    )

    member_subquery = (
        sa
        .select(
            SqlAgentPluginVersionMember.plugin_workspace,
            SqlAgentPluginVersionMember.plugin_organization,
            SqlAgentPluginVersionMember.plugin_name,
        )
        .where(
            SearchUtils.get_sql_comparison_func("=", dialect)(
                SqlAgentPluginVersionMember.member_name, member_name_value
            )
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
    column_map: dict[str, ColumnElement],
    default_tiebreakers: list[ClauseElement],
) -> list[ClauseElement]:
    """Parse an ``order_by`` list into SQLAlchemy clauses.

    ``default_tiebreakers`` are appended when their keys have not been
    explicitly requested, ensuring deterministic pagination order.
    """
    clauses: list[ClauseElement] = []
    observed: set[str] = set()

    if order_by_list:
        for order_by_clause in order_by_list:
            token_value, is_ascending = SearchUtils._parse_order_by_string(order_by_clause)
            key = token_value.strip()
            if key not in valid_keys:
                raise MlflowException.invalid_parameter_value(
                    f"Invalid order_by key '{key}'. Valid keys: {sorted(valid_keys)}"
                )
            if key in observed:
                raise MlflowException.invalid_parameter_value(f"Duplicate order_by field: '{key}'")
            observed.add(key)
            col = column_map[key]
            clauses.append(col.asc() if is_ascending else col.desc())

    for clause in default_tiebreakers:
        col_key = clause.element.key if hasattr(clause, "element") else None
        if col_key not in observed:
            clauses.append(clause)

    return clauses


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
