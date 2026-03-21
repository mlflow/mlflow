"""MLflow filter DSL to OpenSearch Query DSL translator.

This module converts MLflow's SQL-like filter strings (e.g.
``metrics.accuracy > 0.9 AND tags.env = 'prod'``) into OpenSearch
Query DSL dictionaries that can be passed directly to the
``opensearch-py`` client's ``search()`` method.

The translator handles:
- Attribute filters (term, range, wildcard queries)
- Metric, param, and tag filters (cross-index sub-queries)
- Span-level filters for trace search (full-text, nested)
- Logical AND combination (OpenSearch bool/must)
- Pagination via search_after
"""

from __future__ import annotations

import logging

from mlflow.utils.search_utils import SearchUtils

_logger = logging.getLogger(__name__)


# Mapping of MLflow comparator tokens to OpenSearch range operators
_RANGE_COMPARATORS = {
    ">": "gt",
    ">=": "gte",
    "<": "lt",
    "<=": "lte",
}

# Attribute fields that are stored as numeric (long) in OpenSearch
_NUMERIC_ATTRIBUTES = {
    "start_time",
    "end_time",
    "creation_time",
    "last_update_time",
    "request_time",
    "execution_duration",
    "timestamp_ms",
    "execution_time_ms",
    "end_time_ms",
}


def _sql_like_to_wildcard(pattern: str) -> str:
    """Convert a SQL LIKE pattern to an OpenSearch wildcard pattern.

    ``%`` → ``*`` and ``_`` → ``?``.
    """
    return pattern.replace("%", "*").replace("_", "?")


class OpenSearchQueryTranslator:
    """Translate MLflow filter expressions into OpenSearch Query DSL."""

    def translate(
        self,
        filter_string: str,
        entity_type: str = "run",
    ) -> dict:
        """Translate a filter string into an OpenSearch query body.

        Args:
            filter_string: An MLflow filter expression, e.g.
                ``"metrics.accuracy > 0.9 AND tags.env = 'prod'"``.
            entity_type: One of ``"experiment"``, ``"run"``, or ``"trace"``.

        Returns:
            A dictionary suitable for ``opensearch_client.search(body=...)``.
        """
        if not filter_string:
            return {"bool": {"must": [{"match_all": {}}]}}

        parsed = SearchUtils.parse_search_filter(filter_string)
        must_clauses: list[dict] = []
        sub_queries: list[dict] = []

        for parsed_filter in parsed:
            clause, sub = self._translate_single(parsed_filter, entity_type)
            if clause:
                must_clauses.append(clause)
            if sub:
                sub_queries.append(sub)

        query = {"bool": {"must": must_clauses}} if must_clauses else {"match_all": {}}

        if sub_queries:
            query["_sub_queries"] = sub_queries

        return query

    def _translate_single(self, parsed_filter, entity_type: str):
        """Translate a single parsed filter clause.

        Returns:
            Tuple of (main_query_clause | None, sub_query_info | None).
        """
        filter_type = parsed_filter.get("type", "")
        comparator = parsed_filter.get("comparator", "=")
        key = parsed_filter.get("key", "")
        value = parsed_filter.get("value", "")

        if filter_type in ("metric", "metrics"):
            return None, self._translate_metric_filter(key, comparator, value)
        elif filter_type in ("param", "parameter", "params"):
            return None, self._translate_param_filter(key, comparator, value)
        elif filter_type in ("tag", "tags"):
            return self._translate_tag_filter(key, comparator, value, entity_type), None
        elif filter_type in ("attribute", "attr", "attributes", "run"):
            return self._translate_attribute_filter(key, comparator, value), None
        elif filter_type == "span":
            return None, self._translate_span_filter(key, comparator, value)
        else:
            _logger.warning("Unknown filter type '%s', skipping", filter_type)
            return None, None

    def _translate_attribute_filter(self, key: str, comparator: str, value) -> dict:
        """Translate an attribute filter to an OpenSearch query clause."""
        comparator_upper = comparator.upper()

        if comparator_upper == "IS NULL":
            return {"bool": {"must_not": [{"exists": {"field": key}}]}}
        if comparator_upper == "IS NOT NULL":
            return {"exists": {"field": key}}

        if comparator_upper in ("IN", "NOT IN"):
            values = [v.strip().strip("'\"") for v in value] if isinstance(value, list) else [value]
            terms_query = {"terms": {key: values}}
            if comparator_upper == "NOT IN":
                return {"bool": {"must_not": [terms_query]}}
            return terms_query

        if key in _NUMERIC_ATTRIBUTES:
            return self._numeric_filter(key, comparator, value)

        return self._string_filter(key, comparator, value)

    def _numeric_filter(self, field: str, comparator: str, value) -> dict:
        """Build a range or term query for numeric fields."""
        if comparator in _RANGE_COMPARATORS:
            return {"range": {field: {_RANGE_COMPARATORS[comparator]: value}}}
        if comparator == "=":
            return {"term": {field: value}}
        if comparator == "!=":
            return {"bool": {"must_not": [{"term": {field: value}}]}}
        return {"term": {field: value}}

    def _string_filter(self, field: str, comparator: str, value) -> dict:
        """Build a term, wildcard, or regexp query for string fields."""
        comparator_upper = comparator.upper()
        str_value = str(value).strip("'\"")

        if comparator == "=":
            return {"term": {field: str_value}}
        if comparator == "!=":
            return {"bool": {"must_not": [{"term": {field: str_value}}]}}
        if comparator_upper == "LIKE":
            return {"wildcard": {f"{field}.keyword": _sql_like_to_wildcard(str_value)}}
        if comparator_upper == "ILIKE":
            return {
                "wildcard": {
                    f"{field}.keyword": {
                        "value": _sql_like_to_wildcard(str_value.lower()),
                        "case_insensitive": True,
                    }
                }
            }
        if comparator_upper == "RLIKE":
            return {"regexp": {field: str_value}}
        return {"term": {field: str_value}}

    def _translate_metric_filter(self, key: str, comparator: str, value) -> dict:
        """Translate a metric filter into a sub-query targeting the metrics index."""
        must = [{"term": {"key": key}}]
        if comparator in _RANGE_COMPARATORS:
            must.append({"range": {"value": {_RANGE_COMPARATORS[comparator]: float(value)}}})
        elif comparator == "=":
            must.append({"term": {"value": float(value)}})
        elif comparator == "!=":
            must.append({"bool": {"must_not": [{"term": {"value": float(value)}}]}})

        return {
            "_index": "metrics",
            "_return_field": "run_id",
            "query": {"bool": {"must": must}},
        }

    def _translate_param_filter(self, key: str, comparator: str, value) -> dict:
        """Translate a param filter into a sub-query targeting the params index."""
        str_value = str(value).strip("'\"")
        must = [{"term": {"key": key}}]
        value_clause = self._string_filter("value", comparator, str_value)
        must.append(value_clause)

        return {
            "_index": "params",
            "_return_field": "run_id",
            "query": {"bool": {"must": must}},
        }

    def _translate_tag_filter(
        self, key: str, comparator: str, value, entity_type: str
    ) -> dict | None:
        """Translate a tag filter.

        For experiments (with nested tags), returns a nested query.
        For runs/traces (separate tag index), returns a sub-query.
        """
        comparator_upper = comparator.upper()
        str_value = str(value).strip("'\"")

        if comparator_upper == "IS NULL":
            return {"bool": {"must_not": [{"exists": {"field": key}}]}}
        if comparator_upper == "IS NOT NULL":
            return {"exists": {"field": key}}

        if entity_type == "experiment":
            inner_must: list[dict] = [{"term": {"tags.key": key}}]
            if comparator == "=":
                inner_must.append({"term": {"tags.value.keyword": str_value}})
            elif comparator_upper == "LIKE":
                inner_must.append({
                    "wildcard": {"tags.value.keyword": _sql_like_to_wildcard(str_value)}
                })
            elif comparator_upper == "ILIKE":
                inner_must.append({
                    "wildcard": {
                        "tags.value.keyword": {
                            "value": _sql_like_to_wildcard(str_value.lower()),
                            "case_insensitive": True,
                        }
                    }
                })
            elif comparator == "!=":
                inner_must.append({
                    "bool": {"must_not": [{"term": {"tags.value.keyword": str_value}}]}
                })
            return {
                "nested": {
                    "path": "tags",
                    "query": {"bool": {"must": inner_must}},
                }
            }

        # For runs/traces, tag filter is handled via sub-query
        return self._string_filter(f"tags.{key}", comparator, str_value)

    def _translate_span_filter(self, key: str, comparator: str, value) -> dict:
        """Translate a span-level filter into a sub-query targeting the spans index."""
        str_value = str(value).strip("'\"")
        comparator_upper = comparator.upper()

        if key in ("content", "text"):
            if comparator_upper == "LIKE":
                clause = {"wildcard": {"content": _sql_like_to_wildcard(str_value)}}
            elif comparator_upper == "ILIKE":
                clause = {
                    "wildcard": {
                        "content": {
                            "value": _sql_like_to_wildcard(str_value.lower()),
                            "case_insensitive": True,
                        }
                    }
                }
            elif comparator_upper == "RLIKE":
                clause = {"regexp": {"content": str_value}}
            else:
                clause = {"match": {"content": str_value}}
        else:
            clause = self._string_filter(key, comparator, str_value)

        return {
            "_index": "spans",
            "_return_field": "trace_id",
            "query": clause,
        }


def build_sort_clause(order_by: list[str] | None) -> list[dict]:
    """Convert MLflow order_by strings to OpenSearch sort clauses.

    Args:
        order_by: List of strings like ``["start_time DESC", "name ASC"]``.

    Returns:
        List of OpenSearch sort dicts.
    """
    if not order_by:
        return []

    sort_clauses = []
    for clause in order_by:
        parts = clause.strip().split()
        field = parts[0]
        direction = "desc" if len(parts) > 1 and parts[1].upper() == "DESC" else "asc"
        sort_clauses.append({field: {"order": direction}})

    return sort_clauses
