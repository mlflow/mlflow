from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.search_utils import SearchUtils
from mlflow.entities import RunInfo
from sqlparse.sql import Comparison
from mlflow.store.tracking.dbmodels.models import (
    SqlRun,
    SqlParam,
    SqlTag,
    SqlLatestMetric,
)
import sqlalchemy.sql.expression as sql
import math


class SearchRunsUtils(SearchUtils):
    VALID_SEARCH_RUN_ATTRIBUTE_KEYS = set(RunInfo.get_searchable_attributes())
    VALID_ORDER_BY_RUN_ATTRIBUTE_KEYS = set(RunInfo.get_orderable_attributes())
    _ALTERNATE_RUN_ATTRIBUTE_IDENTIFIERS = set(["attr", "attributes", "run"])
    _ALTERNATE_RUN_TAG_IDENTIFIERS = set(["tags"])
    _RUN_IDENTIFIERS = [
        SearchUtils._METRIC_IDENTIFIER,
        SearchUtils._PARAM_IDENTIFIER,
        SearchUtils._TAG_IDENTIFIER,
        SearchUtils._ATTRIBUTE_IDENTIFIER,
    ]
    _VALID_RUN_IDENTIFIERS = set(
        _RUN_IDENTIFIERS
        + list(SearchUtils._ALTERNATE_METRIC_IDENTIFIERS)
        + list(SearchUtils._ALTERNATE_PARAM_IDENTIFIERS)
        + list(_ALTERNATE_RUN_TAG_IDENTIFIERS)
        + list(_ALTERNATE_RUN_ATTRIBUTE_IDENTIFIERS)
    )

    @classmethod
    def _valid_run_entity_type(cls, entity_type):
        entity_type = SearchUtils._trim_backticks(entity_type)
        if entity_type not in cls._VALID_RUN_IDENTIFIERS:
            raise MlflowException(
                "Invalid entity type '%s'. "
                "Valid values are %s" % (entity_type, cls._RUN_IDENTIFIERS),
                error_code=INVALID_PARAMETER_VALUE,
            )

        if entity_type in cls._ALTERNATE_PARAM_IDENTIFIERS:
            return cls._PARAM_IDENTIFIER
        elif entity_type in cls._ALTERNATE_METRIC_IDENTIFIERS:
            return cls._METRIC_IDENTIFIER
        elif entity_type in cls._ALTERNATE_RUN_TAG_IDENTIFIERS:
            return cls._TAG_IDENTIFIER
        elif entity_type in cls._ALTERNATE_RUN_ATTRIBUTE_IDENTIFIERS:
            return cls._ATTRIBUTE_IDENTIFIER
        else:
            # one of ("metric", "parameter", "tag", or "attribute") since it a valid type
            return entity_type

    @classmethod
    def _get_identifier_for_runs(cls, identifier, valid_attribute_keys):
        # search models may not provide table name, in that case default to attribute table
        # (e.g name = 'CNN' is equivalent to attribute.model = 'CNN')
        column_only = identifier.find(".") == -1
        entity_type, key = cls._split_identifier(identifier, column_only)
        identifier = cls._valid_run_entity_type(entity_type)
        key = cls._trim_backticks(cls._strip_quotes(key))
        if identifier == cls._ATTRIBUTE_IDENTIFIER and key not in valid_attribute_keys:
            raise MlflowException(
                "Invalid attribute key '{}' specified. Valid keys"
                " are {}".format(key, valid_attribute_keys),
                error_code=INVALID_PARAMETER_VALUE,
            )
        return {"type": identifier, "key": key}

    @classmethod
    def _get_comparison_for_runs(cls, comparison):
        stripped_comparison = [token for token in comparison.tokens if not token.is_whitespace]
        cls._validate_comparison(stripped_comparison)
        comp = cls._get_identifier_for_runs(
            stripped_comparison[0].value, cls.VALID_SEARCH_RUN_ATTRIBUTE_KEYS
        )
        comp["comparator"] = stripped_comparison[1].value
        comp["value"] = cls._get_value(comp.get("type"), stripped_comparison[2])
        return comp

    @classmethod
    def parse_filter_for_run(cls, filter_string):
        if not filter_string:
            return []
        statement = cls._validate_and_get_sql_statement(filter_string)[0]
        cls._validate_statement_tokens(statement)
        return [
            cls._get_comparison_for_runs(si)
            for si in statement.tokens
            if isinstance(si, Comparison)
        ]

    @classmethod
    def _does_run_match_clause(cls, run, sed):
        key_type = sed.get("type")
        key = sed.get("key")
        value = sed.get("value")
        comparator = sed.get("comparator").upper()

        if cls.is_metric(key_type, comparator):
            lhs = run.data.metrics.get(key, None)
            value = float(value)
        elif cls.is_param(key_type, comparator):
            lhs = run.data.params.get(key, None)
        elif cls.is_tag(key_type, comparator):
            lhs = run.data.tags.get(key, None)
        elif cls.is_attribute(key_type, comparator):
            lhs = getattr(run.info, key)
        else:
            raise MlflowException(
                "Invalid search expression type '%s'" % key_type,
                error_code=INVALID_PARAMETER_VALUE,
            )
        if lhs is None:
            return False

        if comparator in cls.CASE_INSENSITIVE_STRING_COMPARISON_OPERATORS:
            # Change value from sql syntax to regex syntax
            if comparator == "ILIKE":
                value = value.lower()
                lhs = lhs.lower()
            if not value.startswith("%"):
                value = "^" + value
            if not value.endswith("%"):
                value = value + "$"
            value = value.replace("_", ".").replace("%", ".*")
            return cls.filter_ops.get(comparator)(value, lhs)

        elif comparator in cls.filter_ops.keys():
            return cls.filter_ops.get(comparator)(lhs, value)
        else:
            return False

    @classmethod
    def filter_runs(cls, runs, filter_string):
        """Filters a set of runs based on a search filter string."""
        if not filter_string:
            return runs
        parsed = cls.parse_filter_for_run(filter_string)

        def run_matches(run):
            return all([cls._does_run_match_clause(run, s) for s in parsed])

        return [run for run in runs if run_matches(run)]

    @classmethod
    def parse_order_by_for_search_runs(cls, order_by):
        token_value, is_ascending = cls._parse_order_by_string(order_by)
        identifier = cls._get_identifier_for_runs(
            token_value.strip(), cls.VALID_ORDER_BY_RUN_ATTRIBUTE_KEYS
        )
        return identifier["type"], identifier["key"], is_ascending

    @classmethod
    def get_order_by_clauses_for_run(cls, order_by_list, session):
        """Sorts a set of runs based on their natural ordering and an overriding set of order_bys.
        Runs are naturally ordered first by start time descending, then by run id for tie-breaking.
        """
        clauses = []
        ordering_joins = []
        clause_id = 0
        observed_order_by_clauses = set()
        # contrary to filters, it is not easily feasible to separately handle sorting
        # on attributes and on joined tables as we must keep all clauses in the same order
        if order_by_list:
            for order_by_clause in order_by_list:
                clause_id += 1
                key_type, key, ascending = cls.parse_order_by_for_search_runs(order_by_clause)

                if cls.is_attribute(key_type, "="):
                    order_value = getattr(SqlRun, SqlRun.get_attribute_name(key))
                    clauses.append(
                        sql.case([(order_value.is_(None), 1)], else_=0).label(
                            "clause_%s" % clause_id
                        )
                    )
                else:
                    entity = cls._get_subquery_entity_for_runs(key_type, "=")
                    # build a subquery first because we will join it in the main request so that the
                    # metric we want to sort on is available when we apply the sorting clause
                    subquery = session.query(entity).filter(entity.key == key).subquery()
                    ordering_joins.append(subquery)
                    order_value = subquery.c.value
                    # sqlite does not support NULLS LAST expression, so we sort first by
                    # presence of the field (and is_nan for metrics), then by actual value
                    # As the subqueries are created independently and used later in the
                    # same main query, the CASE WHEN columns need to have unique names to
                    # avoid ambiguity
                    if cls.is_metric(key_type, "="):
                        clauses.append(
                            sql.case(
                                [(subquery.c.is_nan.is_(True), 1), (order_value.is_(None), 1)],
                                else_=0,
                            ).label("clause_%s" % clause_id)
                        )
                    else:  # other entities do not have an 'is_nan' field
                        clauses.append(
                            sql.case([(order_value.is_(None), 1)], else_=0).label(
                                "clause_%s" % clause_id
                            )
                        )

                cls._check_for_duplicate_order_by_clause(
                    observed_order_by_clauses, order_by_list, key_type, key
                )
                if ascending:
                    clauses.append(order_value)
                else:
                    clauses.append(order_value.desc())

        if (
            SearchUtils._ATTRIBUTE_IDENTIFIER,
            SqlRun.start_time.key,
        ) not in observed_order_by_clauses:
            clauses.append(SqlRun.start_time.desc())
        clauses.append(SqlRun.run_uuid)

        return clauses, ordering_joins

    @classmethod
    def _get_subquery_entity_for_runs(cls, key_type, comparator):
        if cls.is_metric(key_type, comparator):  # any valid comparator
            return SqlLatestMetric
        elif cls.is_tag(key_type, comparator):
            return SqlTag
        elif cls.is_param(key_type, comparator):
            return SqlParam
        elif cls.is_attribute(key_type, comparator):
            return None
        else:
            raise MlflowException(
                "Invalid identifier type '%s'" % key_type, error_code=INVALID_PARAMETER_VALUE,
            )

    @classmethod
    def get_sqlalchemy_filter_clause_for_run(cls, parsed, session):
        """creates SqlAlchemy sub-queries
        that will be inner-joined to attribute table to act as multi-clause filters."""
        filters = []
        for sql_statement in parsed:
            key_type, key_name, value, comparator = cls._parse_sql_statement(sql_statement)
            if not SearchUtils.is_attribute(key_type, comparator):
                entity = cls._get_subquery_entity_for_runs(key_type, comparator)
                # force cast float type for metric value
                if cls.is_metric(key_type, comparator):
                    value = float(value)
                filter_query = cls._get_sqlalchemy_query(
                    entity, comparator, key_name, value, session
                )
                filters.append(filter_query)
        return filters

    @classmethod
    def get_attributes_filtering_clauses_for_run(cls, parsed):
        clauses = []
        for sql_statement in parsed:
            key_type, key_name, value, comparator = cls._parse_sql_statement(sql_statement)
            if cls.is_attribute(key_type, comparator):
                # key_name is guaranteed to be a valid searchable attribute of entities.RunInfo
                # by the call to parse_search_filter
                attribute = getattr(SqlRun, SqlRun.get_attribute_name(key_name))
                if comparator in cls.CASE_INSENSITIVE_STRING_COMPARISON_OPERATORS:
                    op = cls.get_sql_filter_ops(attribute, comparator)
                    clauses.append(op(value))
                elif comparator in cls.filter_ops:
                    op = cls.filter_ops.get(comparator)
                    clauses.append(op(attribute, value))
        return clauses

    @classmethod
    def _get_value_for_sort(cls, run, key_type, key, ascending):
        """Returns a tuple suitable to be used as a sort key for runs."""
        sort_value = None
        if key_type == cls._METRIC_IDENTIFIER:
            sort_value = run.data.metrics.get(key)
        elif key_type == cls._PARAM_IDENTIFIER:
            sort_value = run.data.params.get(key)
        elif key_type == cls._TAG_IDENTIFIER:
            sort_value = run.data.tags.get(key)
        elif key_type == cls._ATTRIBUTE_IDENTIFIER:
            sort_value = getattr(run.info, key)
        else:
            raise MlflowException(
                "Invalid order_by entity type '%s'" % key_type, error_code=INVALID_PARAMETER_VALUE,
            )

        # Return a key such that None values are always at the end.
        is_null_or_nan = sort_value is None or (
            isinstance(sort_value, float) and math.isnan(sort_value)
        )
        if ascending:
            return (is_null_or_nan, sort_value)
        return (not is_null_or_nan, sort_value)

    @classmethod
    def sort_runs(cls, runs, order_by_list):
        """Sorts a set of runs based on their natural ordering and an overriding set of order_bys.
        Runs are naturally ordered first by start time descending, then by run id for tie-breaking.
        """
        runs = sorted(runs, key=lambda run: (-run.info.start_time, run.info.run_uuid))
        if not order_by_list:
            return runs
        # NB: We rely on the stability of Python's sort function, so that we can apply
        # the ordering conditions in reverse order.
        for order_by_clause in reversed(order_by_list):
            (key_type, key, ascending) = cls.parse_order_by_for_search_runs(order_by_clause)
            # pylint: disable=cell-var-from-loop
            runs = sorted(
                runs,
                key=lambda run: cls._get_value_for_sort(run, key_type, key, ascending),
                reverse=not ascending,
            )
        return runs

    @classmethod
    def paginate_runs(cls, runs, page_token, max_results):
        """Paginates a set of runs based on an offset encoded into the page_token and a max
        results limit. Returns a pair containing the set of paginated runs, followed by
        an optional next_page_token if there are further results that need to be returned.
        """
        start_offset = cls.parse_start_offset_from_page_token(page_token)
        final_offset = start_offset + max_results

        paginated_runs = runs[start_offset:final_offset]
        next_page_token = None
        if final_offset < len(runs):
            next_page_token = cls.create_page_token(final_offset)
        return paginated_runs, next_page_token
