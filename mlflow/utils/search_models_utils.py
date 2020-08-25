from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.search_utils import SearchUtils
from mlflow.store.model_registry.dbmodels.models import (
    SqlRegisteredModel,
    SqlRegisteredModelTag,
)
import sqlalchemy.sql.expression as sql
import sqlparse
from sqlparse.sql import Comparison, Statement


class SearchModelsUtils(SearchUtils):
    _ALTERNATE_MODEL_ATTRIBUTE_IDENTIFIERS = set(
        ["attr", "attributes", "model", "registered_model", "models", "registered_models"]
    )
    _ALTERNATE_MODEL_TAG_IDENTIFIERS = set(
        ["tags", "model_tag", "registered_model_tag", "model_tags", "registered_model_tags"]
    )
    _MODEL_IDENTIFIERS = [SearchUtils._TAG_IDENTIFIER, SearchUtils._ATTRIBUTE_IDENTIFIER]
    _VALID_MODEL_IDENTIFIERS = set(
        _MODEL_IDENTIFIERS
        + list(_ALTERNATE_MODEL_ATTRIBUTE_IDENTIFIERS)
        + list(_ALTERNATE_MODEL_TAG_IDENTIFIERS)
    )
    # Registered Models Constants
    VALID_SEARCH_REGISTERED_MODEL_ATTRIBUTE_KEYS = set(["name"])
    VALID_ORDER_BY_REGISTERED_MODEL_ATTRIBUTE_KEYS = set(
        [SearchUtils.ORDER_BY_KEY_TIMESTAMP, "last_updated_timestamp", "name"]
    )
    VALID_SEARCH_MODEL_VERSION_ATTRIBUTE_KEYS = set(["name", "run_id", "source_path"])
    VALID_TIMESTAMP_ORDER_BY_KEYS = set(
        [SearchUtils.ORDER_BY_KEY_TIMESTAMP, "last_updated_timestamp"]
    )
    VALID_SEARCH_KEYS_FOR_MODEL_VERSIONS = set(["name", "run_id", "source_path"])
    VALID_SEARCH_KEYS_FOR_REGISTERED_MODELS = set(["name"])

    @classmethod
    def _valid_model_entity_type(cls, entity_type):
        entity_type = cls._trim_backticks(entity_type)
        if entity_type not in cls._VALID_MODEL_IDENTIFIERS:
            raise MlflowException(
                "Invalid entity type '%s'. "
                "Valid values are %s" % (entity_type, cls._MODEL_IDENTIFIERS),
                error_code=INVALID_PARAMETER_VALUE,
            )
        if entity_type in cls._ALTERNATE_MODEL_ATTRIBUTE_IDENTIFIERS:
            return cls._ATTRIBUTE_IDENTIFIER
        elif entity_type in cls._ALTERNATE_MODEL_TAG_IDENTIFIERS:
            return cls._TAG_IDENTIFIER
        else:
            return entity_type

    @classmethod
    def _get_identifier_for_registered_models(cls, identifier, valid_attribute_keys):
        # search models may not provide table name, in that case default to attribute table
        # (e.g name = 'CNN' is equivalent to attribute.model = 'CNN')
        column_only = identifier.find(".") == -1
        entity_type, key = cls._split_identifier(identifier, column_only)
        identifier = cls._valid_model_entity_type(entity_type)
        key = cls._trim_backticks(cls._strip_quotes(key))
        if identifier == cls._ATTRIBUTE_IDENTIFIER and key not in valid_attribute_keys:
            raise MlflowException(
                "Invalid attribute key '{}' specified. Valid keys"
                " are {}".format(key, valid_attribute_keys),
                error_code=INVALID_PARAMETER_VALUE,
            )
        return {"type": identifier, "key": key}

    @classmethod
    def _get_comparison_for_registered_models(cls, comparison):
        stripped_comparison = [token for token in comparison.tokens if not token.is_whitespace]
        cls._validate_comparison(stripped_comparison)
        comp = cls._get_identifier_for_registered_models(
            stripped_comparison[0].value, cls.VALID_SEARCH_REGISTERED_MODEL_ATTRIBUTE_KEYS
        )
        comp["comparator"] = stripped_comparison[1].value
        comp["value"] = cls._get_value(comp.get("type"), stripped_comparison[2])
        return comp

    @classmethod
    def parse_filter_for_registered_models(cls, filter_string):
        if not filter_string:
            return []
        statement = cls._validate_and_get_sql_statement(filter_string)[0]
        cls._validate_statement_tokens(statement)
        return [
            cls._get_comparison_for_registered_models(si)
            for si in statement.tokens
            if isinstance(si, Comparison)
        ]

    @classmethod
    def parse_order_by_for_search_registered_models(cls, order_by):
        token_value, is_ascending = cls._parse_order_by_string(order_by)
        identifier = cls._get_identifier_for_registered_models(
            token_value.strip(), cls.VALID_ORDER_BY_REGISTERED_MODEL_ATTRIBUTE_KEYS
        )
        # last updated timestamp field in search registered model has alias,
        # we want to map those alias to the correct db key
        if identifier["key"] in cls.VALID_TIMESTAMP_ORDER_BY_KEYS:
            identifier["key"] = SqlRegisteredModel.last_updated_time.key
        return identifier["type"], identifier["key"], is_ascending

    @classmethod
    def get_order_by_clauses_for_registered_model(cls, order_by_list, session):
        """
        Sorts a set of registered_models based on their natural ordering
        and an overriding set of order_bys.
        Models are naturally ordered by model name for tie-breaking.
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
                (key_type, key, ascending) = cls.parse_order_by_for_search_registered_models(
                    order_by_clause
                )
                if cls.is_attribute(key_type, "="):
                    order_value = getattr(SqlRegisteredModel, key)
                    clauses.append(
                        sql.case([(order_value.is_(None), 1)], else_=0).label(
                            "clause_%s" % clause_id
                        )
                    )
                else:
                    entity = cls._get_subquery_entity_for_registered_models(key_type, "=")
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
            SqlRegisteredModel.name.key,
        ) not in observed_order_by_clauses:
            clauses.append(SqlRegisteredModel.name.asc())

        return clauses, ordering_joins

    @classmethod
    def _get_subquery_entity_for_registered_models(cls, key_type, comparator):
        if cls.is_tag(key_type, comparator):
            return SqlRegisteredModelTag
        elif cls.is_attribute(key_type, comparator):
            return None
        else:
            raise MlflowException(
                "Invalid identifier type '%s'" % key_type, error_code=INVALID_PARAMETER_VALUE,
            )

    @classmethod
    def get_sqlalchemy_filter_clause_for_registered_model(cls, parsed, session):
        """creates SqlAlchemy sub-queries
        that will be inner-joined to attribute table to act as multi-clause filters."""
        filters = []
        for sql_statement in parsed:
            key_type, key_name, value, comparator = cls._parse_sql_statement(sql_statement)
            if not SearchUtils.is_attribute(key_type, comparator):
                entity = cls._get_subquery_entity_for_registered_models(key_type, comparator)
                filter_query = cls._get_sqlalchemy_query(
                    entity, comparator, key_name, value, session
                )
                filters.append(filter_query)
        return filters

    @classmethod
    def get_attributes_filtering_clauses_for_registered_model(cls, parsed):
        clauses = []
        for sql_statement in parsed:
            key_type, key_name, value, comparator = cls._parse_sql_statement(sql_statement)
            if cls.is_attribute(key_type, comparator):
                # key_name is guaranteed to be a valid searchable attribute of SqlRegisteredModel
                # by the call to parse_search_filter
                attribute = getattr(SqlRegisteredModel, key_name)
                if comparator in cls.CASE_INSENSITIVE_STRING_COMPARISON_OPERATORS:
                    op = cls.get_sql_filter_ops(attribute, comparator)
                    clauses.append(op(value))
                elif comparator in cls.filter_ops:
                    op = cls.filter_ops.get(comparator)
                    clauses.append(op(attribute, value))
        return clauses

    # TODO: refactor search_model_version and add tags support
    #       remove _parse_filter_for_model_registry, get_comparison_for_model_registry

    @classmethod
    def _get_comparison_for_model_registry(cls, comparison, valid_search_keys):
        stripped_comparison = [token for token in comparison.tokens if not token.is_whitespace]
        cls._validate_comparison(stripped_comparison)
        key = stripped_comparison[0].value
        if key not in valid_search_keys:
            raise MlflowException(
                "Invalid attribute key '{}' specified. Valid keys "
                " are '{}'".format(key, valid_search_keys),
                error_code=INVALID_PARAMETER_VALUE,
            )
        value_token = stripped_comparison[2]
        if value_token.ttype not in cls.STRING_VALUE_TYPES:
            raise MlflowException(
                "Expected a quoted string value for attributes. "
                "Got value {value}".format(value=value_token.value),
                error_code=INVALID_PARAMETER_VALUE,
            )
        comp = {
            "key": key,
            "comparator": stripped_comparison[1].value,
            "value": cls._strip_quotes(value_token.value, expect_quoted_value=True),
        }
        return comp

    @classmethod
    def _parse_filter_for_model_registry(cls, filter_string, valid_search_keys):
        if not filter_string or filter_string == "":
            return []
        expected = "Expected search filter with single comparison operator. e.g. name='myModelName'"
        try:
            parsed = sqlparse.parse(filter_string)
        except Exception:
            raise MlflowException(
                "Error while parsing filter '%s'. %s" % (filter_string, expected),
                error_code=INVALID_PARAMETER_VALUE,
            )
        if len(parsed) == 0 or not isinstance(parsed[0], Statement):
            raise MlflowException(
                "Invalid filter '%s'. Could not be parsed. %s" % (filter_string, expected),
                error_code=INVALID_PARAMETER_VALUE,
            )
        elif len(parsed) > 1:
            raise MlflowException(
                "Search filter '%s' contains multiple expressions. "
                "%s " % (filter_string, expected),
                error_code=INVALID_PARAMETER_VALUE,
            )
        statement = parsed[0]
        invalids = list(filter(cls._invalid_statement_token, statement.tokens))
        if len(invalids) > 0:
            invalid_clauses = ", ".join("'%s'" % token for token in invalids)
            raise MlflowException(
                "Invalid clause(s) in filter string: %s. " "%s" % (invalid_clauses, expected),
                error_code=INVALID_PARAMETER_VALUE,
            )
        return [
            cls._get_comparison_for_model_registry(si, valid_search_keys)
            for si in statement.tokens
            if isinstance(si, Comparison)
        ]

    @classmethod
    def parse_filter_for_model_versions(cls, filter_string):
        return cls._parse_filter_for_model_registry(
            filter_string, cls.VALID_SEARCH_KEYS_FOR_MODEL_VERSIONS
        )
