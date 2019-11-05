import base64
import json
import sqlparse
from sqlparse.sql import Identifier, Token, Comparison, Statement
from sqlparse.tokens import Token as TokenType

from mlflow.entities import RunInfo
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

import math


class SearchUtils(object):
    VALID_METRIC_COMPARATORS = set(['>', '>=', '!=', '=', '<', '<='])
    VALID_PARAM_COMPARATORS = set(['!=', '='])
    VALID_TAG_COMPARATORS = set(['!=', '='])
    VALID_STRING_ATTRIBUTE_COMPARATORS = set(['!=', '='])
    VALID_SEARCH_ATTRIBUTE_KEYS = set(RunInfo.get_searchable_attributes())
    VALID_ORDER_BY_ATTRIBUTE_KEYS = set(RunInfo.get_orderable_attributes())
    _METRIC_IDENTIFIER = "metric"
    _ALTERNATE_METRIC_IDENTIFIERS = set(["metrics"])
    _PARAM_IDENTIFIER = "parameter"
    _ALTERNATE_PARAM_IDENTIFIERS = set(["parameters", "param", "params"])
    _TAG_IDENTIFIER = "tag"
    _ALTERNATE_TAG_IDENTIFIERS = set(["tags"])
    _ATTRIBUTE_IDENTIFIER = "attribute"
    _ALTERNATE_ATTRIBUTE_IDENTIFIERS = set(["attr", "attributes", "run"])
    _IDENTIFIERS = [_METRIC_IDENTIFIER, _PARAM_IDENTIFIER, _TAG_IDENTIFIER, _ATTRIBUTE_IDENTIFIER]
    _VALID_IDENTIFIERS = set(_IDENTIFIERS
                             + list(_ALTERNATE_METRIC_IDENTIFIERS)
                             + list(_ALTERNATE_PARAM_IDENTIFIERS)
                             + list(_ALTERNATE_TAG_IDENTIFIERS)
                             + list(_ALTERNATE_ATTRIBUTE_IDENTIFIERS))
    STRING_VALUE_TYPES = set([TokenType.Literal.String.Single])
    NUMERIC_VALUE_TYPES = set([TokenType.Literal.Number.Integer, TokenType.Literal.Number.Float])

    @classmethod
    def _trim_ends(cls, string_value):
        return string_value[1:-1]

    @classmethod
    def _is_quoted(cls, value, pattern):
        return len(value) >= 2 and value.startswith(pattern) and value.endswith(pattern)

    @classmethod
    def _trim_backticks(cls, entity_type):
        """Remove backticks from identifier like `param`, if they exist."""
        if cls._is_quoted(entity_type, "`"):
            return cls._trim_ends(entity_type)
        return entity_type

    @classmethod
    def _strip_quotes(cls, value, expect_quoted_value=False):
        """
        Remove quotes for input string.
        Values of type strings are expected to have quotes.
        Keys containing special characters are also expected to be enclose in quotes.
        """
        if cls._is_quoted(value, "'") or cls._is_quoted(value, '"'):
            return cls._trim_ends(value)
        elif expect_quoted_value:
            raise MlflowException("Parameter value is either not quoted or unidentified quote "
                                  "types used for string value %s. Use either single or double "
                                  "quotes." % value, error_code=INVALID_PARAMETER_VALUE)
        else:
            return value

    @classmethod
    def _valid_entity_type(cls, entity_type):
        entity_type = cls._trim_backticks(entity_type)
        if entity_type not in cls._VALID_IDENTIFIERS:
            raise MlflowException("Invalid entity type '%s'. "
                                  "Valid values are %s" % (entity_type, cls._IDENTIFIERS),
                                  error_code=INVALID_PARAMETER_VALUE)

        if entity_type in cls._ALTERNATE_PARAM_IDENTIFIERS:
            return cls._PARAM_IDENTIFIER
        elif entity_type in cls._ALTERNATE_METRIC_IDENTIFIERS:
            return cls._METRIC_IDENTIFIER
        elif entity_type in cls._ALTERNATE_TAG_IDENTIFIERS:
            return cls._TAG_IDENTIFIER
        elif entity_type in cls._ALTERNATE_ATTRIBUTE_IDENTIFIERS:
            return cls._ATTRIBUTE_IDENTIFIER
        else:
            # one of ("metric", "parameter", "tag", or "attribute") since it a valid type
            return entity_type

    @classmethod
    def _get_identifier(cls, identifier, valid_attributes):
        try:
            entity_type, key = identifier.split(".", 1)
        except ValueError:
            raise MlflowException("Invalid identifier '%s'. Columns should be specified as "
                                  "'attribute.<key>', 'metric.<key>', 'tag.<key>', or "
                                  "'param.'." % identifier,
                                  error_code=INVALID_PARAMETER_VALUE)
        identifier = cls._valid_entity_type(entity_type)
        key = cls._trim_backticks(cls._strip_quotes(key))
        if identifier == cls._ATTRIBUTE_IDENTIFIER and key not in valid_attributes:
            raise MlflowException("Invalid attribute key '{}' specified. Valid keys "
                                  " are '{}'".format(key, valid_attributes))
        return {"type": identifier, "key": key}

    @classmethod
    def _get_value(cls, identifier_type, token):
        if identifier_type == cls._METRIC_IDENTIFIER:
            if token.ttype not in cls.NUMERIC_VALUE_TYPES:
                raise MlflowException("Expected numeric value type for metric. "
                                      "Found {}".format(token.value),
                                      error_code=INVALID_PARAMETER_VALUE)
            return token.value
        elif identifier_type == cls._PARAM_IDENTIFIER or identifier_type == cls._TAG_IDENTIFIER:
            if token.ttype in cls.STRING_VALUE_TYPES or isinstance(token, Identifier):
                return cls._strip_quotes(token.value, expect_quoted_value=True)
            raise MlflowException("Expected a quoted string value for "
                                  "{identifier_type} (e.g. 'my-value'). Got value "
                                  "{value}".format(identifier_type=identifier_type,
                                                   value=token.value),
                                  error_code=INVALID_PARAMETER_VALUE)
        elif identifier_type == cls._ATTRIBUTE_IDENTIFIER:
            if token.ttype in cls.STRING_VALUE_TYPES or isinstance(token, Identifier):
                return cls._strip_quotes(token.value, expect_quoted_value=True)
            else:
                raise MlflowException("Expected a quoted string value for attributes. "
                                      "Got value {value}".format(value=token.value),
                                      error_code=INVALID_PARAMETER_VALUE)
        else:
            # Expected to be either "param" or "metric".
            raise MlflowException("Invalid identifier type. Expected one of "
                                  "{}.".format([cls._METRIC_IDENTIFIER, cls._PARAM_IDENTIFIER]))

    @classmethod
    def _validate_comparison(cls, tokens):
        base_error_string = "Invalid comparison clause"
        if len(tokens) != 3:
            raise MlflowException("{}. Expected 3 tokens found {}".format(base_error_string,
                                                                          len(tokens)),
                                  error_code=INVALID_PARAMETER_VALUE)
        if not isinstance(tokens[0], Identifier):
            raise MlflowException("{}. Expected 'Identifier' found '{}'".format(base_error_string,
                                                                                str(tokens[0])),
                                  error_code=INVALID_PARAMETER_VALUE)
        if not isinstance(tokens[1], Token) and tokens[1].ttype != TokenType.Operator.Comparison:
            raise MlflowException("{}. Expected comparison found '{}'".format(base_error_string,
                                                                              str(tokens[1])),
                                  error_code=INVALID_PARAMETER_VALUE)
        if not isinstance(tokens[2], Token) and \
                (tokens[2].ttype not in cls.STRING_VALUE_TYPES.union(cls.NUMERIC_VALUE_TYPES) or
                 isinstance(tokens[2], Identifier)):
            raise MlflowException("{}. Expected value token found '{}'".format(base_error_string,
                                                                               str(tokens[2])),
                                  error_code=INVALID_PARAMETER_VALUE)

    @classmethod
    def _get_comparison(cls, comparison):
        stripped_comparison = [token for token in comparison.tokens if not token.is_whitespace]
        cls._validate_comparison(stripped_comparison)
        comp = cls._get_identifier(stripped_comparison[0].value, cls.VALID_SEARCH_ATTRIBUTE_KEYS)
        comp["comparator"] = stripped_comparison[1].value
        comp["value"] = cls._get_value(comp.get("type"), stripped_comparison[2])
        return comp

    @classmethod
    def _invalid_statement_token(cls, token):
        if isinstance(token, Comparison):
            return False
        elif token.is_whitespace:
            return False
        elif token.match(ttype=TokenType.Keyword, values=["AND"]):
            return False
        else:
            return True

    @classmethod
    def _process_statement(cls, statement):
        # check validity
        invalids = list(filter(cls._invalid_statement_token, statement.tokens))
        if len(invalids) > 0:
            invalid_clauses = ", ".join("'%s'" % token for token in invalids)
            raise MlflowException("Invalid clause(s) in filter string: %s" % invalid_clauses,
                                  error_code=INVALID_PARAMETER_VALUE)
        return [cls._get_comparison(si) for si in statement.tokens if isinstance(si, Comparison)]

    @classmethod
    def _parse_search_filter(cls, filter_string):
        if not filter_string:
            return []
        try:
            parsed = sqlparse.parse(filter_string)
        except Exception:
            raise MlflowException("Error on parsing filter '%s'" % filter_string,
                                  error_code=INVALID_PARAMETER_VALUE)
        if len(parsed) == 0 or not isinstance(parsed[0], Statement):
            raise MlflowException("Invalid filter '%s'. Could not be parsed." %
                                  filter_string, error_code=INVALID_PARAMETER_VALUE)
        elif len(parsed) > 1:
            raise MlflowException("Search filter contained multiple expression '%s'. "
                                  "Provide AND-ed expression list." % filter_string,
                                  error_code=INVALID_PARAMETER_VALUE)
        return SearchUtils._process_statement(parsed[0])

    @classmethod
    def _does_run_match_clause(cls, run, sed):
        key_type = sed.get('type')
        key = sed.get('key')
        value = sed.get('value')
        comparator = sed.get('comparator')
        if key_type == cls._METRIC_IDENTIFIER:
            if comparator not in cls.VALID_METRIC_COMPARATORS:
                raise MlflowException("Invalid comparator '%s' "
                                      "not one of '%s" % (comparator,
                                                          cls.VALID_METRIC_COMPARATORS),
                                      error_code=INVALID_PARAMETER_VALUE)
            lhs = run.data.metrics.get(key, None)
            value = float(value)
        elif key_type == cls._PARAM_IDENTIFIER:
            if comparator not in cls.VALID_PARAM_COMPARATORS:
                raise MlflowException("Invalid comparator '%s' "
                                      "not one of '%s'" % (comparator, cls.VALID_PARAM_COMPARATORS),
                                      error_code=INVALID_PARAMETER_VALUE)
            lhs = run.data.params.get(key, None)
        elif key_type == cls._TAG_IDENTIFIER:
            if comparator not in cls.VALID_TAG_COMPARATORS:
                raise MlflowException("Invalid comparator '%s' "
                                      "not one of '%s" % (comparator, cls.VALID_TAG_COMPARATORS))
            lhs = run.data.tags.get(key, None)
        elif key_type == cls._ATTRIBUTE_IDENTIFIER:
            if comparator not in cls.VALID_STRING_ATTRIBUTE_COMPARATORS:
                raise MlflowException("Invalid comparator '{}' not one of "
                                      "'{}".format(comparator,
                                                   cls.VALID_STRING_ATTRIBUTE_COMPARATORS))
            lhs = getattr(run.info, key)
        else:
            raise MlflowException("Invalid search expression type '%s'" % key_type,
                                  error_code=INVALID_PARAMETER_VALUE)
        if lhs is None:
            return False
        elif comparator == '>':
            return lhs > value
        elif comparator == '>=':
            return lhs >= value
        elif comparator == '=':
            return lhs == value
        elif comparator == '!=':
            return lhs != value
        elif comparator == '<=':
            return lhs <= value
        elif comparator == '<':
            return lhs < value
        else:
            return False

    @classmethod
    def filter(cls, runs, filter_string):
        """Filters a set of runs based on a search filter string."""
        if not filter_string:
            return runs
        parsed = cls._parse_search_filter(filter_string)

        def run_matches(run):
            return all([cls._does_run_match_clause(run, s) for s in parsed])

        return [run for run in runs if run_matches(run)]

    @classmethod
    def _parse_order_by(cls, order_by):
        try:
            parsed = sqlparse.parse(order_by)
        except Exception:
            raise MlflowException("Error on parsing order_by clause '%s'" % order_by,
                                  error_code=INVALID_PARAMETER_VALUE)
        if len(parsed) != 1 or not isinstance(parsed[0], Statement):
            raise MlflowException("Invalid order_by clause '%s'. Could not be parsed." %
                                  order_by, error_code=INVALID_PARAMETER_VALUE)

        statement = parsed[0]
        if len(statement.tokens) != 1 or not isinstance(statement[0], Identifier):
            raise MlflowException("Invalid order_by clause '%s'. Could not be parsed." %
                                  order_by, error_code=INVALID_PARAMETER_VALUE)

        token_value = statement.tokens[0].value
        is_ascending = True
        if token_value.lower().endswith(" desc"):
            is_ascending = False
            token_value = token_value[0:-len(" desc")]
        elif token_value.lower().endswith(" asc"):
            token_value = token_value[0:-len(" asc")]
        identifier = cls._get_identifier(token_value.strip(), cls.VALID_ORDER_BY_ATTRIBUTE_KEYS)
        return (identifier["type"], identifier["key"], is_ascending)

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
            raise MlflowException("Invalid order_by entity type '%s'" % key_type,
                                  error_code=INVALID_PARAMETER_VALUE)

        # Return a key such that None values are always at the end.
        is_null_or_nan = sort_value is None or (isinstance(sort_value, float)
                                                and math.isnan(sort_value))
        if ascending:
            return (is_null_or_nan, sort_value)
        return (not is_null_or_nan, sort_value)

    @classmethod
    def sort(cls, runs, order_by_list):
        """Sorts a set of runs based on their natural ordering and an overriding set of order_bys.
        Runs are naturally ordered first by start time descending, then by run id for tie-breaking.
        """
        runs = sorted(runs, key=lambda run: (-run.info.start_time, run.info.run_uuid))
        if not order_by_list:
            return runs
        # NB: We rely on the stability of Python's sort function, so that we can apply
        # the ordering conditions in reverse order.
        for order_by_clause in reversed(order_by_list):
            (key_type, key, ascending) = cls._parse_order_by(order_by_clause)
            # pylint: disable=cell-var-from-loop
            runs = sorted(runs,
                          key=lambda run: cls._get_value_for_sort(run, key_type, key, ascending),
                          reverse=not ascending)
        return runs

    @classmethod
    def _parse_start_offset_from_page_token(cls, page_token):
        # Note: the page_token is expected to be a base64-encoded JSON that looks like
        # { "offset": xxx }. However, this format is not stable, so it should not be
        # relied upon outside of this method.
        if not page_token:
            return 0

        try:
            decoded_token = base64.b64decode(page_token)
        except TypeError:
            raise MlflowException("Invalid page token, could not base64-decode",
                                  error_code=INVALID_PARAMETER_VALUE)
        except base64.binascii.Error:
            raise MlflowException("Invalid page token, could not base64-decode",
                                  error_code=INVALID_PARAMETER_VALUE)

        try:
            parsed_token = json.loads(decoded_token)
        except ValueError:
            raise MlflowException("Invalid page token, decoded value=%s" % decoded_token,
                                  error_code=INVALID_PARAMETER_VALUE)

        offset_str = parsed_token.get("offset")
        if not offset_str:
            raise MlflowException("Invalid page token, parsed value=%s" % parsed_token,
                                  error_code=INVALID_PARAMETER_VALUE)

        try:
            offset = int(offset_str)
        except ValueError:
            raise MlflowException("Invalid page token, not stringable %s" % offset_str,
                                  error_code=INVALID_PARAMETER_VALUE)

        return offset

    @classmethod
    def _create_page_token(cls, offset):
        return base64.b64encode(json.dumps({"offset": offset}).encode("utf-8"))

    @classmethod
    def paginate(cls, runs, page_token, max_results):
        """Paginates a set of runs based on an offset encoded into the page_token and a max
        results limit. Returns a pair containing the set of paginated runs, followed by
        an optional next_page_token if there are further results that need to be returned.
        """
        start_offset = cls._parse_start_offset_from_page_token(page_token)
        final_offset = start_offset + max_results

        paginated_runs = runs[start_offset:final_offset]
        next_page_token = None
        if final_offset < len(runs):
            next_page_token = cls._create_page_token(final_offset)
        return (paginated_runs, next_page_token)

    # Model Registry specific parser
    # TODO: Tech debt. Refactor search code into common utils, tracking server, and model
    #       registry specific code.

    VALID_SEARCH_KEYS_FOR_MODEL_REGISTRY = set(["name", "run_id", "source_path"])

    @classmethod
    def _get_comparison_for_model_registry(cls, comparison):
        stripped_comparison = [token for token in comparison.tokens if not token.is_whitespace]
        cls._validate_comparison(stripped_comparison)
        key = stripped_comparison[0].value
        if key not in cls.VALID_SEARCH_KEYS_FOR_MODEL_REGISTRY:
            raise MlflowException("Invalid attribute key '{}' specified. Valid keys "
                                  " are '{}'".format(key, cls.VALID_SEARCH_KEYS_FOR_MODEL_REGISTRY))
        value_token = stripped_comparison[2]
        if value_token.ttype not in cls.STRING_VALUE_TYPES:
            raise MlflowException("Expected a quoted string value for attributes. "
                                  "Got value {value}".format(value=value_token.value),
                                  error_code=INVALID_PARAMETER_VALUE)
        comp = {
            "key": key,
            "comparator": stripped_comparison[1].value,
            "value": cls._strip_quotes(value_token.value, expect_quoted_value=True)
        }
        return comp

    @classmethod
    def parse_filter_for_model_registry(cls, filter_string):
        if not filter_string or filter_string == "":
            return []
        expected = "Expected search filter with single comparison operator. e.g. name='myModelName'"
        try:
            parsed = sqlparse.parse(filter_string)
        except Exception:
            raise MlflowException("Error while parsing filter '%s'. %s" % (filter_string, expected),
                                  error_code=INVALID_PARAMETER_VALUE)
        if len(parsed) == 0 or not isinstance(parsed[0], Statement):
            raise MlflowException("Invalid filter '%s'. Could not be parsed. %s" %
                                  (filter_string, expected), error_code=INVALID_PARAMETER_VALUE)
        elif len(parsed) > 1:
            raise MlflowException("Search filter '%s' contains multiple expressions. "
                                  "%s " % (filter_string, expected),
                                  error_code=INVALID_PARAMETER_VALUE)
        statement = parsed[0]
        invalids = list(filter(cls._invalid_statement_token, statement.tokens))
        if len(invalids) > 0:
            invalid_clauses = ", ".join("'%s'" % token for token in invalids)
            raise MlflowException("Invalid clause(s) in filter string: %s. "
                                  "%s" % (invalid_clauses, expected),
                                  error_code=INVALID_PARAMETER_VALUE)
        return [cls._get_comparison_for_model_registry(si)
                for si in statement.tokens if isinstance(si, Comparison)]
