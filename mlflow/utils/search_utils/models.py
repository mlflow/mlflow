from enum import Enum


class KeyType(Enum):
    METRIC = "metric"
    PARAM = "param"
    TAG = "tag"


class ComparisonOperator(Enum):
    EQUAL = "="
    NOT_EQUAL = "!="
    GREATER_THAN = ">"
    GREATER_THAN_EQUAL = ">="
    LESS_THAN = "<"
    LESS_THAN_EQUAL = "<="


VALID_OPERATORS_FOR_KEY_TYPE = {
    KeyType.METRIC: set(ComparisonOperator),
    KeyType.PARAM: {ComparisonOperator.EQUAL, ComparisonOperator.NOT_EQUAL},
    KeyType.TAG: {ComparisonOperator.EQUAL, ComparisonOperator.NOT_EQUAL},
}


def _validate_operator(key_type, operator):
    valid_operators = VALID_OPERATORS_FOR_KEY_TYPE[key_type]
    if operator not in valid_operators:
        raise ValueError("The operator '{}' is not supported for {}s - "
                         "must be one of {}".format(operator.value, key_type.value,
                                                    {o.value for o in valid_operators}))


def _get_run_value(run, key_type, key):
    if key_type == KeyType.METRIC:
        entities_to_search = run.data.metrics
    elif key_type == KeyType.PARAM:
        entities_to_search = run.data.params
    elif key_type == KeyType.TAG:
        entities_to_search = run.data.tags
    else:
        raise ValueError("Invalid key type: {}".format(key_type))

    matching_entity = next((e for e in entities_to_search if e.key == key), None)
    return matching_entity.value if matching_entity else None


class Comparison(object):
    def __init__(self, key_type, key, operator, value):
        _validate_operator(key_type, operator)
        self.key_type = key_type
        self.key = key
        self.operator = operator
        self.value = float(value) if self.key_type == KeyType.METRIC else value

    def __eq__(self, other):
        if not isinstance(other, Comparison):
            return False
        return (self.key_type == other.key_type and self.key == other.key and
                self.operator == other.operator and self.value == other.value)

    def __repr__(self):
        return "{}({}, {}, {}, {})".format(self.__class__.__name__, self.key_type, self.key,
                                           self.operator, self.value)

    def filter(self, run):
        lhs = _get_run_value(run, self.key_type, self.key)
        if lhs is None:
            return False
        elif self.operator == ComparisonOperator.GREATER_THAN:
            return lhs > self.value
        elif self.operator == ComparisonOperator.GREATER_THAN_EQUAL:
            return lhs >= self.value
        elif self.operator == ComparisonOperator.EQUAL:
            return lhs == self.value
        elif self.operator == ComparisonOperator.NOT_EQUAL:
            return lhs != self.value
        elif self.operator == ComparisonOperator.LESS_THAN_EQUAL:
            return lhs <= self.value
        elif self.operator == ComparisonOperator.LESS_THAN:
            return lhs < self.value
        else:
            return False
