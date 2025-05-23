import ast
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import sqlalchemy
import sqlparse

from mlflow.exceptions import MlflowException
from mlflow.store.tracking.dbmodels.models import SqlLoggedModel
from mlflow.utils.search_utils import _join_in_comparison_tokens


class EntityType(Enum):
    ATTRIBUTE = "attributes"
    METRIC = "metrics"
    PARAM = "params"
    TAG = "tags"

    @classmethod
    def from_str(cls, s: str) -> "EntityType":
        if s == "attributes":
            return cls.ATTRIBUTE
        if s == "metrics":
            return cls.METRIC
        if s == "params":
            return cls.PARAM
        if s == "tags":
            return cls.TAG

        raise MlflowException.invalid_parameter_value(
            f"Invalid entity type: {s!r}. Expected one of {[e.value for e in cls]}."
        )


@dataclass
class Entity:
    type: EntityType
    key: str

    IDENTIFIER_RE = re.compile(r"^([a-z]+)\.(.+)$")

    def __repr__(self) -> str:
        return f"{self.type.value}.{self.key}"

    @classmethod
    def from_str(cls, s: str) -> "Entity":
        if m := Entity.IDENTIFIER_RE.match(s):
            return cls(
                type=EntityType.from_str(m.group(1)),
                key=m.group(2).strip("`"),
            )
        return cls(type=EntityType.ATTRIBUTE, key=SqlLoggedModel.ALIASES.get(s, s).strip("`"))

    def is_numeric(self) -> bool:
        """
        Does this entity represent a numeric column?
        """
        return self.type == EntityType.METRIC or (
            self.type == EntityType.ATTRIBUTE and SqlLoggedModel.is_numeric(self.key)
        )

    def validate_op(self, op: str) -> None:
        numeric_ops = ("<", "<=", ">", ">=", "=", "!=")
        string_ops = ("=", "!=", "LIKE", "ILIKE", "IN", "NOT IN")
        ops = numeric_ops if self.is_numeric() else string_ops
        if op not in ops:
            raise MlflowException.invalid_parameter_value(
                f"Invalid comparison operator for {self}: {op!r}. Expected one of {string_ops}."
            )


@dataclass
class Comparison:
    entity: Entity
    op: str
    value: Union[str, float]


def parse_filter_string(filter_string: Optional[str]) -> list[Comparison]:
    if not filter_string:
        return []
    try:
        parsed = sqlparse.parse(filter_string)
    except Exception as e:
        raise MlflowException.invalid_parameter_value(
            f"Invalid filter string: {filter_string!r}. {e!r}"
        ) from e

    if len(parsed) != 1:
        raise MlflowException.invalid_parameter_value(
            f"Invalid filter string: {filter_string!r}. Expected a single SQL expression.",
        )

    comparisons: list[sqlalchemy.BinaryExpression] = []
    for stmt in _join_in_comparison_tokens(parsed[0].tokens):
        # while index < len(statements):
        if isinstance(stmt, sqlparse.sql.Comparison):
            non_whitespace_tokens = [str(t) for t in stmt.tokens if not t.is_whitespace]
            if len(non_whitespace_tokens) != 3:
                raise MlflowException.invalid_parameter_value(
                    f"Invalid comparison: {stmt}. Expected a comparison with 3 tokens."
                )
            identifier, op, value = non_whitespace_tokens
            entity = Entity.from_str(identifier)
            entity.validate_op(op)
            value = float(value) if entity.is_numeric() else value.strip("'")
            if entity.is_numeric():
                value = float(value)
            else:
                if value.startswith("(") and value.endswith(")"):
                    value = ast.literal_eval(value)
                    value = (value,) if isinstance(value, str) else value
                else:
                    value = value.strip("'")
            comparisons.append(Comparison(entity=entity, op=op, value=value))
        elif stmt.value.strip().upper() == "AND":
            # Do nothing, this is just a separator
            pass
        else:
            raise MlflowException.invalid_parameter_value(
                f"Invalid filter string: {filter_string!r}. Expected a list of comparisons "
                f"separated by 'AND' (e.g. 'metrics.loss > 0.1 AND params.lr = 0.01')."
            )

    return comparisons
