"""Built-in ensemble functions for ``make_scorer_ensemble``.

Each function receives the list of per-sub-scorer values and returns a single
``Feedback``. The parameter is named ``values`` on purpose: ``make_scorer_ensemble``
introspects the parameter name to decide whether to pass raw values or full
``Feedback`` objects (a parameter named ``feedbacks`` opts into the latter).
"""

from collections import Counter
from statistics import mean as _statistics_mean
from typing import Any, Callable, Literal, get_args, get_origin

from mlflow.entities.assessment import Feedback
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.constants import _AFFIRMATIVE_VALUES, _NEGATIVE_VALUES

NUMERIC_ENSEMBLES: set[str] = {"mean", "minimum", "maximum"}
BOOL_ENSEMBLES: set[str] = {"agg_all", "agg_any"}


def _coerce_to_bool(value: Any) -> bool | None:
    """Map a sub-scorer value onto a bool, or ``None`` when it has no boolean reading.

    Built-in judges return ``Literal["yes", "no"]`` rather than a bool, so the boolean
    reducers accept the same affirmative/negative vocabulary the judges normalize to
    (``CategoricalRating`` is a ``StrEnum``, so it compares equal to its string value).
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _AFFIRMATIVE_VALUES:
            return True
        if normalized in _NEGATIVE_VALUES:
            return False
    return None


def _require_numeric(values: list[Any]) -> None:
    # Numeric built-ins (mean/minimum/maximum) reject categorical/string values with a
    # clear error rather than letting statistics/min/max raise an opaque TypeError.
    for v in values:
        # bool is an int subclass and is acceptable as a numeric 0/1.
        if not isinstance(v, (bool, int, float)):
            raise MlflowException.invalid_parameter_value(
                f"This ensemble function requires numeric sub-scorer values, but got a value of "
                f"type {type(v).__name__}. Use majority_vote for categorical values."
            )


def _as_bools(values: list[Any]) -> list[bool]:
    # agg_all/agg_any are boolean reducers, so never fall back to Python truthiness: a bare
    # `all(["no", "no"])` is True because non-empty strings are truthy. Coerce each value to
    # an explicit bool and reject anything with no boolean reading.
    bools = []
    for value in values:
        coerced = _coerce_to_bool(value)
        if coerced is None:
            raise MlflowException.invalid_parameter_value(
                f"This ensemble function requires boolean sub-scorer values, but got "
                f"{value!r} (type {type(value).__name__}), which has no yes/no reading. "
                f"Use majority_vote for other categorical values."
            )
        bools.append(coerced)
    return bools


def is_bool_feedback_type(feedback_value_type) -> bool:
    """Whether a declared ``feedback_value_type`` denotes a boolean value.

    Used to validate sub-scorers up front against boolean built-ins (agg_all/agg_any). A
    ``Literal[...]`` qualifies when every member has a boolean reading, which includes the
    ``Literal["yes", "no"]`` that built-in judges declare. Unknown/unannotated types return
    False (caller treats them as non-bool).
    """
    if feedback_value_type is bool:
        return True
    if get_origin(feedback_value_type) is Literal:
        return all(_coerce_to_bool(arg) is not None for arg in get_args(feedback_value_type))
    return False


def is_numeric_feedback_type(feedback_value_type) -> bool:
    """Whether a declared ``feedback_value_type`` denotes a numeric value.

    Used to validate sub-scorers up front against numeric built-ins. A ``Literal[...]`` is
    numeric only when every member is an int/float. ``bool`` counts as numeric (0/1).
    Unknown/unannotated types return False (caller treats them as categorical).
    """
    if feedback_value_type in (bool, int, float):
        return True
    if get_origin(feedback_value_type) is Literal:
        return all(isinstance(arg, (bool, int, float)) for arg in get_args(feedback_value_type))
    return False


def _require_complete(values: list[Any]) -> None:
    # Built-ins treat any failed/empty sub-scorer (surfaced as None) as fatal so a
    # partial ensemble never silently produces a misleading aggregate.
    if not values:
        raise MlflowException.invalid_parameter_value(
            "Aggregation failed: no sub-scorer values were provided."
        )
    if any(v is None for v in values):
        raise MlflowException.invalid_parameter_value(
            "Aggregation failed: at least one sub-scorer returned no value (it errored "
            "or produced an empty assessment). Built-in aggregation functions require "
            "every sub-scorer to return a value."
        )


def majority_vote(values: list[Any]) -> Feedback:
    _require_complete(values)
    counts = Counter(values)
    top = max(counts.values())
    # Deterministic tie-break: lexicographic order of the string form of each tied value.
    # This is intended for categorical/bool values. For numerics the string ordering is NOT
    # numeric ordering (e.g. str(10) < str(9), and False sorts before True), so majority_vote
    # is not appropriate for continuous numerics -- use mean/minimum/maximum for those.
    winner = min((v for v, c in counts.items() if c == top), key=lambda v: str(v))
    return Feedback(
        value=winner,
        rationale=f"Majority vote over {len(values)} scorers: {dict(counts)}",
    )


def mean(values: list[Any]) -> Feedback:
    _require_complete(values)
    _require_numeric(values)
    result = _statistics_mean(values)
    return Feedback(value=result, rationale=f"Mean over {len(values)} scorers = {result}")


def minimum(values: list[Any]) -> Feedback:
    _require_complete(values)
    _require_numeric(values)
    result = min(values)
    return Feedback(value=result, rationale=f"Minimum over {len(values)} scorers = {result}")


def maximum(values: list[Any]) -> Feedback:
    _require_complete(values)
    _require_numeric(values)
    result = max(values)
    return Feedback(value=result, rationale=f"Maximum over {len(values)} scorers = {result}")


def agg_all(values: list[Any]) -> Feedback:
    _require_complete(values)
    result = all(_as_bools(values))
    return Feedback(value=result, rationale=f"all() over {len(values)} scorers = {result}")


def agg_any(values: list[Any]) -> Feedback:
    _require_complete(values)
    result = any(_as_bools(values))
    return Feedback(value=result, rationale=f"any() over {len(values)} scorers = {result}")


BUILTIN_ENSEMBLES: dict[str, Callable[[list[Any]], Feedback]] = {
    "majority_vote": majority_vote,
    "mean": mean,
    "minimum": minimum,
    "maximum": maximum,
    "agg_all": agg_all,
    "agg_any": agg_any,
}
