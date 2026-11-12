"""Private, in-process ListScorers authorization contract (never an HTTP selector)."""

from dataclasses import dataclass
from urllib.parse import quote, unquote

from mlflow.exceptions import MlflowNotImplementedException
from mlflow.utils.validation import _EXPERIMENT_ID_REGEX


def reject_unsupported_scorer_scope(scorer_scope):
    if scorer_scope is not None:
        raise MlflowNotImplementedException(
            "This tracking backend cannot enforce an internal scorer-list authorization scope. "
            "Use a SQL tracking backend."
        )


def is_canonical_scorer_experiment_id(value: str) -> bool:
    # Bound syntax before int(): persisted grants need not be SQL-representable.
    if not isinstance(value, str) or _EXPERIMENT_ID_REGEX.fullmatch(value) is None:
        return False
    try:
        return str(int(value)) == value
    except ValueError:
        return True


def scorer_row_key(experiment_id: str, scorer_name: str) -> str:
    return f"{experiment_id}/{quote(scorer_name, safe='')}"


def parse_scorer_row_key(pattern: str) -> tuple[str, str] | None:
    experiment_id, separator, encoded_name = pattern.partition("/")
    name = unquote(encoded_name)
    if (
        separator
        and is_canonical_scorer_experiment_id(experiment_id)
        and name.strip()
        and scorer_row_key(experiment_id, name) == pattern
    ):
        return experiment_id, name
    return None


@dataclass(frozen=True)
class ScorerListScope:
    """Positive union minus exclusions; outer None alone means unrestricted.

    Inner ``experiment_ids=None`` includes all experiments, subject to exclusions.
    The caller's experiment selection, workspace and active lifecycle still intersect
    this scope. Plugins may use opaque experiment IDs; SQL checks representability.
    """

    experiment_ids: frozenset[str] | None = frozenset()
    scorer_pairs: frozenset[tuple[str, str]] = frozenset()
    excluded_experiment_ids: frozenset[str] = frozenset()
    excluded_scorer_pairs: frozenset[tuple[str, str]] = frozenset()

    def __post_init__(self):
        for field in (
            "experiment_ids",
            "excluded_experiment_ids",
            "scorer_pairs",
            "excluded_scorer_pairs",
        ):
            value = getattr(self, field)
            if value is None and field == "experiment_ids":
                continue
            if not isinstance(value, (set, frozenset, list, tuple)):
                raise ValueError(f"Invalid {field} in internal scorer-list scope")
            pairs = field.endswith("pairs")
            for item in value:
                if pairs:
                    if (
                        not isinstance(item, tuple)
                        or len(item) != 2
                        or not is_canonical_scorer_experiment_id(item[0])
                        or not isinstance(item[1], str)
                        or not item[1].strip()
                    ):
                        raise ValueError(f"Invalid pair in {field}")
                elif not is_canonical_scorer_experiment_id(item):
                    raise ValueError(f"Invalid experiment ID in {field}")
            object.__setattr__(self, field, frozenset(value))

    def permits(self, experiment_id: str, scorer_name: str) -> bool:
        pair = (experiment_id, scorer_name)
        return (
            (
                self.experiment_ids is None
                or experiment_id in self.experiment_ids
                or pair in self.scorer_pairs
            )
            and experiment_id not in self.excluded_experiment_ids
            and pair not in self.excluded_scorer_pairs
        )
