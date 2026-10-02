from collections.abc import Iterable
from dataclasses import dataclass

from mlflow.exceptions import MlflowException
from mlflow.protos.service_pb2 import ScorerFilter as ProtoScorerFilter
from mlflow.utils.validation import _validate_experiment_id


@dataclass(frozen=True, init=False)
class ScorerFilter:
    """Select whole experiments and exact scorer pairs by union.

    An empty filter selects nothing. Pass None to listing APIs for unrestricted
    selection. Names are literal, including a name consisting of an asterisk.
    """

    experiment_ids: frozenset[str]
    scorers: frozenset[tuple[str, str]]

    def __init__(self, experiment_ids: Iterable[str] = (), scorers: Iterable[tuple[str, str]] = ()):
        experiment_ids = frozenset(experiment_ids)
        scorers = frozenset(scorers)
        for experiment_id in experiment_ids | {eid for eid, _ in scorers}:
            if not isinstance(experiment_id, str) or not experiment_id:
                raise MlflowException.invalid_parameter_value(
                    "Scorer filter experiment IDs must be nonempty strings."
                )
            _validate_experiment_id(experiment_id)
        for _, name in scorers:
            if not isinstance(name, str) or not name.strip():
                raise MlflowException.invalid_parameter_value(
                    "Scorer filter names must be nonempty strings."
                )
        object.__setattr__(self, "experiment_ids", experiment_ids)
        object.__setattr__(
            self, "scorers", frozenset(p for p in scorers if p[0] not in experiment_ids)
        )

    @property
    def candidate_experiment_ids(self) -> frozenset[str]:
        return self.experiment_ids | {eid for eid, _ in self.scorers}

    def intersect(self, other: "ScorerFilter") -> "ScorerFilter":
        return ScorerFilter(
            experiment_ids=self.experiment_ids & other.experiment_ids,
            scorers=(
                (self.scorers & other.scorers)
                | {p for p in self.scorers if p[0] in other.experiment_ids}
                | {p for p in other.scorers if p[0] in self.experiment_ids}
            ),
        )

    @classmethod
    def from_proto(cls, proto: ProtoScorerFilter) -> "ScorerFilter":
        return cls(
            experiment_ids=frozenset(proto.experiment_ids),
            scorers=frozenset((s.experiment_id, s.scorer_name) for s in proto.scorers),
        )

    def to_proto(self) -> ProtoScorerFilter:
        return ProtoScorerFilter(
            experiment_ids=sorted(self.experiment_ids),
            scorers=[
                ProtoScorerFilter.Scorer(experiment_id=eid, scorer_name=name)
                for eid, name in sorted(self.scorers)
            ],
        )
