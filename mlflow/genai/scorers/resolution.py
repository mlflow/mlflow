"""Resolution of scorer names to :py:class:`~mlflow.genai.scorers.base.Scorer` instances.

Per-record scorers are persisted as names, so the harness needs to turn a name back into a
runnable scorer. A name resolves against the built-in scorers first, then the scorer registry
for the active experiment.
"""

import logging

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.base import Scorer

_logger = logging.getLogger(__name__)


def _builtin_scorers_by_name() -> dict[str, Scorer]:
    from mlflow.genai.scorers.builtin_scorers import get_all_scorers

    return {scorer.name: scorer for scorer in get_all_scorers()}


def resolve_scorer_names(names: set[str], experiment_id: str | None = None) -> dict[str, Scorer]:
    """Resolve scorer names to scorer instances.

    Args:
        names: Scorer names to resolve.
        experiment_id: Experiment whose registered scorers to search. Defaults to the active
            experiment.

    Returns:
        Mapping of name to the resolved scorer.

    Raises:
        MlflowException: If any name resolves to neither a built-in nor a registered scorer.
    """
    if not names:
        return {}

    resolved: dict[str, Scorer] = {}
    builtins = _builtin_scorers_by_name()
    unresolved: list[str] = []

    for name in sorted(names):
        if builtin := builtins.get(name):
            resolved[name] = builtin
            continue

        try:
            resolved[name] = _get_registered_scorer(name, experiment_id)
        except Exception as e:
            _logger.debug(f"Failed to resolve scorer {name!r} from the registry: {e}")
            unresolved.append(name)

    if unresolved:
        raise MlflowException.invalid_parameter_value(
            f"The following per-record scorers could not be resolved: {unresolved}. "
            f"A per-record scorer must be either a built-in scorer or a scorer registered "
            f"to the experiment. Register a custom scorer with `my_scorer.register()` before "
            f"referencing it by name. Available built-in scorers: {sorted(builtins)}."
        )

    return resolved


def _get_registered_scorer(name: str, experiment_id: str | None) -> Scorer:
    from mlflow.genai.scorers.registry import get_scorer

    return get_scorer(name=name, experiment_id=experiment_id)
