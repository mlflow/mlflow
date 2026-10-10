from urllib.parse import quote, unquote

from mlflow.exceptions import MlflowException
from mlflow.utils.validation import _EXPERIMENT_ID_REGEX


def format_scorer_key(experiment_id: str, scorer_name: str) -> str:
    """Build the canonical ``<experiment_id>/<URL-encoded-scorer-name>`` key.

    Scorer names may contain arbitrary characters including ``/`` (see
    ``validate_scorer_name``, which only forbids empty/whitespace), so the name
    component is URL-encoded to keep the key unambiguous. The auth store's
    scorer grant patterns and the ``ListScorers.scorer_keys`` selector share
    this format.
    """
    return f"{experiment_id}/{quote(scorer_name, safe='')}"


def parse_scorer_key(scorer_key: str) -> tuple[str, str]:
    """Parse a canonical ``<experiment_id>/<URL-encoded-scorer-name>`` key."""
    experiment_id, separator, encoded_name = scorer_key.partition("/")
    scorer_name = unquote(encoded_name)
    if (
        not separator
        or _EXPERIMENT_ID_REGEX.fullmatch(experiment_id) is None
        or (
            experiment_id.isascii()
            and experiment_id.isdecimal()
            and str(int(experiment_id)) != experiment_id
        )
        or not encoded_name
        or format_scorer_key(experiment_id, scorer_name) != scorer_key
    ):
        raise MlflowException.invalid_parameter_value(
            f"Invalid scorer key {scorer_key!r}. Expected "
            "'<experiment_id>/<URL-encoded-scorer-name>'."
        )
    return experiment_id, scorer_name


def experiment_id_sort_key(experiment_id: str) -> tuple[int, int, str]:
    """Sort numeric experiment IDs numerically and non-numeric IDs after them."""
    if experiment_id.isascii() and experiment_id.isdecimal():
        return 0, int(experiment_id), experiment_id
    return 1, 0, experiment_id
