"""Record what an evaluation run used: its dataset, scorers, and agent."""

import functools
import hashlib
import inspect
import json
import logging
from typing import Any, Callable

from mlflow.entities import Dataset as DatasetEntity
from mlflow.entities import RunTag
from mlflow.genai.datasets.evaluation_dataset import compute_records_sha256
from mlflow.genai.scorers.scorer_utils import get_scorer_definition_digest
from mlflow.tracking.client import MlflowClient
from mlflow.utils.mlflow_tags import (
    MLFLOW_GENAI_EVALUATE_AGENT_DIGEST,
    MLFLOW_GENAI_EVALUATE_AGENT_FUNCTION,
    MLFLOW_GENAI_EVALUATE_AGENT_URI,
    MLFLOW_GENAI_EVALUATE_DATASET_IDENTITY,
    MLFLOW_GENAI_EVALUATE_SCORERS_DIGEST,
)
from mlflow.utils.validation import MAX_TAG_VAL_LENGTH

_logger = logging.getLogger(__name__)

# Attribute that `to_predict_fn` sets on the function it returns.
AGENT_URI_ATTR = "_mlflow_agent_uri"


def _sha256(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_dataset_entity_from_attrs(data: Any) -> DatasetEntity | None:
    """
    Return the dataset that produced ``data`` via ``EvaluationDataset.to_df()``, if ``data``
    is a DataFrame whose records have not changed since.
    """
    import pandas as pd

    if not isinstance(data, pd.DataFrame):
        return None
    if not isinstance(identity := data.attrs.get(MLFLOW_GENAI_EVALUATE_DATASET_IDENTITY), dict):
        return None
    try:
        if compute_records_sha256(data) != identity.get("records_sha256"):
            return None
        return DatasetEntity(
            name=identity["name"],
            digest=identity["digest"],
            source_type=identity["source_type"],
            source=identity["source"],
            schema=identity.get("schema"),
            profile=identity.get("profile"),
        )
    except Exception:
        _logger.debug("Failed to read the dataset identity from the DataFrame", exc_info=True)
        return None


def get_scorers_digest(scorers: list[Any]) -> dict[str, Any] | None:
    """
    Digest the definitions of the scorers of an evaluation, registered or not. A scorer whose
    definition can't be hashed is only counted, so the digest is partial when ``hashed < total``.
    """
    if not scorers:
        return None
    digests = [digest for scorer in scorers if (digest := get_scorer_definition_digest(scorer))]
    value: dict[str, Any] = {}
    if digests:
        value["digest"] = _sha256("\n".join(sorted(digests)))
    value["hashed"] = len(digests)
    value["total"] = len(scorers)
    return value


def _unwrap_partial(predict_fn: Callable[..., Any]) -> Callable[..., Any]:
    while isinstance(predict_fn, functools.partial):
        predict_fn = predict_fn.func
    return predict_fn


def get_predict_fn_metadata(predict_fn: Callable[..., Any] | None) -> dict[str, str]:
    """Identify the agent under evaluation. Pass the user's ``predict_fn``, not a wrapper."""
    if predict_fn is None:
        return {}
    try:
        target = _unwrap_partial(predict_fn)
        if (uri := getattr(target, AGENT_URI_ATTR, None)) is not None:
            # The `to_predict_fn` wrapper is not the agent, so record only the remote target.
            return {MLFLOW_GENAI_EVALUATE_AGENT_URI: uri}

        if not (inspect.isroutine(target) or inspect.isclass(target)):
            # A callable instance is identified by its class.
            target = type(target)
        tags = {}
        module = getattr(target, "__module__", None)
        qualname = getattr(target, "__qualname__", None)
        if module and qualname:
            tags[MLFLOW_GENAI_EVALUATE_AGENT_FUNCTION] = f"{module}.{qualname}"
        try:
            tags[MLFLOW_GENAI_EVALUATE_AGENT_DIGEST] = _sha256(inspect.getsource(target))
        except (OSError, TypeError):
            pass
        return tags
    except Exception:
        _logger.debug("Failed to identify the evaluated agent", exc_info=True)
        return {}


def log_lineage_tags(
    run_id: str, scorers: list[Any], predict_fn: Callable[..., Any] | None
) -> None:
    """Write the lineage tags of an evaluation run. Never raises."""
    try:
        tags = get_predict_fn_metadata(predict_fn)
        if (scorers_digest := get_scorers_digest(scorers)) is not None:
            tags[MLFLOW_GENAI_EVALUATE_SCORERS_DIGEST] = json.dumps(scorers_digest)

        run_tags = []
        for key, value in tags.items():
            # A truncated value would be invalid JSON or a wrong identity, so skip it instead.
            if len(value) > MAX_TAG_VAL_LENGTH:
                _logger.warning(
                    f"Skipping run tag '{key}' because its value has {len(value)} characters, "
                    f"which exceeds the limit of {MAX_TAG_VAL_LENGTH}."
                )
                continue
            run_tags.append(RunTag(key, value))
        if run_tags:
            MlflowClient().log_batch(run_id, tags=run_tags)
    except Exception:
        _logger.debug("Failed to log evaluation lineage tags", exc_info=True)
