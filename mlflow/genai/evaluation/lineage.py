"""Record what an evaluation run used: its dataset, scorers, and agent."""

import logging
from typing import Any

from mlflow.entities import Dataset as DatasetEntity
from mlflow.genai.datasets.evaluation_dataset import DATASET_IDENTITY_ATTR, compute_records_sha256

_logger = logging.getLogger(__name__)


def get_dataset_entity_from_attrs(data: Any) -> DatasetEntity | None:
    """
    Return the dataset that produced ``data`` via ``EvaluationDataset.to_df()``, if ``data``
    is a DataFrame whose records have not changed since.
    """
    import pandas as pd

    if not isinstance(data, pd.DataFrame):
        return None
    if not isinstance(identity := data.attrs.get(DATASET_IDENTITY_ATTR), dict):
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
