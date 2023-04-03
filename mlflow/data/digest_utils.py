import hashlib
from typing import List, Any

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE


def compute_pandas_digest(df) -> str:
    """
    Computes a digest for the given Pandas DataFrame.

    :param df: A Pandas DataFrame.
    :return: A string digest.
    """
    import numpy as np
    import pandas as pd

    # drop object columns
    df = df.select_dtypes(exclude=["object"])
    return get_normalized_md5_digest(
        [
            pd.util.hash_pandas_object(df).values,
            np.int64(len(df)),
        ]
        + [x.encode() for x in df.columns]
    )


def get_normalized_md5_digest(elements: List[Any]) -> str:
    """
    Computes a normalized digest for a list of hashable elements.

    :param elements: A list of hashable elements for inclusion in the md5 digest.
    :return: An 8-character, truncated md5 digest.
    """

    if not elements:
        raise MlflowException(
            "No hashable elements were provided for md5 digest creation",
            INVALID_PARAMETER_VALUE,
        )

    md5 = hashlib.md5()
    for element in elements:
        md5.update(element)

    return md5.hexdigest()[:8]
