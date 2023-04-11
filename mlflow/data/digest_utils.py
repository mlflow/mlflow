import hashlib
from typing import List, Any

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

MAX_ROWS = 10000


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

    # trim to max rows
    trimmed_df = df.head(MAX_ROWS)

    return get_normalized_md5_digest(
        [
            pd.util.hash_pandas_object(trimmed_df).values,
            np.int64(len(df)),
        ]
        + [str(x).encode() for x in df.columns]
    )


def compute_numpy_digest(features, targets=None) -> str:
    """
    Computes a digest for the given numpy array.

    :param features: A numpy array containing dataset features.
    :param targets: A numpy array containing dataset targets. Optional.
    :return: A string digest.
    """
    import numpy as np
    import pandas as pd

    hashable_elements = []
    for array in [features, targets]:
        if array is None:
            continue
        flattened_array = array.flatten()
        trimmed_array = flattened_array[0:MAX_ROWS]
        try:
            hashable_elements.append(pd.util.hash_array(trimmed_array))
        except TypeError:
            hashable_elements.append(np.int64(trimmed_array.size))

        # hash full array dimensions
        for x in array.shape:
            hashable_elements.append(np.int64(x))

    return get_normalized_md5_digest(hashable_elements)


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
