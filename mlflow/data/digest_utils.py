import hashlib
from typing import Any

from packaging.version import Version
import narwhals.stable.v1 as nw

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

MAX_ROWS = 10000


def compute_pandas_digest(df) -> str:
    """Computes a digest for the given Pandas DataFrame.

    Args:
        df: A Pandas DataFrame.

    Returns:
        A string digest.
    """
    import numpy as np
    import pandas as pd

    # trim to max rows
    trimmed_df = (
        nw.from_native(df)
        .head(MAX_ROWS)
        .select(*[nw.selectors.string(), nw.selectors.numeric()])
        .to_native()
    )

    return get_normalized_md5_digest(
        [
            pd.util.hash_pandas_object(trimmed_df).values,
            np.int64(len(df)),
        ]
        + [str(x).encode() for x in df.columns]
    )


def compute_numpy_digest(features, targets=None) -> str:
    """Computes a digest for the given numpy array.

    Args:
        features: A numpy array containing dataset features.
        targets: A numpy array containing dataset targets. Optional.

    Returns:
        A string digest.
    """
    import numpy as np
    import pandas as pd

    hashable_elements = []

    def hash_array(array):
        flattened_array = array.flatten()
        trimmed_array = flattened_array[0:MAX_ROWS]
        try:
            hashable_elements.append(pd.util.hash_array(trimmed_array))
        except TypeError:
            hashable_elements.append(np.int64(trimmed_array.size))

        # hash full array dimensions
        for x in array.shape:
            hashable_elements.append(np.int64(x))

    def hash_dict_of_arrays(array_dict):
        for key in sorted(array_dict.keys()):
            hash_array(array_dict[key])

    for item in [features, targets]:
        if item is None:
            continue
        if isinstance(item, dict):
            hash_dict_of_arrays(item)
        else:
            hash_array(item)

    return get_normalized_md5_digest(hashable_elements)


def get_normalized_md5_digest(elements: list[Any]) -> str:
    """Computes a normalized digest for a list of hashable elements.

    Args:
        elements: A list of hashable elements for inclusion in the md5 digest.

    Returns:
        An 8-character, truncated md5 digest.
    """

    if not elements:
        raise MlflowException(
            "No hashable elements were provided for md5 digest creation",
            INVALID_PARAMETER_VALUE,
        )

    md5 = hashlib.md5(usedforsecurity=False)
    for element in elements:
        md5.update(element)

    return md5.hexdigest()[:8]
