import hashlib
from typing import Any

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

MAX_ROWS = 10000
_DIGEST_SIZE = 32


def compute_pandas_digest(df) -> str:
    """Computes a digest for the given Pandas DataFrame.

    Uses head+tail sampling to detect changes beyond the first MAX_ROWS rows,
    and includes all column types (not just string/numeric) to prevent
    collision attacks via excluded columns.

    Args:
        df: A Pandas DataFrame.

    Returns:
        A string digest.
    """
    import numpy as np
    import pandas as pd

    hashable_elements = []

    # For large DataFrames, sample both head and tail to prevent deterministic collision attacks
    if len(df) > MAX_ROWS:
        sample_size = MAX_ROWS // 2
        head_sample = df.head(sample_size)
        tail_sample = df.tail(sample_size)
        hashable_elements.append(pd.util.hash_pandas_object(head_sample).values)
        hashable_elements.append(pd.util.hash_pandas_object(tail_sample).values)
    else:
        # For small DataFrames, hash all rows
        hashable_elements.append(pd.util.hash_pandas_object(df).values)

    # Include total row count
    hashable_elements.append(np.int64(len(df)))

    # Include column names
    hashable_elements.extend(str(col).encode() for col in df.columns)

    # Include dtype information to prevent type-coercion collisions
    hashable_elements.extend(str(dtype).encode() for dtype in df.dtypes)

    return _compute_sha256_digest(hashable_elements)


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

        # For large arrays, sample both head and tail
        if flattened_array.size > MAX_ROWS:
            sample_size = MAX_ROWS // 2
            head_sample = flattened_array[:sample_size]
            tail_sample = flattened_array[-sample_size:]
            try:
                hashable_elements.append(pd.util.hash_array(head_sample))
                hashable_elements.append(pd.util.hash_array(tail_sample))
            except TypeError:
                hashable_elements.append(np.int64(head_sample.size))
                hashable_elements.append(np.int64(tail_sample.size))
        else:
            # For small arrays, hash all elements
            try:
                hashable_elements.append(pd.util.hash_array(flattened_array))
            except TypeError:
                hashable_elements.append(np.int64(flattened_array.size))

        # Hash full array dimensions
        hashable_elements.extend(np.int64(x) for x in array.shape)
        # Include dtype to prevent type-coercion collisions
        hashable_elements.append(str(array.dtype).encode())

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

    return _compute_sha256_digest(hashable_elements)


def _compute_sha256_digest(elements: list[Any]) -> str:
    """Computes a SHA-256 digest for a list of hashable elements.

    Args:
        elements: A list of hashable elements for inclusion in the digest.

    Returns:
        A hex digest string truncated to _DIGEST_SIZE characters.
    """
    if not elements:
        raise MlflowException(
            "No hashable elements were provided for digest creation",
            INVALID_PARAMETER_VALUE,
        )

    sha = hashlib.sha256()
    for element in elements:
        sha.update(element)

    return sha.hexdigest()[:_DIGEST_SIZE]


def get_normalized_md5_digest(elements: list[Any]) -> str:
    """Computes a normalized digest for a list of hashable elements.

    .. deprecated::
        This function now uses SHA-256 internally. The name is retained for
        backward compatibility. Use _compute_sha256_digest for new code.

    Args:
        elements: A list of hashable elements for inclusion in the digest.

    Returns:
        A hex digest string.
    """
    return _compute_sha256_digest(elements)
