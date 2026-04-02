import hashlib
from typing import Any

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

MAX_ROWS = 10000
# 32 hex characters = 128 bits of collision resistance (birthday bound ~2^64)
_DIGEST_SIZE = 32


def compute_pandas_digest(df) -> str:
    """Computes a digest for the given Pandas DataFrame.

    All column types are included in the digest (not just string and numeric),
    and for large DataFrames both head and tail rows are hashed to detect
    changes beyond the first MAX_ROWS rows.

    Args:
        df: A Pandas DataFrame.

    Returns:
        A string digest.
    """
    import numpy as np
    import pandas as pd

    hashable_elements = []

    # Hash the first MAX_ROWS rows (all columns, all dtypes)
    trimmed_head = df.head(MAX_ROWS)
    hashable_elements.append(pd.util.hash_pandas_object(trimmed_head).values)

    # For large DataFrames, also hash the tail to cover content beyond the head
    if len(df) > MAX_ROWS:
        trimmed_tail = df.tail(MAX_ROWS)
        hashable_elements.append(pd.util.hash_pandas_object(trimmed_tail).values)

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
        trimmed_head = flattened_array[0:MAX_ROWS]
        try:
            hashable_elements.append(pd.util.hash_array(trimmed_head))
        except TypeError:
            hashable_elements.append(np.int64(trimmed_head.size))

        # For large arrays, also hash the tail
        if flattened_array.size > MAX_ROWS:
            trimmed_tail = flattened_array[-MAX_ROWS:]
            try:
                hashable_elements.append(pd.util.hash_array(trimmed_tail))
            except TypeError:
                hashable_elements.append(np.int64(trimmed_tail.size))

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
        backward compatibility. Prefer ``_compute_sha256_digest`` for new code.

    Args:
        elements: A list of hashable elements for inclusion in the digest.

    Returns:
        A hex digest string.
    """
    return _compute_sha256_digest(elements)
