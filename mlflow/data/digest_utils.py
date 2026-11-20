import hashlib
from typing import Any

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
    trimmed_df = df.head(MAX_ROWS)

    # keep string, number, boolean, datetime, and categorical columns, drop other column types
    # (e.g. object columns of lists or dicts, which cannot be hashed)
    # `skipna=True` keeps string columns that contain missing values, as well as object columns
    # of booleans, dates, or datetimes that contain missing values (which `select_dtypes` below
    # doesn't match). Inferring on an object array keeps categorical columns of strings, for
    # which `infer_dtype` returns "categorical".
    inferred_columns = trimmed_df.columns[
        [
            pd.api.types.infer_dtype(column.to_numpy(dtype=object), skipna=True)
            in ("string", "boolean", "date", "datetime")
            for _, column in trimmed_df.items()
        ]
    ]
    typed_columns = trimmed_df.select_dtypes(
        include=[np.number, "bool", "boolean", "datetime", "datetimetz", "category"]
    ).columns

    desired_columns = inferred_columns.union(typed_columns)
    trimmed_df = _with_datetimes_in_nanoseconds(trimmed_df[desired_columns])

    return get_normalized_md5_digest(
        [
            pd.util.hash_pandas_object(trimmed_df).values,
            np.int64(len(df)),
        ]
        + [str(x).encode() for x in df.columns]
    )


def _with_datetimes_in_nanoseconds(df):
    """
    Convert the datetime columns of ``df`` to nanoseconds, so that the digest doesn't depend on
    the unit pandas stores them in. The hash covers the stored integers, and ``pd.to_datetime``
    on strings returns nanoseconds in pandas 2 but microseconds in pandas 3. A column outside
    the nanosecond range keeps its unit. PyArrow timestamps are left as they are, since their
    unit is part of the dtype.
    """
    import numpy as np
    import pandas as pd

    converted = None
    for position, (_, column) in enumerate(df.items()):
        dtype = column.dtype
        is_datetime = isinstance(dtype, pd.DatetimeTZDtype) or (
            isinstance(dtype, np.dtype) and dtype.kind == "M"
        )
        # `.dt.unit` is absent before pandas 2, which only stores nanoseconds.
        if not is_datetime or getattr(column.dt, "unit", "ns") == "ns":
            continue
        try:
            column = column.dt.as_unit("ns")
        except pd.errors.OutOfBoundsDatetime:
            continue
        if converted is None:
            converted = df.copy()
        # Set by position, which also handles duplicate column names.
        converted.isetitem(position, column)
    return df if converted is None else converted


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
        hashable_elements.extend(np.int64(x) for x in array.shape)

    def hash_dict_of_arrays(array_dict):
        for key in sorted(array_dict.keys()):
            hashable_elements.append(str(key).encode())
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
