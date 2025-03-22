from __future__ import annotations

import hashlib
from itertools import chain
from typing import TYPE_CHECKING, Any

import narwhals.stable.v1 as nw
from narwhals.typing import IntoDataFrameT

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    import pyarrow as pa

MAX_ROWS = 10000


def _trim_dataframe(df: IntoDataFrameT) -> IntoDataFrameT:
    return (
        nw.from_native(df)
        .head(MAX_ROWS)
        .select(*[nw.selectors.string(), nw.selectors.numeric()])
        .to_native()
    )


def compute_pandas_digest(df: pd.DataFrame) -> str:
    """Computes a digest for the given Pandas DataFrame.

    Args:
        df: A Pandas DataFrame.

    Returns:
        A string digest.
    """
    import numpy as np
    import pandas as pd

    num_rows = len(df)
    trimmed_df = _trim_dataframe(df)

    return get_normalized_md5_digest(
        [
            pd.util.hash_pandas_object(trimmed_df).values,
            np.int64(num_rows),
        ]
        + [str(x).encode() for x in df.columns]
    )


def compute_polars_digest(df: pl.DataFrame) -> str:
    """Computes a digest for the given polars DataFrame.

    Adjusted from the following stackoverflow answer:
    https://stackoverflow.com/a/79092287/12411536

    Args:
        df: A polars DataFrame.

    Returns:
        A string digest.
    """
    num_rows = len(df)
    trimmed_df = _trim_dataframe(df)

    elements = list(
        chain(
            [col_name.encode() for col_name in trimmed_df.columns],
            [str(col_type).encode() for col_type in trimmed_df.dtypes],
            [h.to_bytes(64, "big") for h in trimmed_df.hash_rows()],
            [str(num_rows).encode()],
        )
    )
    return get_normalized_md5_digest(elements)


def compute_pyarrow_digest(df: pa.Table) -> str:
    """Computes a digest for the pyarrow Table.

    Args:
        df: A pyarrow Table

    Returns:
        A string digest.
    """
    import pyarrow as pa

    num_rows = len(df)
    trimmed_df = _trim_dataframe(df)

    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, trimmed_df.schema) as writer:
        writer.write_table(trimmed_df)
    ipc_bytes = sink.getvalue().to_pybytes()

    elements = list(
        chain(
            [x.name.encode() for x in trimmed_df.schema],
            [str(x.type).encode() for x in trimmed_df.schema],
            [ipc_bytes],
            [str(num_rows).encode()],
        )
    )
    return get_normalized_md5_digest(elements)


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
