from typing import Optional, Union

import pyarrow as pa

from mlflow.data.dataframe_dataset import DataFrameDataset, from_dataframe
from mlflow.data.dataset_source import DatasetSource


class ArrowDataset(DataFrameDataset[pa.Table]):
    """Represents a PyArrow Table for use with MLflow Tracking."""

    backend_name = "pyarrow"


def from_arrow(
    df: pa.Table,
    source: Union[str, DatasetSource] = None,
    targets: Optional[str] = None,
    name: Optional[str] = None,
    digest: Optional[str] = None,
    predictions: Optional[str] = None,
) -> ArrowDataset:
    """
    Constructs a :py:class:`ArrowDataset <mlflow.data.pyarrow_dataset.ArrowDataset>` instance from
    a PyArrow Table, optional targets, optional predictions, and source.

    Args:
        df: A PyArrow Table.
        source: The source from which the DataFrame was derived, e.g. a filesystem
            path, an S3 URI, an HTTPS URL, a delta table name with version, or
            spark table etc. ``source`` may be specified as a URI, a path-like string,
            or an instance of
            :py:class:`DatasetSource <mlflow.data.dataset_source.DatasetSource>`.
            If unspecified, the source is assumed to be the code location
            (e.g. notebook cell, script, etc.) where
            :py:func:`from_arrow <mlflow.data.from_arrow>` is being called.
        targets: An optional target column name for supervised training. This column
            must be present in the dataframe (``df``).
        name: The name of the dataset. If unspecified, a name is generated.
        digest: The dataset digest (hash). If unspecified, a digest is computed
            automatically.
        predictions: An optional predictions column name for model evaluation. This column
            must be present in the dataframe (``df``).

    .. code-block:: python
        :test:
        :caption: Example

        import mlflow
        import pyarrow as pa

        data = {
            "Name": ["tom", "nick", "july"],
            "Age": [10, 15, 14],
            "Label": [1, 0, 1],
            "ModelOutput": [1, 1, 1],
        }
        x = pa.table(data)
        dataset = mlflow.data.from_arrow(x, targets="Label", predictions="ModelOutput")
    """
    if not isinstance(df, pa.Table):
        raise TypeError(
            f"The specified dataframe must be an instance of pyarrow.Table. Got: {type(df)}."
        )
    return from_dataframe(
        df=df,
        source=source,
        targets=targets,
        name=name,
        digest=digest,
        predictions=predictions,
        dataset_cls=ArrowDataset,
    )
