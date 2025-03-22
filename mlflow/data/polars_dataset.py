from typing import Optional, Union

import polars as pl

from mlflow.data.dataframe_dataset import DataFrameDataset, from_dataframe
from mlflow.data.dataset_source import DatasetSource


class PolarsDataset(DataFrameDataset[pl.DataFrame]):
    """Represents a Polars DataFrame for use with MLflow Tracking."""

    backend_name = "polars"


def from_polars(
    df: pl.DataFrame,
    source: Union[str, DatasetSource] = None,
    targets: Optional[str] = None,
    name: Optional[str] = None,
    digest: Optional[str] = None,
    predictions: Optional[str] = None,
) -> PolarsDataset:
    """
    Constructs a :py:class:`PolarsDataset <mlflow.data.polars_dataset.PolarsDataset>` instance from
    a Polars DataFrame, optional targets, optional predictions, and source.

    Args:
        df: A Polars DataFrame.
        source: The source from which the DataFrame was derived, e.g. a filesystem
            path, an S3 URI, an HTTPS URL, a delta table name with version, or
            spark table etc. ``source`` may be specified as a URI, a path-like string,
            or an instance of
            :py:class:`DatasetSource <mlflow.data.dataset_source.DatasetSource>`.
            If unspecified, the source is assumed to be the code location
            (e.g. notebook cell, script, etc.) where
            :py:func:`from_polars <mlflow.data.from_polars>` is being called.
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
        import polars as pl

        data = {
            "Name": ["tom", "nick", "july"],
            "Age": [10, 15, 14],
            "Label": [1, 0, 1],
            "ModelOutput": [1, 1, 1],
        }
        x = pl.DataFrame(data)
        dataset = mlflow.data.from_polars(x, targets="Label", predictions="ModelOutput")
    """
    if not isinstance(df, pl.DataFrame):
        raise TypeError(
            f"The specified dataframe must be an instance of polars.DataFrame. Got: {type(df)}."
        )
    dataframe_dataset = from_dataframe(
        df=df,
        source=source,
        targets=targets,
        name=name,
        digest=digest,
        predictions=predictions,
    )
    return PolarsDataset(
        df=df,
        source=dataframe_dataset.source,
        targets=dataframe_dataset.targets,
        name=dataframe_dataset.name,
        digest=dataframe_dataset.digest,
        predictions=dataframe_dataset.predictions,
    )
