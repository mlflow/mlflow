from typing import Optional, Union

import pandas as pd

from mlflow.data.dataframe_dataset import DataFrameDataset, from_dataframe
from mlflow.data.dataset_source import DatasetSource


class PandasDataset(DataFrameDataset[pd.DataFrame]):
    """Represents a Pandas DataFrame for use with MLflow Tracking."""

    backend_name = "pandas"


def from_pandas(
    df: pd.DataFrame,
    source: Union[str, DatasetSource] = None,
    targets: Optional[str] = None,
    name: Optional[str] = None,
    digest: Optional[str] = None,
    predictions: Optional[str] = None,
) -> PandasDataset:
    """
    Constructs a :py:class:`PandasDataset <mlflow.data.pandas_dataset.PandasDataset>` instance from
    a Pandas DataFrame, optional targets, optional predictions, and source.

    Args:
        df: A Pandas DataFrame.
        source: The source from which the DataFrame was derived, e.g. a filesystem
            path, an S3 URI, an HTTPS URL, a delta table name with version, or
            spark table etc. ``source`` may be specified as a URI, a path-like string,
            or an instance of
            :py:class:`DatasetSource <mlflow.data.dataset_source.DatasetSource>`.
            If unspecified, the source is assumed to be the code location
            (e.g. notebook cell, script, etc.) where
            :py:func:`from_pandas <mlflow.data.from_pandas>` is being called.
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
        import pandas as pd

        x = pd.DataFrame(
            [["tom", 10, 1, 1], ["nick", 15, 0, 1], ["july", 14, 1, 1]],
            columns=["Name", "Age", "Label", "ModelOutput"],
        )
        dataset = mlflow.data.from_pandas(x, targets="Label", predictions="ModelOutput")
    """
    return from_dataframe(
        df=df,
        source=source,
        targets=targets,
        name=name,
        digest=digest,
        predictions=predictions,
    )
