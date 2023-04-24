import sys
from typing import Union, Any

from mlflow.data import dataset_registry
from mlflow.data.dataset import Dataset
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.dataset_source_registry import get_dataset_source_from_json
from mlflow.entities import DatasetInput
from mlflow.entities import Dataset as DatasetEntity
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE


def load_source(dataset: Union[DatasetEntity, DatasetInput, Dataset]) -> Any:
    """
    Loads source files / objects for the specified dataset. The exact loading behavior and return
    type is determined by the type of dataset source. For example, this may download source CSV
    files for a :py:class:``mlflow.data.pandas_dataset.PandasDataset`` from S3 to the local
    filesystem, or it might load a Delta Table source for a
    :py:class:``mlflow.data.spark_dataset.SparkDataset`` as a Spark DataFrame.

    :param dataset: An instance of :py:class:`mlflow.data.Dataset`,
                    :py:class:`mlflow.entities.Dataset`, or
                    :py:class:`mlflow.entities.DatasetInput`.
    :return: The downloaded source, e.g. a local filesystem path, a Spark DataFrame, etc.
    """
    if isinstance(dataset, DatasetInput):
        dataset: DatasetEntity = dataset.dataset

    if isinstance(dataset, DatasetEntity):
        dataset_source: DatasetSource = get_dataset_source_from_json(
            source_json=dataset.source,
            source_type=dataset.source_type,
        )
    elif isinstance(dataset, Dataset):
        dataset_source: DatasetSource = dataset.source
    else:
        raise MlflowException(
            f"Unrecognized dataset type {type(dataset)}. Expected one of: `mlflow.data.Dataset`,"
            f" `mlflow.entities.Dataset`, `mlflow.entities.DatasetInput`.",
            INVALID_PARAMETER_VALUE,
        )

    return dataset_source.load()


__all__ = ["load_source"]


def _define_dataset_constructors_in_current_module():
    data_module = sys.modules[__name__]
    for (
        constructor_name,
        constructor_fn,
    ) in dataset_registry.get_registered_constructors().items():
        setattr(data_module, constructor_name, constructor_fn)
        __all__.append(constructor_name)


_define_dataset_constructors_in_current_module()
