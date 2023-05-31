import sys
from typing import Union

from . import sources as mlflow_data_sources
from mlflow.data import dataset_registry
from mlflow.data.dataset import Dataset
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.dataset_source_registry import get_dataset_source_from_json, get_registered_sources
from mlflow.entities import DatasetInput
from mlflow.entities import Dataset as DatasetEntity
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.annotations import experimental


@experimental
def get_source(dataset: Union[DatasetEntity, DatasetInput, Dataset]) -> DatasetSource:
    """
    Obtains the source of the specified dataset or dataset input.

    :param dataset: An instance of :py:class:`mlflow.data.dataset.Dataset
                    <mlflow.data.dataset.Dataset>`,
                    :py:class:`mlflow.entities.Dataset`, or
                    :py:class:`mlflow.entities.DatasetInput`.
    :return: An instance of :py:class:`DatasetSource <mlflow.data.dataset_source.DatasetSource>`.
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
            f"Unrecognized dataset type {type(dataset)}. Expected one of: "
            f"`mlflow.data.dataset.Dataset`,"
            f" `mlflow.entities.Dataset`, `mlflow.entities.DatasetInput`.",
            INVALID_PARAMETER_VALUE,
        )

    return dataset_source


__all__ = ["get_source"]


def _define_dataset_constructors_in_current_module():
    data_module = sys.modules[__name__]
    for (
        constructor_name,
        constructor_fn,
    ) in dataset_registry.get_registered_constructors().items():
        setattr(data_module, constructor_name, constructor_fn)
        __all__.append(constructor_name)


_define_dataset_constructors_in_current_module()


def _define_dataset_sources_in_sources_module():
    for source in get_registered_sources():
        setattr(mlflow_data_sources, source.__name__, source)
        mlflow_data_sources.__all__.append(source.__name__)


_define_dataset_sources_in_sources_module()
