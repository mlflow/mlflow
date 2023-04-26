import sys

from . import sources as mlflow_data_sources
from mlflow.data import dataset_registry
from mlflow.data import dataset_source_registry


__all__ = []


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
    for source in dataset_source_registry.get_registered_sources():
        setattr(mlflow_data_sources, source.__name__, source)
        mlflow_data_sources.__all__.append(source.__name__)


_define_dataset_sources_in_sources_module()
