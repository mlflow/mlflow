import sys

from mlflow.data import dataset_registry


__all__ = []


def define_dataset_constructors_in_current_module():
    data_module = sys.modules[__name__]
    for (
        constructor_name,
        constructor_fn,
    ) in dataset_registry.get_registered_constructors().items():
        setattr(data_module, constructor_name, constructor_fn)
        __all__.append(constructor_name)


define_dataset_constructors_in_current_module()
