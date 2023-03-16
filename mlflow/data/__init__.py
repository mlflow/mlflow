import sys

from mlflow.data import dataset_registry


__all__ = []


# TODO: Clean this up
data_module = sys.modules[__name__]
for (
    constructor_name,
    constructor_fn,
) in dataset_registry._dataset_registry._constructors.items():
    setattr(data_module, constructor_name, constructor_fn)
    __all__.append(constructor_name)
