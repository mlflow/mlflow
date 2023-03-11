import sys

from mlflow.data import dataset_registry


# TODO: Clean this up
data_module = sys.modules[__name__]
for loader_name, loader_fn in dataset_registry._dataset_registry._loaders.items():
    setattr(data_module, loader_name, loader_fn)

for constructor_name, constructor_fn in dataset_registry._dataset_registry._constructors.items():
    setattr(data_module, constructor_name, constructor_fn)

