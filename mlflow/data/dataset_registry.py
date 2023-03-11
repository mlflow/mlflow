import warnings
import entrypoints
from typing import Any, Callable
from typing_extensions import Protocol

import mlflow.data
from mlflow.exceptions import MlflowException
from mlflow.data.dataset import Dataset
from mlflow.data.dataset_source import DatasetSource
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE


class LoaderFunction(Protocol):

    def __call__(self, *args: str) -> Dataset:
        pass


class ConstructorFunction(Protocol):

    def __call__(self, *args: str) -> Dataset:
        pass


class DatasetRegistry:

    def register_loader(self, loader_name: str, loader_fn: LoaderFunction):
        """
        Registers a dataset loader.

        :param loader_name: The name of the loader, e.g. "load_delta". The name
                            must begin with the string "load_".
        :param loader_fn: A function that accepts at least the following
                          inputs and returns an instance of a subclass of
                          mlflow.data.Dataset:

                          - source: Required. A DatasetSource object.
                          - name: Optional. A string dataset name
                          - digest: Optional. A string dataset digest.
        """
        setattr(mlflow.data, loader_name, loader_fn)

    def register_constructor(self, constructor_name: str, constructor_fn: ConstructorFunction):
        """
        Registers a dataset constructor.

        :param constructor_name: The name of the constructor, e.g.
                                 "from_spark". The name must begin with the
                                 string "from_".
        :param constructor_fn: A function that accepts at least the following
                               inputs and returns an instance of a subclass of
                               mlflow.data.Dataset:

                               - source: Required. A DatasetSource object.
                               - name: Optional. A string dataset name
                               - digest: Optional. A string dataset digest.
        """
        # Verify that constructor_name is prefixed with "from_"
        # Validate constructor_fn; verify that it accepts name, digest, source

        # When a user calls a constructor like mlflow.data.from_spark,
        # they don't pass in a DatasetSource. They pass in a raw source object,
        # e.g. a string like "mycatalog.myschema.mytable@2". Accordingly,
        # we create a wrapped function that resolves the raw source to a
        # DatasetSource so that the developer doesn't have to do it.
        # wrapper_fn = create_wrapper_that_resolves_source(constructor_fn)
        # setattr(mlflow.data, constructor_name, wrapper_fn)
        setattr(mlflow.data, constructor_name, constructor_fn)

    def register_entrypoints(self):
        """
        Registers dataset sources defined as Python entrypoints. For reference, see
        https://mlflow.org/docs/latest/plugins.html#defining-a-plugin.
        """
        for entrypoint in entrypoints.get_group_all("mlflow.dataset_loader"):
            try:
                self.register_loader(entrypoint.name, entrypoint.load())
            except (AttributeError, ImportError) as exc:
                warnings.warn(
                    f'Failure attempting to register dataset loader "{entrypoint.name}": {exc}',
                    stacklevel=2,
                )

        for entrypoint in entrypoints.get_group_all("mlflow.dataset_constructor"):
            try:
                self.register_constructor(entrypoint.name, entrypoint.load())
            except (AttributeError, ImportError) as exc:
                warnings.warn(
                    f'Failure attempting to register dataset loader "{entrypoint.name}": {exc}',
                    stacklevel=2,
                )


dataset_registry = DatasetRegistry()
# dataset_registry.register(DeltaTableDatasetSource)
# dataset_registry.register(HuggingFaceHubDatasetSource)
# dataset_registry.register(S3DatasetSource)
# dataset_registry.register_entrypoints()
