import inspect
import entrypoints
import warnings
from typing import Any
from typing_extensions import Protocol

import mlflow.data
from mlflow.exceptions import MlflowException
from mlflow.data.dataset import Dataset
from mlflow.data.pandas_dataset import from_pandas 
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE


class LoaderFunction(Protocol):
    def __call__(self, *args: Any) -> Dataset:
        pass


class ConstructorFunction(Protocol):
    def __call__(self, *args: Any) -> Dataset:
        pass


class DatasetRegistry:

    def __init__(self):
        self._loaders = {}
        self._constructors = {}

    def register_loader(self, loader_name: str, loader_fn: LoaderFunction):
        """
        Registers a dataset loader.

        :param loader_name: The name of the loader, e.g. "load_delta". The name
                            must begin with the string "load_".
        :param loader_fn: A function that accepts at least the following
                          inputs and returns an instance of a subclass of
                          mlflow.data.Dataset:

                          - name: Optional. A string dataset name
                          - digest: Optional. A string dataset digest.
        """
        DatasetRegistry._validate_loader_or_constructor_fn(loader_fn)
        self._loaders[loader_name] = loader_fn

    def register_constructor(self, constructor_name: str, constructor_fn: ConstructorFunction):
        """
        Registers a dataset constructor.

        :param constructor_name: The name of the constructor, e.g.
                                 "from_spark". The name must begin with the
                                 string "from_".
        :param constructor_fn: A function that accepts at least the following
                               inputs and returns an instance of a subclass of
                               mlflow.data.Dataset:

                               - name: Optional. A string dataset name
                               - digest: Optional. A string dataset digest.
        """
        DatasetRegistry._validate_loader_or_constructor_fn(constructor_fn)
        self._constructors[constructor_name] = constructor_fn 

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
                    f'Failure attempting to register dataset constructor "{entrypoint.name}": {exc}',
                    stacklevel=2,
                )

    @staticmethod
    def _validate_loader_or_constructor_fn(fn):
        parameters = inspect.signature(fn).parameters
        for expected_kwarg in ["name", "digest"]:
            if expected_kwarg not in parameters or parameters[expected_kwarg].kind not in [
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ]:
                raise MlflowException(
                    "Invalid loader or constructor function: {fn.__name__}. Function must define"
                    "an optional parameter named '{expected_kwarg}'.",
                    INVALID_PARAMETER_VALUE,
                )


def register_loader(loader_name: str, loader_fn: LoaderFunction):
    """
    Registers a dataset loader.

    :param loader_name: The name of the loader, e.g. "load_delta". The name
                        must begin with the string "load_".
    :param loader_fn: A function that accepts at least the following
                      inputs and returns an instance of a subclass of
                      mlflow.data.Dataset:

                      - name: Optional. A string dataset name
                      - digest: Optional. A string dataset digest.
    """
    _dataset_registry.register_loader(loader_name, loader_fn)
    setattr(mlflow.data, loader_name, loader_fn)


def register_constructor(constructor_name: str, constructor_fn: ConstructorFunction):
    """
    Registers a dataset constructor.

    :param constructor_name: The name of the constructor, e.g.
                             "from_spark". The name must begin with the
                             string "from_".
    :param constructor_fn: A function that accepts at least the following
                           inputs and returns an instance of a subclass of
                           mlflow.data.Dataset:

                           - name: Optional. A string dataset name
                           - digest: Optional. A string dataset digest.
    """
    _dataset_registry.register_constructor(constructor_name, constructor_fn)
    setattr(mlflow.data, constructor_name, constructor_fn)


_dataset_registry = DatasetRegistry()
_dataset_registry.register_entrypoints()
_dataset_registry.register_constructor(constructor_name="from_pandas", constructor_fn=from_pandas)
