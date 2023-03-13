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


class ConstructorFunction(Protocol):
    def __call__(self, *args: Any) -> Dataset:
        pass


class DatasetRegistry:

    def __init__(self):
        self._constructors = {}

    def register_constructor(self, constructor_fn: ConstructorFunction, constructor_name: str = None):
        """
        Registers a dataset constructor.

        :param constructor_fn: A function that accepts at least the following
                               inputs and returns an instance of a subclass of
                               mlflow.data.Dataset:

                               - name: Optional. A string dataset name
                               - digest: Optional. A string dataset digest.

        :param constructor_name: The name of the constructor, e.g.
                                 "from_spark". The name must begin with the
                                 string "from_" or "load_". If unspecified, the `__name__`
                                 attribute of the `constructor_fn` is used instead and must
                                 begin with the string "from_" or "load_".
        """
        if constructor_name is None:
            constructor_name = constructor_fn.__name__
        DatasetRegistry._validate_constructor(constructor_fn, constructor_name)
        self._constructors[constructor_name] = constructor_fn 

    def register_entrypoints(self):
        """
        Registers dataset sources defined as Python entrypoints. For reference, see
        https://mlflow.org/docs/latest/plugins.html#defining-a-plugin.
        """
        for entrypoint in entrypoints.get_group_all("mlflow.dataset_constructor"):
            try:
                self.register_constructor(constructor_fn=entrypoint.load(), constructor_name=entrypoint.name)
            except (AttributeError, ImportError) as exc:
                warnings.warn(
                    f'Failure attempting to register dataset constructor "{entrypoint.name}": {exc}',
                    stacklevel=2,
                )

    @staticmethod
    def _validate_constructor(constructor_fn: ConstructorFunction, constructor_name: str):
        if not constructor_name.startswith("load_") and not constructor_name.startswith("from_"):
            raise MlflowException(
                f"Invalid dataset constructor name: {constructor_name}."
                f" Constructor name must start with 'load_' or 'from_'.",
                INVALID_PARAMETER_VALUE,
            )

        signature = inspect.signature(constructor_fn)
        parameters = signature.parameters
        for expected_kwarg in ["name", "digest"]:
            if expected_kwarg not in parameters or parameters[expected_kwarg].kind not in [
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ]:
                raise MlflowException(
                    f"Invalid dataset constructor function: {constructor_fn.__name__}. Function"
                    f" must define an optional parameter named '{expected_kwarg}'.",
                    INVALID_PARAMETER_VALUE,
                )

        if not issubclass(signature.return_annotation, Dataset):
            raise MlflowException(
                f"Invalid dataset constructor function: {constructor_fn.__name__}. Function must"
                f" have a return type annotation that is a subclass of `mlflow.data.Dataset`.",
                INVALID_PARAMETER_VALUE,
            )


def register_constructor(constructor_fn: ConstructorFunction, constructor_name: str):
    """
    Registers a dataset constructor.

    :param constructor_fn: A function that accepts at least the following
                           inputs and returns an instance of a subclass of
                           mlflow.data.Dataset:

                           - name: Optional. A string dataset name
                           - digest: Optional. A string dataset digest.

    :param constructor_name: The name of the constructor, e.g.
                             "from_spark". The name must begin with the
                             string "from_" or "load_". If unspecified, the `__name__`
                             attribute of the `constructor_fn` is used instead and must
                             begin with the string "from_" or "load_".
    """
    _dataset_registry.register_constructor(constructor_name, constructor_fn)
    setattr(mlflow.data, constructor_name, constructor_fn)


_dataset_registry = DatasetRegistry()
_dataset_registry.register_entrypoints()
_dataset_registry.register_constructor(constructor_name="from_pandas", constructor_fn=from_pandas)
