import warnings
from abc import ABCMeta, abstractmethod

import entrypoints

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.uri import get_uri_scheme


class UnsupportedModelRegistryStoreURIException(MlflowException):
    """Exception thrown when building a model registry store with an unsupported URI"""

    def __init__(self, unsupported_uri, supported_uri_schemes):
        message = (
            " Model registry functionality is unavailable; got unsupported URI"
            f" '{unsupported_uri}' for model registry data storage. Supported URI schemes are:"
            f" {supported_uri_schemes}."
            " See https://www.mlflow.org/docs/latest/tracking.html#storage for how to run"
            " an MLflow server against one of the supported backend storage locations."
        )
        super().__init__(message, error_code=INVALID_PARAMETER_VALUE)
        self.supported_uri_schemes = supported_uri_schemes


class StoreRegistry(metaclass=ABCMeta):
    """
    Abstract class defining a scheme-based registry for store implementations.

    This class allows the registration of a function or class to provide an
    implementation for a given scheme of `store_uri` through the `register`
    methods. Implementations declared though the entrypoints can be automatically
    registered through the `register_entrypoints` method.

    When instantiating a store through the `get_store` method, the scheme of
    the store URI provided (or inferred from environment) will be used to
    select which implementation to instantiate, which will be called with same
    arguments passed to the `get_store` method.
    """

    @abstractmethod
    def __init__(self, group_name):
        self._registry = {}
        self.group_name = group_name

    def register(self, scheme, store_builder):
        self._registry[scheme] = store_builder

    def register_entrypoints(self):
        """Register tracking stores provided by other packages"""
        for entrypoint in entrypoints.get_group_all(self.group_name):
            try:
                self.register(entrypoint.name, entrypoint.load())
            except (AttributeError, ImportError) as exc:
                warnings.warn(
                    'Failure attempting to register store for scheme "{}": {}'.format(
                        entrypoint.name, str(exc)
                    ),
                    stacklevel=2,
                )

    def get_store_builder(self, store_uri):
        """Get a store from the registry based on the scheme of store_uri

        :param store_uri: The store URI. If None, it will be inferred from the environment. This
                          URI is used to select which tracking store implementation to instantiate
                          and is passed to the constructor of the implementation.
        :return: A function that returns an instance of
                 ``mlflow.store.{tracking|model_registry}.AbstractStore`` that fulfills the store
                  URI requirements.
        """
        scheme = (
            store_uri if store_uri in {"databricks", "databricks-uc"} else get_uri_scheme(store_uri)
        )
        try:
            store_builder = self._registry[scheme]
        except KeyError:
            raise UnsupportedModelRegistryStoreURIException(
                unsupported_uri=store_uri, supported_uri_schemes=list(self._registry.keys())
            )
        return store_builder
