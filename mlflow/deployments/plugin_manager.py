import warnings
import abc

import entrypoints
from mlflow.deployments.base_plugin import BasePlugin
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, INTERNAL_ERROR


# TODO: refactor to have a common base class for all the plugin implementation in MLFlow
#   mlflow/tracking/context/registry.py
#   mlflow/tracking/registry
#   mlflow/store/artifact/artifact_repository_registry.py


class PluginManager(abc.ABC):
    """
    Abstract class defining a entrypoint based plugin registration.

    This class allows the registration of a function or class to provide an implementation
    for a given key/name. Implementations declared though the entrypoints can be automatically
    registered through the `register_entrypoints` method.
    """

    @abc.abstractmethod
    def __init__(self, group_name):
        self._registry = {}
        self.group_name = group_name
        self._has_plugins_loaded = None

    @abc.abstractmethod
    def __getitem__(self, item):
        # Letting the child class create this function so that the child
        # can raise custom exceptions if it needs to
        pass

    @property
    def registry(self):
        """
        Registry stores the registered plugin as a key value pair where key is the
        name of the plugin and value is the plugin object
        """
        return self._registry

    @property
    def has_plugins_loaded(self):
        """
        Returns bool representing whether the "register_entrypoints" has run or not. This
        doesn't return True if `register` method is called outside of `register_entrypoints`
        to register plugins
        """
        return self._has_plugins_loaded

    def register(self, scheme, plugin_object):
        self._registry[scheme] = plugin_object

    def register_entrypoints(self):
        """
        Runs through all the packages that has the `group_name` defined as the entrypoint
        and register that into the registry
        """
        for entrypoint in entrypoints.get_group_all(self.group_name):
            try:
                PluginClass = entrypoint.load()
                self.register(entrypoint.name, PluginClass())
            except (AttributeError, ImportError) as exc:
                warnings.warn(
                    'Failure attempting to register store for scheme "{}": {}'.format(
                        entrypoint.name, str(exc)
                    ),
                    stacklevel=2
                )
        self._has_plugins_loaded = True


class DeploymentPlugins(PluginManager):
    def __init__(self, auto_register=False):
        super(DeploymentPlugins, self).__init__('mlflow.deployments')
        self.auto_register = auto_register

    def __getitem__(self, item):
        if not self.has_plugins_loaded and self.auto_register:
            self.register_entrypoints()
        try:
            return self.registry[item]
        except KeyError:
            msg = 'No registered plugin found for the target "{}". For activating/registering ' \
                  'a new plugin, find the required plugin from `community-plugins ' \
                  '<https://mlflow.org/docs/latest/plugins.html#community-plugins>`_ and ' \
                  'install it using your package manager (pip, conda etc). MlFlow will ' \
                  'autoregister the plugin on the launch'.format(item)
            raise MlflowException(msg, error_code=RESOURCE_DOES_NOT_EXIST)

    def register_entrypoints(self):
        super(DeploymentPlugins, self).register_entrypoints()
        for name, plugin_obj in self._registry.items():
            if not isinstance(plugin_obj, BasePlugin):
                raise MlflowException("Plugin registered for the target {} is not a child "
                                      "class of ``BasePlugin`` from ``mlflow.deployments`` and "
                                      "hence considered invalid. Consider raising it to the "
                                      "plugin developers".format(name),
                                      error_code=INTERNAL_ERROR)
