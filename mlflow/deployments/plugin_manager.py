import warnings
import sys
import abc

import entrypoints
from mlflow.deployments.base_plugin import BasePlugin


if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})

# TODO: refactor to have a common base class for all the plugin implementation


class PluginManager(ABC):
    """
    Abstract class defining a entrypoint based plugin registration.

    This class allows the registration of a function or class to provide an
    implementation for a given key/name. Implementations declared though the
    entrypoints can be automatically registered through the
    `register_entrypoints` method.
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
        return self._registry

    @property
    def has_plugins_loaded(self):
        return self._has_plugins_loaded

    def register(self, scheme, store_builder):
        self._registry[scheme] = store_builder

    def register_entrypoints(self):
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
    def __init__(self):
        super(DeploymentPlugins, self).__init__('mlflow.deployments')

    def __getitem__(self, item):
        try:
            return self.registry[item]
        except KeyError:
            raise RuntimeError('No registered plugin found in the name "%s"' % item)

    def register_entrypoints(self):
        super(DeploymentPlugins, self).register_entrypoints()
        for name, plugin_obj in self._registry.items():
            if not isinstance(plugin_obj, BasePlugin):
                raise RuntimeError("{} is not a subclass of ``BasePlugin`` from "
                                   "``mlflow.deployments``".format(name))
