import abc
import inspect

import entrypoints
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, INTERNAL_ERROR
from mlflow.deployments.base import BaseDeploymentClient
from mlflow.deployments.utils import parse_target_uri

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
        self._has_registered = None

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
    def has_registered(self):
        """
        Returns bool representing whether the "register_entrypoints" has run or not. This
        doesn't return True if `register` method is called outside of `register_entrypoints`
        to register plugins
        """
        return self._has_registered

    def register_entrypoints(self):
        """
        Runs through all the packages that has the `group_name` defined as the entrypoint
        and register that into the registry
        """
        for entrypoint in entrypoints.get_group_all(self.group_name):
            self.registry[entrypoint.name] = entrypoint
        self._has_registered = True


class DeploymentPlugins(PluginManager):
    def __init__(self):
        super().__init__("mlflow.deployments")
        self.register_entrypoints()

    def __getitem__(self, item):
        """Override __getitem__ so that we can directly look up plugins via dict-like syntax"""
        try:
            target_name = parse_target_uri(item)
            plugin_like = self.registry[target_name]
        except KeyError:
            msg = (
                'No plugin found for managing model deployments to "{target}". '
                'In order to deploy models to "{target}", find and install an appropriate '
                "plugin from "
                "https://mlflow.org/docs/latest/plugins.html#community-plugins using "
                "your package manager (pip, conda etc).".format(target=item)
            )
            raise MlflowException(msg, error_code=RESOURCE_DOES_NOT_EXIST)

        if isinstance(plugin_like, entrypoints.EntryPoint):
            try:
                plugin_obj = plugin_like.load()
            except (AttributeError, ImportError) as exc:
                raise RuntimeError('Failed to load the plugin "{}": {}'.format(item, str(exc)))
            self.registry[item] = plugin_obj
        else:
            plugin_obj = plugin_like

        # Testing whether the plugin is valid or not
        expected = {"target_help", "run_local"}
        deployment_classes = []
        for name, obj in inspect.getmembers(plugin_obj):
            if name in expected:
                expected.remove(name)
            elif (
                inspect.isclass(obj)
                and issubclass(obj, BaseDeploymentClient)
                and not obj == BaseDeploymentClient
            ):
                deployment_classes.append(name)
        if len(expected) > 0:
            raise MlflowException(
                "Plugin registered for the target {} does not has all "
                "the required interfaces. Raise an issue with the "
                "plugin developers.\n"
                "Missing interfaces: {}".format(item, expected),
                error_code=INTERNAL_ERROR,
            )
        if len(deployment_classes) > 1:
            raise MlflowException(
                "Plugin registered for the target {} has more than one "
                "child class of BaseDeploymentClient. Raise an issue with"
                " the plugin developers. "
                "Classes found are {}".format(item, deployment_classes)
            )
        elif len(deployment_classes) == 0:
            raise MlflowException(
                "Plugin registered for the target {} has no child class"
                " of BaseDeploymentClient. Raise an issue with the "
                "plugin developers".format(item)
            )
        return plugin_obj
