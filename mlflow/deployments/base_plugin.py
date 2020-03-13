import sys
import abc
from pathlib import Path

from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.models import Model


if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})


class BasePlugin(ABC):

    @staticmethod
    def _get_local_path(model_uri):
        return Path(_download_artifact_from_uri(model_uri))

    @staticmethod
    def _load_model_config(path):
        model_config = path / 'MLmodel'
        return Model.load(model_config)

    @staticmethod
    def _validate_deployment_flavor(model_config, flavor, target_supported):
        """
        Checks that the specified flavor is a supported deployment flavor
        and is contained in the specified model. If one of these conditions
        is not met, an exception is thrown.
        """
        if flavor not in target_supported or flavor not in model_config.flavors:
            model_supported = tuple(model_config.flavors.keys())
            allowed = set(target_supported).intersection(model_supported)
            raise RuntimeError("The specified flavor `{}` is not allowed. The flavor must"
                               " be supported by the target ({}) and should be saved as "
                               "part of ``Model`` ({}) Please use one of the allowed "
                               "flavor: {}".format(flavor, target_supported,
                                                   model_supported, allowed))

    @abc.abstractmethod
    def create(self, model_uri, flavor=None, **kwargs):
        pass

    @abc.abstractmethod
    def delete(self, deployment_id, **kwargs):
        pass

    @abc.abstractmethod
    def update(self, deployment_id, model_uri=None, rollback=False, **kwargs):
        pass

    @abc.abstractmethod
    def list(self, **kwargs):
        pass

    @abc.abstractmethod
    def describe(self, deployment_id, **kwargs):
        pass
