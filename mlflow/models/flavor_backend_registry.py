"""
Registry of supported flavor backends. Contains a mapping of flavors to flavor backends. This
mapping is used to select suitable flavor when deploying generic MLflow models.

Flavor backend can deploy particular flavor locally to generate predictions, deploy as a local
REST api endpoint, or build a docker image for serving the model locally or remotely.
Not all flavors have a flavor backend.
"""
import logging

from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.file_utils import TempDir
from mlflow.utils.uri import append_to_uri_path

_logger = logging.getLogger(__name__)


def _get_flavor_backend_for_local_model(model=None, build_docker=True, **kwargs):
    from mlflow import pyfunc
    from mlflow.pyfunc.backend import PyFuncBackend
    from mlflow.rfunc.backend import RFuncBackend

    if not model:
        return pyfunc.FLAVOR_NAME, PyFuncBackend({}, **kwargs)

    _flavor_backends = {pyfunc.FLAVOR_NAME: PyFuncBackend, "crate": RFuncBackend}
    for flavor_name, flavor_config in model.flavors.items():
        if flavor_name in _flavor_backends:
            backend = _flavor_backends[flavor_name](flavor_config, **kwargs)
            if build_docker and backend.can_build_image() or backend.can_score_model():
                return flavor_name, backend
    return None, None


def get_flavor_backend(model_uri, **kwargs):
    if model_uri:
        with TempDir() as tmp:
            if ModelsArtifactRepository.is_models_uri(model_uri):
                underlying_model_uri = ModelsArtifactRepository.get_underlying_uri(model_uri)
            else:
                underlying_model_uri = model_uri
            local_path = _download_artifact_from_uri(
                append_to_uri_path(underlying_model_uri, MLMODEL_FILE_NAME), output_path=tmp.path()
            )
            model = Model.load(local_path)
    else:
        model = None
    flavor_name, flavor_backend = _get_flavor_backend_for_local_model(model, **kwargs)
    if flavor_backend is None:
        raise Exception("No suitable flavor backend was found for the model.")
    _logger.info("Selected backend for flavor '%s'", flavor_name)
    return flavor_backend
