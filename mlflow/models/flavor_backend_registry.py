"""
Registry of supported flavor backends. Contains a mapping of flavors to flavor backends. This
mapping is used to select suitable flavor when deploying generic MLflow models.

Flavor backend can deploy particular flavor locally to generate predictions, deploy as a local
REST api endpoint, or build a docker image for serving the model locally or remotely.
Not all flavors have a flavor backend.
"""
import logging

from mlflow.models.model import Model
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.tracking.artifact_utils import (
    _get_root_uri_and_artifact_path,
)
from mlflow.utils.file_utils import TempDir

_logger = logging.getLogger(__name__)


def _get_flavor_backend_for_local_model(model=None, build_docker=True, **kwargs):
    from mlflow import pyfunc, rfunc
    from mlflow.pyfunc.backend import PyFuncBackend
    from mlflow.rfunc.backend import RFuncBackend

    if not model:
        return pyfunc.FLAVOR_NAME, PyFuncBackend({}, **kwargs)

    backends = {pyfunc.FLAVOR_NAME: PyFuncBackend, rfunc.FLAVOR_NAME: RFuncBackend}
    for flavor, Backend in backends.items():
        if flavor in model.flavors:
            backend = Backend(model.flavors[flavor], **kwargs)
            if (build_docker and backend.can_build_image()) or backend.can_score_model():
                return flavor, backend

    return None, None


def get_flavor_backend(model_uri, **kwargs):
    if model_uri:
        with TempDir() as tmp:
            root_uri, artifact_path = _get_root_uri_and_artifact_path(model_uri)
            artifact_repo = get_artifact_repository(root_uri)
            local_path = artifact_repo.download_artifacts(artifact_path, dst_path=tmp.path())
            model = Model.load(local_path)
    else:
        model = None
    flavor_name, flavor_backend = _get_flavor_backend_for_local_model(model, **kwargs)
    if flavor_backend is None:
        raise Exception("No suitable flavor backend was found for the model.")
    _logger.info("Selected backend for flavor '%s'", flavor_name)
    return flavor_backend
