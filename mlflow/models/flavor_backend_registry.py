"""
Registry of supported flavor backends. Contains a mapping of flavors to flavor backends. This
mapping is used to select suitable flavor when deploying generic MLflow models.

Flavor backend can deploy particular flavor locally to generate predictions or to deploy as a local
REST api endpoint. Not all flavors have a flavor backend.
"""
import mlflow.pyfunc as pyfunc
from mlflow.pyfunc.backend import PyFuncBackend
from mlflow.rfunc.backend import RFuncBackend


_flavor_backends = {
    pyfunc.FLAVOR_NAME: PyFuncBackend,
    "crate": RFuncBackend
}


def get_flavor_backend(model, **kwargs):
    for flavor_name, flavor_config in model.flavors.items():
        if flavor_name in _flavor_backends:
            backend = _flavor_backends[flavor_name](flavor_config, **kwargs)
            if backend.can_score_model():
                return flavor_name, backend
    return None, None
