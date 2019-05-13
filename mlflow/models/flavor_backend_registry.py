import mlflow.pyfunc as pyfunc
from mlflow.pyfunc.backend import PyFuncBackend
from mlflow.rfunc.backend import RFuncBackend


_flavor_backends = {
    pyfunc.FLAVOR_NAME: PyFuncBackend,
    "crate": RFuncBackend
}


def get_flavor_backend(model, no_conda):
    for flavor_name, flavor_config in model.flavors.items():
        if flavor_name in _flavor_backends:
            backend = _flavor_backends[flavor_name](flavor_config)
            if backend.can_score_model(no_conda=no_conda):
                return backend
    return None
