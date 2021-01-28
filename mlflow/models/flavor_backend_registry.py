"""
Registry of supported flavor backends. Contains a mapping of flavors to flavor backends. This
mapping is used to select suitable flavor when deploying generic MLflow models.

Flavor backend can deploy particular flavor locally to generate predictions, deploy as a local
REST api endpoint, or build a docker image for serving the model locally or remotely.
Not all flavors have a flavor backend.
"""
import mlflow.pyfunc as pyfunc


def get_flavor_backend(model, build_docker=True, **kwargs):
    from mlflow.pyfunc.backend import PyFuncBackend
    from mlflow.rfunc.backend import RFuncBackend

    _flavor_backends = {pyfunc.FLAVOR_NAME: PyFuncBackend, "crate": RFuncBackend}
    for flavor_name, flavor_config in model.flavors.items():
        if flavor_name in _flavor_backends:
            backend = _flavor_backends[flavor_name](flavor_config, **kwargs)
            if build_docker and backend.can_build_image() or backend.can_score_model():
                return flavor_name, backend
    return None, None
