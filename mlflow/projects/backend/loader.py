import entrypoints
import logging

ENTRYPOINT_GROUP_NAME = "mlflow.mlproject_backend"

__logger__ = logging.getLogger(__name__)


def load_backend(backend_name):
    # backends from plugin
    try:
        backend_builder = entrypoints.get_single(ENTRYPOINT_GROUP_NAME,
                                                 backend_name).load()
        return backend_builder()
    except entrypoints.NoSuchEntryPoint:
        # TODO Should be a error when all backends are migrated here
        __logger__.warning("Backend '{backend_name}' is not available.",
                           backend_name=backend_name)

    return None
