from functools import wraps
from mlflow.deployments.plugin_manager import DeploymentPlugins

plugin_store = DeploymentPlugins()


def _register_plugin(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if plugin_store.has_registered is None:
            plugin_store.register_entrypoints()
        return f(*args, **kwargs)
    return wrapper


listType = list


@_register_plugin
def create(target, model_uri, flavor=None, *args, **kwargs):
    deployment = plugin_store[target].create(model_uri, flavor, *args, **kwargs)
    if not isinstance(deployment, dict) or \
            all([k in ('deployment_id', 'flavor') for k in deployment]):
        raise TypeError("Deployment creation must return a dictionary with values for "
                        "``deployment_id`` and ``flavor``")
    return deployment


@_register_plugin
def delete(target, deployment_id, **kwargs):
    plugin_store[target].delete(deployment_id, **kwargs)


@_register_plugin
def update(target, deployment_id, rollback=False, model_uri=None, *args, **kwargs):
    plugin_store[target].update(deployment_id, rollback, model_uri, *args, **kwargs)


@_register_plugin
def list(target, *args, **kwargs):
    ids = plugin_store[target].list(*args, **kwargs)
    if not isinstance(ids, listType):
        raise TypeError("IDs must be returned as a ``list``")
    return ids


@_register_plugin
def describe(target, deployment_id, *args, **kwargs):
    desc = plugin_store[target].describe(deployment_id, *args, **kwargs)
    if not isinstance(desc, dict):
        raise TypeError("Description must be returned as a dictionary")
    return desc
