from mlflow.deployments.plugin_manager import DeploymentPlugins

listType = list
plugin_store = DeploymentPlugins()


# TODO: standardise exceptions


def create(target, model_uri, flavor=None, **kwargs):
    deployment = plugin_store[target].create(model_uri, flavor, **kwargs)
    if not isinstance(deployment, dict) or \
            not all([k in ('deployment_id', 'flavor') for k in deployment]):
        raise TypeError("Deployment creation must return a dictionary with values for "
                        "``deployment_id`` and ``flavor``")
    return deployment


def delete(target, deployment_id, **kwargs):
    plugin_store[target].delete(deployment_id, **kwargs)


def update(target, deployment_id, model_uri=None, rollback=False, **kwargs):
    if all((rollback, model_uri)):
        raise RuntimeError("``update`` has got both ``model_uri`` and ``rollback``")
    plugin_store[target].update(deployment_id, rollback, model_uri, **kwargs)


# TODO: It's a good practise to avoid using ``list`` here.
def list(target, **kwargs):
    ids = plugin_store[target].list(**kwargs)
    if not isinstance(ids, listType):
        raise TypeError("IDs must be returned as a ``list``")
    return ids


def describe(target, deployment_id, **kwargs):
    desc = plugin_store[target].describe(deployment_id, **kwargs)
    if not isinstance(desc, dict):
        raise TypeError("Description must be returned as a dictionary")
    return desc
