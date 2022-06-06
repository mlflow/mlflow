import os
from mlflow.deployments import BaseDeploymentClient

f_deployment_name = "fake_deployment_name"
f_endpoint_name = "fake_endpoint_name"


class PluginDeploymentClient(BaseDeploymentClient):
    def create_deployment(self, name, model_uri, flavor=None, config=None, endpoint=None):
        if config and config.get("raiseError") == "True":
            raise RuntimeError("Error requested")
        return {"name": f_deployment_name, "flavor": flavor}

    def delete_deployment(self, name, config=None, endpoint=None):
        if config and config.get("raiseError") == "True":
            raise RuntimeError("Error requested")
        return None

    def update_deployment(self, name, model_uri=None, flavor=None, config=None, endpoint=None):
        return {"flavor": flavor}

    def list_deployments(self, endpoint=None):
        if os.environ.get("raiseError") == "True":
            raise RuntimeError("Error requested")
        return [{"name": f_deployment_name}]

    def get_deployment(self, name, endpoint=None):
        return {"key1": "val1", "key2": "val2"}

    def predict(self, deployment_name=None, df=None, endpoint=None):
        return "1"

    def explain(self, deployment_name=None, df=None, endpoint=None):
        return "1"

    def create_endpoint(self, name, config=None):
        if config and config.get("raiseError") == "True":
            raise RuntimeError("Error requested")
        return {"name": f_endpoint_name}

    def update_endpoint(self, endpoint, config=None):
        return None

    def delete_endpoint(self, endpoint):
        return None

    def list_endpoints(self):
        return [{"name": f_endpoint_name}]

    def get_endpoint(self, endpoint):
        return {"name": f_endpoint_name}


def run_local(name, model_uri, flavor=None, config=None):
    # pylint: disable-next=print-function
    print(
        "Deployed locally at the key {} using the model from {}. ".format(name, model_uri)
        + "It's flavor is {} and config is {}".format(flavor, config)
    )


def target_help():
    return "Target help is called"
