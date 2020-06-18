import os
from mlflow.deployments import BaseDeploymentClient

f_deployment_name = 'fake_deployment_name'


class PluginDeploymentClient(BaseDeploymentClient):
    def create_deployment(self, name, model_uri, flavor=None, config=None):
        if config and config.get('raiseError') == 'True':
            raise RuntimeError("Error requested")
        return {'name': f_deployment_name, 'flavor': flavor}

    def delete_deployment(self, name):
        return None

    def update_deployment(self, name, model_uri=None, flavor=None, config=None):
        return {'flavor': flavor}

    def list_deployments(self):
        if os.environ.get('raiseError') == 'True':
            raise RuntimeError('Error requested')
        return [f_deployment_name]

    def get_deployment(self, name):
        return {'key1': 'val1', 'key2': 'val2'}

    def predict(self, deployment_name, df):
        return 1


def run_local(name, model_uri, flavor=None, config=None):
    print(f"Deployed locally at the key {name} using the model from {model_uri}. "
          f"It's flavor is {flavor} and config is {config}")


def target_help():
    return "Target help is called"
