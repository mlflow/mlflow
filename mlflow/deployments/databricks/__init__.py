from mlflow.deployments import BaseDeploymentClient


class DatabricksDeploymentClient(BaseDeploymentClient):
    def create_deployment(self, name, model_uri, flavor=None, config=None, endpoint=None):
        raise NotImplementedError

    def update_deployment(self, name, model_uri=None, flavor=None, config=None, endpoint=None):
        raise NotImplementedError

    def delete_deployment(self, name, config=None, endpoint=None):
        raise NotImplementedError

    def list_deployments(self, endpoint=None):
        raise NotImplementedError

    def get_deployment(self, name, endpoint=None):
        raise NotImplementedError

    def predict(self, deployment_name=None, inputs=None, endpoint=None):
        raise NotImplementedError("TODO")

    def create_endpoint(self, name, config=None):
        raise NotImplementedError("TODO")

    def update_endpoint(self, endpoint, config=None):
        raise NotImplementedError("TODO")

    def delete_endpoint(self, endpoint):
        raise NotImplementedError("TODO")

    def list_endpoints(self):
        raise NotImplementedError("TODO")

    def get_endpoint(self, endpoint):
        raise NotImplementedError("TODO")


def run_local(name, model_uri, flavor=None, config=None):
    pass


def target_help():
    pass
