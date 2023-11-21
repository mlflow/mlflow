from mlflow.deployments import BaseDeploymentClient
from mlflow.utils.annotations import experimental


class MLflowDeploymentClient(BaseDeploymentClient):
    """
    TODO
    """

    def create_deployment(self, name, model_uri, flavor=None, config=None, endpoint=None):
        """
        .. warning::
            This method is not implemented for `MLflowDeploymentClient`.
        """
        raise NotImplementedError

    def update_deployment(self, name, model_uri=None, flavor=None, config=None, endpoint=None):
        """
        .. warning::
            This method is not implemented for `MLflowDeploymentClient`.
        """
        raise NotImplementedError

    def delete_deployment(self, name, config=None, endpoint=None):
        """
        .. warning::
            This method is not implemented for `MLflowDeploymentClient`.
        """
        raise NotImplementedError

    def list_deployments(self, endpoint=None):
        """
        .. warning::
            This method is not implemented for `MLflowDeploymentClient`.
        """
        raise NotImplementedError

    def get_deployment(self, name, endpoint=None):
        """
        TODO
        """
        raise NotImplementedError

    def create_endpoint(self, name, config=None):
        """
        .. warning::
            This method is not implemented for `MLflowDeploymentClient`.
        """
        raise NotImplementedError

    def update_endpoint(self, endpoint, config=None):
        """
        .. warning::
            This method is not implemented for `MLflowDeploymentClient`.
        """
        raise NotImplementedError

    def delete_endpoint(self, endpoint):
        """
        .. warning::
            This method is not implemented for `MLflowDeploymentClient`.
        """
        raise NotImplementedError

    @experimental
    def predict(self, deployment_name=None, inputs=None, endpoint=None):
        """
        TODO
        """
        raise NotImplementedError("TODO")

    @experimental
    def list_endpoints(self):
        """
        TODO
        """
        raise NotImplementedError("TODO")

    @experimental
    def get_endpoint(self, endpoint):
        """
        TODO
        """
        raise NotImplementedError("TODO")


def run_local(name, model_uri, flavor=None, config=None):
    pass


def target_help():
    pass
