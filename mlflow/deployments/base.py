"""
This module contains the base interface implemented by MLflow model deployment plugins.
In particular, a valid deployment plugin module must implement:

1. Exactly one client class subclassed from :py:class:`BaseDeploymentClient`, exposing the primary
   user-facing APIs used to manage deployments.
2. :py:func:`run_local`, for testing deployment by deploying a model locally
3. :py:func:`target_help`, which returns a help message describing target-specific URI format
   and deployment config
"""

import abc

from mlflow.exceptions import MlflowException
from mlflow.utils.annotations import developer_stable


def run_local(target, name, model_uri, flavor=None, config=None):
    """Deploys the specified model locally, for testing. This function should be defined
    within the plugin module. Also note that this function has a signature which is very
    similar to :py:meth:`BaseDeploymentClient.create_deployment` since both does logically
    similar operation.

    .. Note::
        This function is kept here only for documentation purpose and not implementing the
        actual feature. It should be implemented in the plugin's top level namescope and should
        be callable with ``plugin_module.run_local``

    Args:
        target: Which target to use. This information is used to call the appropriate plugin.
        name: Unique name to use for deployment. If another deployment exists with the same
            name, create_deployment will raise a
            :py:class:`mlflow.exceptions.MlflowException`.
        model_uri: URI of model to deploy.
        flavor: (optional) Model flavor to deploy. If unspecified, default flavor is chosen.
        config: (optional) Dict containing updated target-specific config for the deployment.

    Returns:
        None
    """
    raise NotImplementedError(
        "This function should be implemented in the deployment plugin. It is "
        "kept here only for documentation purpose and shouldn't be used in "
        "your application"
    )


def target_help():
    """
    .. Note::
        This function is kept here only for documentation purpose and not implementing the
        actual feature. It should be implemented in the plugin's top level namescope and should
        be callable with ``plugin_module.target_help``

    Return a string containing detailed documentation on the current deployment target, to be
    displayed when users invoke the ``mlflow deployments help -t <target-name>`` CLI. This
    method should be defined within the module specified by the plugin author.
    The string should contain:

    * An explanation of target-specific fields in the ``config`` passed to ``create_deployment``,
      ``update_deployment``
    * How to specify a ``target_uri`` (e.g. for AWS SageMaker, ``target_uri`` have a scheme of
      "sagemaker:/<aws-cli-profile-name>", where aws-cli-profile-name is the name of an AWS
      CLI profile https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-profiles.html)
    * Any other target-specific details.

    """
    raise NotImplementedError(
        "This function should be implemented in the deployment plugin. It is "
        "kept here only for documentation purpose and shouldn't be used in "
        "your application"
    )


@developer_stable
class BaseDeploymentClient(abc.ABC):
    """
    Base class exposing Python model deployment APIs.

    Plugin implementors should define target-specific deployment logic via a subclass of
    ``BaseDeploymentClient`` within the plugin module, and customize method docstrings with
    target-specific information.

    .. Note::
        Subclasses should raise :py:class:`mlflow.exceptions.MlflowException` in error cases (e.g.
        on failure to deploy a model).
    """

    def __init__(self, target_uri):
        self.target_uri = target_uri

    @abc.abstractmethod
    def create_deployment(self, name, model_uri, flavor=None, config=None, endpoint=None):
        """
        Deploy a model to the specified target. By default, this method should block until
        deployment completes (i.e. until it's possible to perform inference with the deployment).
        In the case of conflicts (e.g. if it's not possible to create the specified deployment
        without due to conflict with an existing deployment), raises a
        :py:class:`mlflow.exceptions.MlflowException` or an `HTTPError` for remote
        deployments. See target-specific plugin documentation
        for additional detail on support for asynchronous deployment and other configuration.

        Args:
            name: Unique name to use for deployment. If another deployment exists with the same
                name, raises a :py:class:`mlflow.exceptions.MlflowException`
            model_uri: URI of model to deploy
            flavor: (optional) Model flavor to deploy. If unspecified, a default flavor
                will be chosen.
            config: (optional) Dict containing updated target-specific configuration for the
                deployment
            endpoint: (optional) Endpoint to create the deployment under. May not be supported
                by all targets

        Returns:
            Dict corresponding to created deployment, which must contain the 'name' key.

        """
        pass

    @abc.abstractmethod
    def update_deployment(self, name, model_uri=None, flavor=None, config=None, endpoint=None):
        """
        Update the deployment with the specified name. You can update the URI of the model, the
        flavor of the deployed model (in which case the model URI must also be specified), and/or
        any target-specific attributes of the deployment (via `config`). By default, this method
        should block until deployment completes (i.e. until it's possible to perform inference
        with the updated deployment). See target-specific plugin documentation for additional
        detail on support for asynchronous deployment and other configuration.

        Args:
            name: Unique name of deployment to update.
            model_uri: URI of a new model to deploy.
            flavor: (optional) new model flavor to use for deployment. If provided,
                ``model_uri`` must also be specified. If ``flavor`` is unspecified but
                ``model_uri`` is specified, a default flavor will be chosen and the
                deployment will be updated using that flavor.
            config: (optional) dict containing updated target-specific configuration for the
                deployment.
            endpoint: (optional) Endpoint containing the deployment to update. May not be
                supported by all targets.

        Returns:
            None

        """
        pass

    @abc.abstractmethod
    def delete_deployment(self, name, config=None, endpoint=None):
        """Delete the deployment with name ``name`` from the specified target.

        Deletion should be idempotent (i.e. deletion should not fail if retried on a non-existent
        deployment).

        Args:
            name: Name of deployment to delete
            config: (optional) dict containing updated target-specific configuration for the
                deployment
            endpoint: (optional) Endpoint containing the deployment to delete. May not be
                supported by all targets

        Returns:
            None
        """
        pass

    @abc.abstractmethod
    def list_deployments(self, endpoint=None):
        """List deployments.

        This method is expected to return an unpaginated list of all
        deployments (an alternative would be to return a dict with a 'deployments' field
        containing the actual deployments, with plugins able to specify other fields, e.g.
        a next_page_token field, in the returned dictionary for pagination, and to accept
        a `pagination_args` argument to this method for passing pagination-related args).

        Args:
            endpoint: (optional) List deployments in the specified endpoint. May not be
                supported by all targets

        Returns:
            A list of dicts corresponding to deployments. Each dict is guaranteed to
            contain a 'name' key containing the deployment name. The other fields of
            the returned dictionary and their types may vary across deployment targets.
        """
        pass

    @abc.abstractmethod
    def get_deployment(self, name, endpoint=None):
        """
        Returns a dictionary describing the specified deployment, throwing either a
        :py:class:`mlflow.exceptions.MlflowException` or an `HTTPError` for remote
        deployments if no deployment exists with the provided ID.
        The dict is guaranteed to contain an 'name' key containing the deployment name.
        The other fields of the returned dictionary and their types may vary across
        deployment targets.

        Args:
            name: ID of deployment to fetch.
            endpoint: (optional) Endpoint containing the deployment to get. May not be
                supported by all targets.

        Returns:
            A dict corresponding to the retrieved deployment. The dict is guaranteed to
            contain a 'name' key corresponding to the deployment name. The other fields of
            the returned dictionary and their types may vary across targets.
        """
        pass

    @abc.abstractmethod
    def predict(self, deployment_name=None, inputs=None, endpoint=None):
        """Compute predictions on inputs using the specified deployment or model endpoint.

        Note that the input/output types of this method match those of `mlflow pyfunc predict`.

        Args:
            deployment_name: Name of deployment to predict against.
            inputs: Input data (or arguments) to pass to the deployment or model endpoint for
                inference.
            endpoint: Endpoint to predict against. May not be supported by all targets.

        Returns:
            A :py:class:`mlflow.deployments.PredictionsResponse` instance representing the
            predictions and associated Model Server response metadata.

        """
        pass

    def predict_stream(self, deployment_name=None, inputs=None, endpoint=None):
        """
        Submit a query to a configured provider endpoint, and get streaming response

        Args:
            deployment_name: Name of deployment to predict against.
            inputs: The inputs to the query, as a dictionary.
            endpoint: The name of the endpoint to query.

        Returns:
            An iterator of dictionary containing the response from the endpoint.
        """
        raise NotImplementedError()

    def explain(self, deployment_name=None, df=None, endpoint=None):
        """
        Generate explanations of model predictions on the specified input pandas Dataframe
        ``df`` for the deployed model. Explanation output formats vary by deployment target,
        and can include details like feature importance for understanding/debugging predictions.

        Args:
            deployment_name: Name of deployment to predict against
            df: Pandas DataFrame to use for explaining feature importance in model prediction
            endpoint: Endpoint to predict against. May not be supported by all targets

        Returns:
            A JSON-able object (pandas dataframe, numpy array, dictionary), or
            an exception if the implementation is not available in deployment target's class
        """
        raise MlflowException(
            "Computing model explanations is not yet supported for this deployment target"
        )

    def create_endpoint(self, name, config=None):
        """
        Create an endpoint with the specified target. By default, this method should block until
        creation completes (i.e. until it's possible to create a deployment within the endpoint).
        In the case of conflicts (e.g. if it's not possible to create the specified endpoint
        due to conflict with an existing endpoint), raises a
        :py:class:`mlflow.exceptions.MlflowException` or an `HTTPError` for remote
        deployments. See target-specific plugin documentation
        for additional detail on support for asynchronous creation and other configuration.

        Args:
            name: Unique name to use for endpoint. If another endpoint exists with the same
                name, raises a :py:class:`mlflow.exceptions.MlflowException`.
            config: (optional) Dict containing target-specific configuration for the
                endpoint.

        Returns:
            Dict corresponding to created endpoint, which must contain the 'name' key.

        """
        raise MlflowException(
            "Method is unimplemented in base client. Implementation should be "
            "provided by specific target plugins."
        )

    def update_endpoint(self, endpoint, config=None):
        """
        Update the endpoint with the specified name. You can update any target-specific attributes
        of the endpoint (via `config`). By default, this method should block until the update
        completes (i.e. until it's possible to create a deployment within the endpoint). See
        target-specific plugin documentation for additional detail on support for asynchronous
        update and other configuration.

        Args:
            endpoint: Unique name of endpoint to update
            config: (optional) dict containing target-specific configuration for the
                endpoint

        Returns:
            None

        """
        raise MlflowException(
            "Method is unimplemented in base client. Implementation should be "
            "provided by specific target plugins."
        )

    def delete_endpoint(self, endpoint):
        """
        Delete the endpoint from the specified target. Deletion should be idempotent (i.e. deletion
        should not fail if retried on a non-existent deployment).

        Args:
            endpoint: Name of endpoint to delete

        Returns:
            None
        """
        raise MlflowException(
            "Method is unimplemented in base client. Implementation should be "
            "provided by specific target plugins."
        )

    def list_endpoints(self):
        """
        List endpoints in the specified target. This method is expected to return an
        unpaginated list of all endpoints (an alternative would be to return a dict with
        an 'endpoints' field containing the actual endpoints, with plugins able to specify
        other fields, e.g. a next_page_token field, in the returned dictionary for pagination,
        and to accept a `pagination_args` argument to this method for passing
        pagination-related args).

        Returns:
            A list of dicts corresponding to endpoints. Each dict is guaranteed to
            contain a 'name' key containing the endpoint name. The other fields of
            the returned dictionary and their types may vary across targets.
        """
        raise MlflowException(
            "Method is unimplemented in base client. Implementation should be "
            "provided by specific target plugins."
        )

    def get_endpoint(self, endpoint):
        """
        Returns a dictionary describing the specified endpoint, throwing a
        py:class:`mlflow.exception.MlflowException` or an `HTTPError` for remote
        deployments if no endpoint exists with the provided
        name.
        The dict is guaranteed to contain an 'name' key containing the endpoint name.
        The other fields of the returned dictionary and their types may vary across targets.

        Args:
            endpoint: Name of endpoint to fetch

        Returns:
            A dict corresponding to the retrieved endpoint. The dict is guaranteed to
            contain a 'name' key corresponding to the endpoint name. The other fields of
            the returned dictionary and their types may vary across targets.
        """
        raise MlflowException(
            "Method is unimplemented in base client. Implementation should be "
            "provided by specific target plugins."
        )
