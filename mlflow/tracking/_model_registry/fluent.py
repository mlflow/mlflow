from mlflow.exceptions import MlflowException
from mlflow.entities.model_registry import ModelVersion
from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS, ErrorCode
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.tracking import MlflowClient
from mlflow.utils.logging_utils import eprint
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS


def register_model(
    model_uri, name, await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS
) -> ModelVersion:
    """
    Create a new model version in model registry for the model files specified by ``model_uri``.
    Note that this method assumes the model registry backend URI is the same as that of the
    tracking backend.

    :param model_uri: URI referring to the MLmodel directory. Use a ``runs:/`` URI if you want to
                      record the run ID with the model in model registry. ``models:/`` URIs are
                      currently not supported.
    :param name: Name of the registered model under which to create a new model version. If a
                 registered model with the given name does not exist, it will be created
                 automatically.
    :param await_registration_for: Number of seconds to wait for the model version to finish
                            being created and is in ``READY`` status. By default, the function
                            waits for five minutes. Specify 0 or None to skip waiting.
    :return: Single :py:class:`mlflow.entities.model_registry.ModelVersion` object created by
             backend.

    .. code-block:: python
        :caption: Example

        import mlflow.sklearn
        from sklearn.ensemble import RandomForestRegressor

        mlflow.set_tracking_uri("sqlite:////tmp/mlruns.db")
        params = {"n_estimators": 3, "random_state": 42}

        # Log MLflow entities
        with mlflow.start_run() as run:
           rfr = RandomForestRegressor(**params).fit([[0, 1]], [1])
           mlflow.log_params(params)
           mlflow.sklearn.log_model(rfr, artifact_path="sklearn-model")

        model_uri = "runs:/{}/sklearn-model".format(run.info.run_id)
        mv = mlflow.register_model(model_uri, "RandomForestRegressionModel")
        print("Name: {}".format(mv.name))
        print("Version: {}".format(mv.version))

    .. code-block:: text
        :caption: Output

        Name: RandomForestRegressionModel
        Version: 1
    """
    client = MlflowClient()
    try:
        create_model_response = client.create_registered_model(name)
        eprint("Successfully registered model '%s'." % create_model_response.name)
    except MlflowException as e:
        if e.error_code == ErrorCode.Name(RESOURCE_ALREADY_EXISTS):
            eprint(
                "Registered model '%s' already exists. Creating a new version of this model..."
                % name
            )
        else:
            raise e

    if RunsArtifactRepository.is_runs_uri(model_uri):
        source = RunsArtifactRepository.get_underlying_uri(model_uri)
        (run_id, _) = RunsArtifactRepository.parse_runs_uri(model_uri)
        create_version_response = client.create_model_version(
            name, source, run_id, await_creation_for=await_registration_for
        )
    else:
        create_version_response = client.create_model_version(
            name, source=model_uri, run_id=None, await_creation_for=await_registration_for
        )
    eprint(
        "Created version '{version}' of model '{model_name}'.".format(
            version=create_version_response.version, model_name=create_version_response.name
        )
    )
    return create_version_response
