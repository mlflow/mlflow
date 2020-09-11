from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS, ErrorCode
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.tracking import MlflowClient
from mlflow.utils.annotations import experimental
from mlflow.utils.logging_utils import eprint


@experimental
def register_model(model_uri, name):
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
    :return: Single :py:class:`mlflow.entities.model_registry.ModelVersion` object created by
             backend.

    .. code-block:: python
        :caption: Example

        from pprint import pprint
        import mlflow

        local_store_uri = "sqlite:///api_mlruns.db"
        mlflow.set_tracking_uri(local_store_uri)

        # Set an existing run_id
        run_id = "acd04001d9874ce5956f701583596cbc"
        model_uri = "runs:/{}".format(run_id)

        # Register the model
        mv = mlflow.register_model(model_uri, "RandomForestRegressionModel")
        pprint("Registered Model Version Info: {}".format(mv))

    .. code-block:: text
        :caption: Output

        Successfully registered model 'RandomForestRegressionModel'.
        Created version '1' of model 'RandomForestRegressionModel'.

        ('Registered Model Version Info=<ModelVersion: '
         "creation_timestamp=1599148895473, current_stage='None', description=None,"
         "last_updated_timestamp=1599148895473, name='RandomForestRegressionModel',"
         "run_id='acd04001d9874ce5956f701583596cbc', run_link=None,"
         "source='./mlruns/0/acd04001d9874ce5956f701583596cbc/artifacts',"
         "status='READY', status_message=None, tags={}, user_id=None, version=1>")
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
        create_version_response = client.create_model_version(name, source, run_id)
    else:
        create_version_response = client.create_model_version(name, source=model_uri, run_id=None)
    eprint(
        "Created version '{version}' of model '{model_name}'.".format(
            version=create_version_response.version, model_name=create_version_response.name
        )
    )
    return create_version_response
