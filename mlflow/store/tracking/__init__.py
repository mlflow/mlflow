"""
An MLflow tracking server has two properties related to how data is stored: *backend store* to
record ML experiments, runs, parameters, metrics, etc., and *artifact store* to store run
artifacts like models, plots, images, etc.

Several constants are used by multiple backend store implementations.
"""

# Path to default location for backend when using local FileStore or ArtifactStore.
# Also used as default location for artifacts, when not provided, in non local file based backends
# (eg MySQL)
DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH = "./mlruns"
# Used for defining the artifacts uri (`--default-artifact-root`) for the tracking server when
# configuring the server to use the option `--serve-artifacts` mode. This default can be
# overridden by specifying an override to `--default-artifact-root` for the MLflow tracking server.
# When the server is not operating in `--serve-artifacts` configuration, the default artifact
# storage location will be `DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH`.
DEFAULT_ARTIFACTS_URI = "mlflow-artifacts:/"
SEARCH_MAX_RESULTS_DEFAULT = 1000
SEARCH_MAX_RESULTS_THRESHOLD = 50000
GET_METRIC_HISTORY_MAX_RESULTS = 25000
