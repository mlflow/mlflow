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
