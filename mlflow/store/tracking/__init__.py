"""
An MLflow tracking server has two properties related to how data is stored: *backend store* to
record ML experiments, runs, parameters, metrics, etc., and *artifact store* to store run
artifacts like models, plots, images, etc.

Several constants are used by multiple backend store implementations.
"""

# Path to default location for backend when using local FileStore or ArtifactStore.
# Also used as default location for artifacts, when not provided, in non local file based backends
# (eg MySQL)
import os
import mlflow

DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH = "./mlruns"
DEFAULT_BACKEND_STORE_URI = f"sqlite:///{os.path.abspath(os.path.join(mlflow.__path__[0], DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH))}/default.db"
SEARCH_MAX_RESULTS_DEFAULT = 1000
SEARCH_MAX_RESULTS_THRESHOLD = 50000
