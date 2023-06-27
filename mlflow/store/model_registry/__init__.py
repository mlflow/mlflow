# Path to default location for backend when using local FileStore.
DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH = "./mlruns"

SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT = 100
SEARCH_REGISTERED_MODEL_MAX_RESULTS_THRESHOLD = 1000
# Some backends have a low maximum results threshold; for example, Databricks only allows
# `max_results` request parameter values up to 10,000. Accordingly, **be very careful** when
# increasing this default maximum results value to avoid breaking compatibility with such backends
SEARCH_MODEL_VERSION_MAX_RESULTS_DEFAULT = 10000
SEARCH_MODEL_VERSION_MAX_RESULTS_THRESHOLD = 200_000
