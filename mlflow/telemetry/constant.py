# NB: Kinesis PutRecords API has a limit of 500 records per request
BATCH_SIZE = 500
BATCH_TIME_INTERVAL_SECONDS = 30
MAX_QUEUE_SIZE = 1000
MAX_WORKERS = 1
CONFIG_STAGING_URL = "https://config-staging.mlflow-telemetry.io"
CONFIG_URL = "https://config.mlflow-telemetry.io"
RETRYABLE_ERRORS = [
    429,  # Throttled
    500,  # Interval Server Error
]
UNRECOVERABLE_ERRORS = [
    400,  # Bad Request
    401,  # Unauthorized
    403,  # Forbidden
    404,  # Not Found
]

PACKAGES_TO_CHECK_IMPORT = [
    # Classic ML
    "catboost",
    "diviner",
    "h2o",
    "lightgbm",
    "prophet",
    "pyspark.ml",
    "sklearn",
    "spacy",
    "statsmodels",
    "xgboost",
    # Deep Learning
    "accelerate",
    "deepspeed",
    "fastai",
    "flax",
    "jax",
    "keras",
    "lightning",
    "mxnet",
    "paddle",
    "sentence_transformers",
    "tensorflow",
    "timm",
    "torch",
    "transformers",
]
