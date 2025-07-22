# NB: Kinesis PutRecords API has a limit of 500 records per request
BATCH_SIZE = 500
BATCH_TIME_INTERVAL_SECONDS = 30
MAX_QUEUE_SIZE = 1000
MAX_WORKERS = 1
BASE_URL = "https://api.mlflow-telemetry.io"

PACKAGES_TO_CHECK_IMPORT = [
    # Classic ML
    "catboost",
    "diviner",
    "h2o",
    "lightgbm",
    "prophet",
    "pyspark",
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
