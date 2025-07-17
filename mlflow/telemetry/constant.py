# TODO: Switch to production URL before release
TELEMETRY_URL = "https://api.mlflow-telemetry.io/test/log"
# NB: Kinesis PutRecords API has a limit of 500 records per request
BATCH_SIZE = 500
BATCH_TIME_INTERVAL_SECONDS = 30
MAX_QUEUE_SIZE = 1000
MAX_WORKERS = 1
