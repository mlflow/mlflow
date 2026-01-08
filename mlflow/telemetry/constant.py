from mlflow.ml_package_versions import GENAI_FLAVOR_TO_MODULE_NAME, NON_GENAI_FLAVOR_TO_MODULE_NAME

# NB: Kinesis PutRecords API has a limit of 500 records per request
BATCH_SIZE = 500
BATCH_TIME_INTERVAL_SECONDS = 10
MAX_QUEUE_SIZE = 1000
MAX_WORKERS = 1
CONFIG_STAGING_URL = "https://config-staging.mlflow-telemetry.io"
CONFIG_URL = "https://config.mlflow-telemetry.io"
UI_CONFIG_STAGING_URL = "https://d34z9x6fp23d2z.cloudfront.net"
UI_CONFIG_URL = "https://d139nb52glx00z.cloudfront.net"
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

GENAI_MODULES = {
    "agno",
    "anthropic",
    "autogen",
    "chromadb",
    "crewai",
    "dspy",
    "faiss",
    "google.genai",  # gemini
    "groq",
    "haystack",
    "langchain",
    "langgraph",
    "langsmith",
    "litellm",
    "llama_cpp",
    "llama_index.core",
    "milvus",
    "mistralai",
    "openai",
    "pinecone",
    "pydantic_ai",
    "qdrant",
    "ragas",
    "semantic_kernel",
    "smolagents",
    "vllm",
    "weaviate",
} | set(GENAI_FLAVOR_TO_MODULE_NAME.values())

NON_GENAI_MODULES = {
    # Classic ML
    "catboost",
    "h2o",
    "lightgbm",
    "optuna",
    "prophet",
    "pyspark.ml",
    "sklearn",
    "spacy",
    "statsmodels",
    "xgboost",
    # Deep Learning
    "accelerate",
    "bitsandbytes",
    "deepspeed",
    "diffusers",
    "fastai",
    "flash_attn",
    "flax",
    "jax",
    "keras",
    "lightning",
    "mxnet",
    "paddle",
    "peft",
    "sentence_transformers",
    "tensorflow",
    "timm",
    "torch",
    "transformers",
} | set(NON_GENAI_FLAVOR_TO_MODULE_NAME.values()) - {"pyspark"}

MODULES_TO_CHECK_IMPORT = GENAI_MODULES | NON_GENAI_MODULES

# fallback config to use for UI telemetry in case fetch fails
FALLBACK_UI_CONFIG = {
    "disable_ui_telemetry": True,
    "disable_ui_events": [],
    "ui_rollout_percentage": 0,
}
