import json
import platform
import sys
import uuid
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Optional

from mlflow.version import IS_MLFLOW_SKINNY, IS_TRACING_SDK_ONLY, VERSION


class APIStatus(str, Enum):
    UNKNOWN = "unknown"
    SUCCESS = "success"
    FAILURE = "failure"


class ModelType(str, Enum):
    MODEL_PATH = "model_path"
    MODEL_OBJECT = "model_object"
    PYTHON_FUNCTION = "python_function"
    PYTHON_MODEL = "python_model"
    CHAT_MODEL = "chat_model"
    CHAT_AGENT = "chat_agent"
    RESPONSES_AGENT = "responses_agent"
    # pyfunc log_model can accept either python_model or loader_module,
    # we set model type to LOADER_MODULE if loader_module is specified
    LOADER_MODULE = "loader_module"


@dataclass
class BaseParams:
    """
    Base class for params that are logged to telemetry.
    """

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class LogModelParams(BaseParams):
    flavor: str
    model: ModelType
    is_pip_requirements_set: bool = False
    is_extra_pip_requirements_set: bool = False
    is_code_paths_set: bool = False
    is_params_set: bool = False
    is_metadata_set: bool = False


@dataclass
class AutologParams(BaseParams):
    flavor: str
    disable: bool
    log_traces: bool
    log_models: bool


@dataclass
class GenaiEvaluateParams(BaseParams):
    scorers: list[str]
    is_predict_fn_set: bool = False


@dataclass
class APIRecord:
    api_module: str
    api_name: str
    timestamp_ns: int
    params: Optional[BaseParams] = None
    status: APIStatus = APIStatus.UNKNOWN
    duration_ms: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_ns": self.timestamp_ns,
            "api_module": self.api_module,
            "api_name": self.api_name,
            # dump params to string so we can parse them easily in ETL pipeline
            "params": self.params.to_json() if self.params else None,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
        }


@dataclass
class Imports(BaseParams):
    # Classic ML
    anomalib: bool = False
    autogluon: bool = False
    autokeras: bool = False
    catboost: bool = False
    composer: bool = False
    dask: bool = False
    datasets: bool = False
    deepchecks: bool = False
    deepctr: bool = False
    diviner: bool = False
    evaluate: bool = False
    h2o: bool = False
    joblib: bool = False
    lightgbm: bool = False
    merlin: bool = False
    onnx: bool = False
    optuna: bool = False
    paddle: bool = False
    prefect: bool = False
    prefect_ray: bool = False
    prophet: bool = False
    pycaret: bool = False
    ray: bool = False
    recbole: bool = False
    sacred: bool = False
    shap: bool = False
    sklearn: bool = False
    spark: bool = False
    sparseml: bool = False
    statsmodels: bool = False
    xgboost: bool = False
    zenml: bool = False

    # Deep Learning
    accelerate: bool = False
    albumentations: bool = False
    allennlp: bool = False
    catalyst: bool = False
    colossalai: bool = False
    curated_transformers: bool = False  # curated-transformers
    deepchem: bool = False
    detectron2: bool = False
    diffusers: bool = False
    dgl: bool = False
    elegy: bool = False
    fairseq: bool = False
    fastai: bool = False
    flair: bool = False
    flash: bool = False
    flax: bool = False
    huggingface_hub: bool = False
    ignite: bool = False
    jax: bool = False
    jina: bool = False
    keras: bool = False
    keras_core: bool = False  # keras-core
    keras_cv: bool = False
    kornia: bool = False
    lightning: bool = False
    lightning_fabric: bool = False  # lightning-fabric
    mmcls: bool = False
    mmcv: bool = False
    mmdet: bool = False
    mmengine: bool = False
    mmocr: bool = False
    mmseg: bool = False
    monai: bool = False
    nanodet: bool = False
    optimum: bool = False
    orjson: bool = False
    paddlenlp: bool = False
    paddleocr: bool = False
    paddleseg: bool = False
    peft: bool = False
    ppdet: bool = False
    pytorch_lightning: bool = False
    pytorchvideo: bool = False
    segmentation_models_pytorch: bool = False
    sentence_transformers: bool = False
    simpletransformers: bool = False
    skorch: bool = False
    spacy: bool = False
    stability_sdk: bool = False
    syft: bool = False
    tensorflow: bool = False
    timm: bool = False
    torch: bool = False
    torch_geometric: bool = False
    torchdrug: bool = False
    torchtext: bool = False
    torchvision: bool = False
    transformers: bool = False
    trl: bool = False
    trlx: bool = False
    TTS: bool = False

    # Generative AI
    ag2: bool = False
    agno: bool = False
    anthropic: bool = False
    autogen: bool = False
    bedrock: bool = False
    chromadb: bool = False
    cohere: bool = False
    crewai: bool = False
    dspy: bool = False
    gemini: bool = False
    groq: bool = False
    johnsnowlabs: bool = False
    langchain: bool = False
    langflow: bool = False
    langgraph: bool = False
    litellm: bool = False
    llama_index: bool = False
    metaflow: bool = False
    mistral: bool = False
    openai: bool = False
    pinecone: bool = False  # pinecone-client
    promptflow: bool = False
    promptlayer: bool = False
    pydantic_ai: bool = False
    semantic_kernel: bool = False  # semantic-kernel
    smolagents: bool = False
    weaviate: bool = False  # weaviate-client


class SourceSDK(str, Enum):
    MLFLOW_TRACING = "mlflow-tracing"
    MLFLOW = "mlflow"
    MLFLOW_SKINNY = "mlflow-skinny"


def get_source_sdk() -> SourceSDK:
    if IS_TRACING_SDK_ONLY:
        return SourceSDK.MLFLOW_TRACING
    elif IS_MLFLOW_SKINNY:
        return SourceSDK.MLFLOW_SKINNY
    else:
        return SourceSDK.MLFLOW


@dataclass
class TelemetryInfo:
    session_id: str = uuid.uuid4().hex
    source_sdk: str = get_source_sdk().value
    mlflow_version: str = VERSION
    schema_version: int = 1
    python_version: str = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    operating_system: str = platform.platform()
    backend_store_scheme: Optional[str] = None
