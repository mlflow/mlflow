import logging

from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, Schema

_logger = logging.getLogger(__name__)


def _signature_has_required_params(signature: ModelSignature) -> bool:
    """
    PLACEHOLDER: validate that the signature supports the relevant inference type
    """
    return True


def generate_signature_output(index, data, model_config=None, flavor_config=None, params=None):
    """
    PLACEHOLDER
    """
    # Lazy import to avoid circular dependencies. Ideally we should move _LlamaIndexModelWrapper
    # out from __init__.py to avoid this.
    from mlflow.llama_index import _LlamaIndexModelWrapper

    return _LlamaIndexModelWrapper(
        index=index, model_config=model_config, flavor_config=flavor_config
    ).predict(data, params=params)


def infer_signature_from_input_example(
    index, example=None, model_config=None, flavor_config=None
) -> ModelSignature:
    """
    PLACEHOLDER
    """
    return ModelSignature(
        inputs=Schema([ColSpec("string")]),
        outputs=Schema([ColSpec("string")]),
        params={},
    )


def validate_and_resolve_signature(signature: ModelSignature) -> ModelSignature:
    """
    PLACEHOLDER
    """
    return signature
