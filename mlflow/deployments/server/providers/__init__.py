from typing import Type

from mlflow.exceptions import MlflowException
from mlflow.gateway.config import Provider
from mlflow.deployments.server.providers.base import BaseProvider


def get_provider(provider: Provider) -> Type[BaseProvider]:
    from mlflow.deployments.server.providers.ai21labs import AI21LabsProvider
    from mlflow.deployments.server.providers.anthropic import AnthropicProvider
    from mlflow.deployments.server.providers.bedrock import AmazonBedrockProvider
    from mlflow.deployments.server.providers.cohere import CohereProvider
    from mlflow.deployments.server.providers.huggingface import HFTextGenerationInferenceServerProvider
    from mlflow.deployments.server.providers.mistral import MistralProvider
    from mlflow.deployments.server.providers.mlflow import MlflowModelServingProvider
    from mlflow.deployments.server.providers.mosaicml import MosaicMLProvider
    from mlflow.deployments.server.providers.openai import OpenAIProvider
    from mlflow.deployments.server.providers.palm import PaLMProvider
    from mlflow.deployments.server.providers.togetherai import TogetherAIProvider

    provider_to_class = {
        Provider.OPENAI: OpenAIProvider,
        Provider.ANTHROPIC: AnthropicProvider,
        Provider.COHERE: CohereProvider,
        Provider.AI21LABS: AI21LabsProvider,
        Provider.MOSAICML: MosaicMLProvider,
        Provider.PALM: PaLMProvider,
        Provider.MLFLOW_MODEL_SERVING: MlflowModelServingProvider,
        Provider.HUGGINGFACE_TEXT_GENERATION_INFERENCE: HFTextGenerationInferenceServerProvider,
        Provider.BEDROCK: AmazonBedrockProvider,
        Provider.MISTRAL: MistralProvider,
        Provider.TOGETHERAI: TogetherAIProvider,
    }
    if prov := provider_to_class.get(provider):
        return prov

    raise MlflowException.invalid_parameter_value(f"Provider {provider} not found")
