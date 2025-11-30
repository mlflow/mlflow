import json
import time
from enum import Enum

from mlflow.gateway.config import AmazonBedrockConfig, AWSIdAndKey, AWSRole, EndpointConfig
from mlflow.gateway.constants import (
    MLFLOW_AI_GATEWAY_ANTHROPIC_DEFAULT_MAX_TOKENS,
)
from mlflow.gateway.exceptions import AIGatewayConfigException, AIGatewayException
from mlflow.gateway.providers.anthropic import AnthropicAdapter
from mlflow.gateway.providers.base import BaseProvider, ProviderAdapter
from mlflow.gateway.providers.cohere import CohereAdapter
from mlflow.gateway.providers.utils import rename_payload_keys
from mlflow.gateway.schemas import completions

AWS_BEDROCK_ANTHROPIC_MAXIMUM_MAX_TOKENS = 8191


class AmazonBedrockAnthropicAdapter(AnthropicAdapter):
    @classmethod
    def chat_to_model(cls, payload, config):
        payload = super().chat_to_model(payload, config)
        # "model" keys are not supported in Bedrock"
        payload.pop("model", None)
        payload["anthropic_version"] = "bedrock-2023-05-31"
        return payload

    @classmethod
    def completions_to_model(cls, payload, config):
        payload = super().completions_to_model(payload, config)

        if "\n\nHuman:" not in payload.get("stop_sequences", []):
            payload.setdefault("stop_sequences", []).append("\n\nHuman:")

        payload["max_tokens_to_sample"] = min(
            payload.get("max_tokens_to_sample", MLFLOW_AI_GATEWAY_ANTHROPIC_DEFAULT_MAX_TOKENS),
            AWS_BEDROCK_ANTHROPIC_MAXIMUM_MAX_TOKENS,
        )
        payload["anthropic_version"] = "bedrock-2023-05-31"

        # "model" keys are not supported in Bedrock"
        payload.pop("model", None)
        return payload

    @classmethod
    def model_to_completions(cls, payload, config):
        payload["model"] = config.model.name
        return super().model_to_completions(payload, config)


class AWSTitanAdapter(ProviderAdapter):
    # TODO handle top_p, top_k, etc.
    @classmethod
    def completions_to_model(cls, payload, config):
        n = payload.pop("n", 1)
        if n != 1:
            raise AIGatewayException(
                status_code=422,
                detail=f"'n' must be '1' for AWS Titan models. Received value: '{n}'.",
            )

        # The range of Titan's temperature is 0-1, but ours is 0-2, so we halve it
        if "temperature" in payload:
            payload["temperature"] = 0.5 * payload["temperature"]
        return {
            "inputText": payload.pop("prompt"),
            "textGenerationConfig": rename_payload_keys(
                payload, {"max_tokens": "maxTokenCount", "stop": "stopSequences"}
            ),
        }

    @classmethod
    def model_to_completions(cls, resp, config):
        return completions.ResponsePayload(
            created=int(time.time()),
            object="text_completion",
            model=config.model.name,
            choices=[
                completions.Choice(
                    index=idx,
                    text=candidate.get("outputText"),
                    finish_reason=None,
                )
                for idx, candidate in enumerate(resp.get("results", []))
            ],
            usage=completions.CompletionsUsage(
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
            ),
        )

    @classmethod
    def embeddings_to_model(cls, payload, config):
        raise NotImplementedError

    @classmethod
    def model_to_embeddings(cls, resp, config):
        raise NotImplementedError


class AI21Adapter(ProviderAdapter):
    # TODO handle top_p, top_k, etc.
    @classmethod
    def completions_to_model(cls, payload, config):
        return rename_payload_keys(
            payload,
            {
                "stop": "stopSequences",
                "n": "numResults",
                "max_tokens": "maxTokens",
            },
        )

    @classmethod
    def model_to_completions(cls, resp, config):
        return completions.ResponsePayload(
            created=int(time.time()),
            object="text_completion",
            model=config.model.name,
            choices=[
                completions.Choice(
                    index=idx,
                    text=candidate.get("data", {}).get("text"),
                    finish_reason=None,
                )
                for idx, candidate in enumerate(resp.get("completions", []))
            ],
            usage=completions.CompletionsUsage(
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
            ),
        )

    @classmethod
    def embeddings_to_model(cls, payload, config):
        raise NotImplementedError

    @classmethod
    def model_to_embeddings(cls, resp, config):
        raise NotImplementedError


class AmazonBedrockModelProvider(Enum):
    AMAZON = "amazon"
    COHERE = "cohere"
    AI21 = "ai21"
    ANTHROPIC = "anthropic"

    @property
    def adapter_class(self) -> type[ProviderAdapter]:
        return AWS_MODEL_PROVIDER_TO_ADAPTER.get(self)

    @classmethod
    def of_str(cls, name: str):
        name = name.lower()

        for opt in cls:
            if opt.name.lower() in name or opt.value.lower() in name:
                return opt


AWS_MODEL_PROVIDER_TO_ADAPTER = {
    AmazonBedrockModelProvider.COHERE: CohereAdapter,
    AmazonBedrockModelProvider.ANTHROPIC: AmazonBedrockAnthropicAdapter,
    AmazonBedrockModelProvider.AMAZON: AWSTitanAdapter,
    AmazonBedrockModelProvider.AI21: AI21Adapter,
}


class AmazonBedrockProvider(BaseProvider):
    NAME = "Amazon Bedrock"
    CONFIG_TYPE = AmazonBedrockConfig

    def __init__(self, config: EndpointConfig):
        super().__init__(config)

        if config.model.config is None or not isinstance(config.model.config, AmazonBedrockConfig):
            raise TypeError(f"Invalid config type {config.model.config}")
        self.bedrock_config: AmazonBedrockConfig = config.model.config
        self._client = None
        self._client_created = 0

    def _client_expired(self):
        if not isinstance(self.bedrock_config.aws_config, AWSRole):
            return False

        return (
            (time.monotonic_ns() - self._client_created)
            >= (self.bedrock_config.aws_config.session_length_seconds) * 1_000_000_000,
        )

    def get_bedrock_client(self):
        import boto3
        import botocore.exceptions

        if self._client is not None and not self._client_expired():
            return self._client

        session = boto3.Session(**self._construct_session_args())

        try:
            self._client = session.client(
                service_name="bedrock-runtime", **self._construct_client_args(session)
            )
            self._client_created = time.monotonic_ns()
            return self._client
        except botocore.exceptions.UnknownServiceError as e:
            raise AIGatewayConfigException(
                "Cannot create Amazon Bedrock client; ensure boto3/botocore "
                "linked from the Amazon Bedrock user guide are installed. "
                "Otherwise likely missing credentials or accessing account without to "
                "Amazon Bedrock Private Preview"
            ) from e

    def _construct_session_args(self):
        session_args = {
            "region_name": self.bedrock_config.aws_config.aws_region,
        }

        return {k: v for k, v in session_args.items() if v}

    def _construct_client_args(self, session):
        aws_config = self.bedrock_config.aws_config

        if isinstance(aws_config, AWSRole):
            role = session.client(service_name="sts").assume_role(
                RoleArn=aws_config.aws_role_arn,
                RoleSessionName="ai-gateway-bedrock",
                DurationSeconds=aws_config.session_length_seconds,
            )
            return {
                "aws_access_key_id": role["Credentials"]["AccessKeyId"],
                "aws_secret_access_key": role["Credentials"]["SecretAccessKey"],
                "aws_session_token": role["Credentials"]["SessionToken"],
            }
        elif isinstance(aws_config, AWSIdAndKey):
            return {
                "aws_access_key_id": aws_config.aws_access_key_id,
                "aws_secret_access_key": aws_config.aws_secret_access_key,
                "aws_session_token": aws_config.aws_session_token,
            }
        else:
            return {}

    @property
    def _underlying_provider(self):
        if (not self.config.model.name) or "." not in self.config.model.name:
            return None
        return AmazonBedrockModelProvider.of_str(self.config.model.name)

    @property
    def adapter_class(self) -> type[ProviderAdapter]:
        provider = self._underlying_provider
        if not provider:
            raise AIGatewayException(
                status_code=422,
                detail=f"Unknown Amazon Bedrock model type {self._underlying_provider}",
            )
        adapter = provider.adapter_class
        if not adapter:
            raise AIGatewayException(
                status_code=422,
                detail=f"Don't know how to handle {self._underlying_provider} for Amazon Bedrock",
            )
        return adapter

    def _request(self, body):
        import botocore.exceptions

        try:
            response = self.get_bedrock_client().invoke_model(
                body=json.dumps(body).encode(),
                modelId=self.config.model.name,
                # defaults
                # save=False,
                accept="application/json",
                contentType="application/json",
            )
            return json.loads(response.get("body").read())

        # TODO work though botocore.exceptions to make this catchable.
        # except botocore.exceptions.ValidationException as e:
        #     raise HTTPException(status_code=422, detail=str(e)) from e

        except botocore.exceptions.ReadTimeoutError as e:
            raise AIGatewayException(status_code=408) from e

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        from fastapi.encoders import jsonable_encoder

        self.check_for_model_field(payload)
        payload = jsonable_encoder(payload, exclude_none=True, exclude_defaults=True)
        payload = self.adapter_class.completions_to_model(payload, self.config)
        response = self._request(payload)
        return self.adapter_class.model_to_completions(response, self.config)
