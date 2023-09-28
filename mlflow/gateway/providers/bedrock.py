import json
import os
import time
from enum import Enum

import boto3
import botocore.config
import botocore.exceptions
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from mlflow.gateway.config import (AWSBedrockConfig, AWSIdAndKey, AWSRole,
                                   RouteConfig)
from mlflow.gateway.constants import (
    MLFLOW_AI_GATEWAY_ANTHROPIC_DEFAULT_MAX_TOKENS,
    MLFLOW_AI_GATEWAY_ANTHROPIC_MAXIMUM_MAX_TOKENS)
from mlflow.gateway.exceptions import AIGatewayConfigException
from mlflow.gateway.providers.anthropic import AnthropicAdapter
from mlflow.gateway.providers.base import BaseProvider, ProviderAdapter
from mlflow.gateway.providers.cohere import CohereAdapter
from mlflow.gateway.providers.utils import rename_payload_keys, send_request
from mlflow.gateway.schemas import chat, completions, embeddings

AWS_BEDROCK_ANTHROPIC_MAXIMUM_MAX_TOKENS = 8191


class AWSBedrockAnthropicAdapter(AnthropicAdapter):
    @classmethod
    def completions_to_model(cls, payload, config):
        payload = super().completions_to_model(payload, config)

        if "\n\nHuman:" not in payload.get("stop_sequences", []):
            payload.setdefault("stop_sequences", []).append("\n\nHuman:")

        payload["max_tokens_to_sample"] = min(
            payload.get("max_tokens_to_sample", MLFLOW_AI_GATEWAY_ANTHROPIC_DEFAULT_MAX_TOKENS),
            AWS_BEDROCK_ANTHROPIC_MAXIMUM_MAX_TOKENS,
        )
        return payload


class AWSTitanAdapter(ProviderAdapter):
    # TODO handle top_p, top_k, etc.
    @classmethod
    def completions_to_model(cls, payload, config):
        return {
            "inputText": payload.pop("prompt"),
            "textGenerationConfig": rename_payload_keys(
                payload, {"max_tokens": "maxTokenCount", "stop": "stopSequences"}
            ),
        }

    @classmethod
    def model_to_completions(cls, resp, config):
        return completions.ResponsePayload(
            **{
                "metadata": {
                    "model": config.model.name,
                    "route_type": config.route_type,
                },
                "candidates": [
                    {"text": candidate.get("outputText"), "metadata": {}}
                    for candidate in resp.get("results", [])
                ],
            }
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
                "candidate_count": "numResults",
                "max_tokens": "maxTokens",
            },
        )

    @classmethod
    def model_to_completions(cls, resp, config):
        return completions.ResponsePayload(
            **{
                "metadata": {
                    "model": config.model.name,
                    "route_type": config.route_type,
                },
                "candidates": [
                    # second .get ensures item only has key "text",
                    # but might be redundant/undesirable
                    {"text": candidate.get("data", {}).get("text"), "metadata": {}}
                    for candidate in resp.get("completions", [])
                ],
            }
        )

    @classmethod
    def embeddings_to_model(cls, payload, config):
        raise NotImplementedError

    @classmethod
    def model_to_embeddings(cls, resp, config):
        raise NotImplementedError


class AWSBedrockModelProvider(Enum):
    AMAZON = "amazon"
    COHERE = "cohere"
    AI21 = "ai21"
    ANTHROPIC = "anthropic"

    @property
    def adapter(self):
        return AWS_MODEL_PROVIDER_TO_ADAPTER.get(self)

    @classmethod
    def of_str(cls, name: str):
        name = name.lower()

        for opt in cls:
            if opt.name.lower() == name or opt.value.lower() == name:
                return opt


AWS_MODEL_PROVIDER_TO_ADAPTER = {
    AWSBedrockModelProvider.COHERE: CohereAdapter,
    AWSBedrockModelProvider.ANTHROPIC: AWSBedrockAnthropicAdapter,
    AWSBedrockModelProvider.AMAZON: AWSTitanAdapter,
    AWSBedrockModelProvider.AI21: AI21Adapter,
}


class AWSBedrockProvider(BaseProvider):
    def __init__(self, config: RouteConfig):
        super().__init__(config)

        if config.model.config is None or not isinstance(config.model.config, AWSBedrockConfig):
            raise TypeError(f"Invalid config type {config.model.config}")
        self.bedrock_config: AWSBedrockConfig = config.model.config
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
        if self._client is not None and not self._client_expired():
            return self._client

        session_args, config_args, client_args = {}, {}, {}
        aws_config = self.bedrock_config.aws_config

        region_name = aws_config.aws_region or os.environ.get(
            "AWS_REGION", os.environ.get("AWS_DEFAULT_REGION")
        )

        if region_name:
            config_args["region_name"] = session_args["region_name"] = region_name

        # check profile, mostly for dev
        profile_name = os.environ.get("AWS_PROFILE")
        if profile_name:
            session_args["profile_name"] = profile_name

        session = boto3.Session(**{k: v for k, v in session_args.items() if v is not None})

        if isinstance(aws_config, AWSRole):
            role = session.client(service_name="sts").assume_role(
                RoleArn=aws_config.aws_role_arn,
                RoleSessionName="ai-gateway-bedrock",
                DurationSeconds=aws_config.session_length_seconds,
            )
            client_args["aws_access_key_id"] = role["Credentials"]["AccessKeyId"]
            client_args["aws_secret_access_key"] = role["Credentials"]["SecretAccessKey"]
            client_args["aws_session_token"] = role["Credentials"]["SessionToken"]
        elif isinstance(aws_config, AWSIdAndKey):
            client_args["aws_access_key_id"] = aws_config.aws_access_key_id
            client_args["aws_secret_access_key"] = aws_config.aws_secret_access_key
            client_args["aws_session_token"] = aws_config.aws_session_token
        else:
            # defer to standard CredentialProviderChain
            pass

        if config_args:
            client_args["config"] = botocore.config.Config(**config_args)

        try:
            # use default/prevailing credentials
            self._client, self._client_created = (
                session.client(service_name="bedrock", **client_args),
                time.monotonic_ns(),
            )
            return self._client

        except botocore.exceptions.UnknownServiceError as e:
            raise AIGatewayConfigException(
                "Cannot create AWS Bedrock client; ensure boto3/botocore "
                "linked from the AWS Bedrock user guide are installed. "
                "Otherwise likely missing credentials or accessing account without to "
                "AWS Bedrock Private Preview"
            ) from e




            )
            }
            }
        else:
    @property
    def _underlying_provider(self):
        if (not self.config.model.name) or "." not in self.config.model.name:
            return None
        provider = self.config.model.name.split(".")[0]
        return AWSBedrockModelProvider.of_str(provider)

    @property
    def underlying_provider_adapter(self) -> ProviderAdapter:
        provider = self._underlying_provider
        if not provider:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown AWS Bedrock model type {self._underlying_provider}",
            )
        adapter = provider.adapter
        if not adapter:
            raise HTTPException(
                status_code=422,
                detail=f"Don't know how to handle {self._underlying_provider} for AWS Bedrock",
            )
        return adapter

    def _make_request(self, body):
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
            raise HTTPException(status_code=408) from e

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        self.check_for_model_field(payload)
        payload = jsonable_encoder(payload, exclude_none=True, exclude_defaults=True)
        payload = self.underlying_provider_adapter.completions_to_model(payload, self.config)
        response = self._request(payload)
        return self.underlying_provider_adapter.model_to_completions(response, self.config)

    async def chat(self, payload: chat.RequestPayload) -> None:
        # AWS Bedrock does not have a chat endpoint
        raise HTTPException(
            status_code=404, detail="The chat route is not available for AWS Bedrock models."
        )

    async def embeddings(self, payload: embeddings.RequestPayload) -> None:
        # AWS Bedrock does not have an embeddings endpoint
        raise HTTPException(
            status_code=404, detail="The embeddings route is not available for AWS Bedrock models."
        )
