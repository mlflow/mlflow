import json
import os
import time
from enum import Enum

import boto3
import botocore.config
import botocore.exceptions
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from mlflow.gateway.config import AWSBedrockConfig, AWSIdAndKey, AWSRole, RouteConfig
from mlflow.gateway.constants import (
    MLFLOW_AI_GATEWAY_ANTHROPIC_DEFAULT_MAX_TOKENS,
    MLFLOW_AI_GATEWAY_ANTHROPIC_MAXIMUM_MAX_TOKENS,
)
from mlflow.gateway.exceptions import AIGatewayConfigException
from mlflow.gateway.providers.base import BaseProvider
from mlflow.gateway.providers.utils import rename_payload_keys, send_request
from mlflow.gateway.schemas import chat, completions, embeddings


class AWSBedrockModelProvider(Enum):
    AMAZON = "amazon"
    COHERE = "cohere"
    AI21 = "ai21"
    ANTHROPIC = "anthropic"

    @classmethod
    def of_str(cls, name: str):
        name = name.lower()

        for opt in cls:
            if opt.name.lower() == name or opt.value.lower() == name:
                return opt


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

    @property
    def _underlying_provider(self):
        if (not self.config.model.name) or "." not in self.config.model.name:
            return None
        provider, *_ = self.config.model.name.split(".")
        return AWSBedrockModelProvider.of_str(provider)

    def _prepare_input(self, payload: dict):
        prompt, preped = payload.pop("prompt"), {}

        # TODO handle top_p, top_k, etc.

        if self._underlying_provider is AWSBedrockModelProvider.AMAZON:
            preped["inputText"] = prompt
            preped["textGenerationConfig"] = rename_payload_keys(
                payload, {"max_tokens": "maxTokenCount", "stop": "stopSequences"}
            )
        elif self._underlying_provider is AWSBedrockModelProvider.ANTHROPIC:
            if not prompt.startswith("Human: "):
                prompt = f"Human: {prompt}"
            if not prompt.endswith("\n\nAssistant:"):
                prompt = f"{prompt}\n\nAssistant:"

            preped["prompt"] = prompt

            # rename and ensure
            preped["max_tokens_to_sample"] = max_tokens = payload.pop(
                "max_tokens", MLFLOW_AI_GATEWAY_ANTHROPIC_DEFAULT_MAX_TOKENS
            )

            preped["max_tokens_to_sample"] = min(preped["max_tokens_to_sample"], 8191)
            if max_tokens > MLFLOW_AI_GATEWAY_ANTHROPIC_MAXIMUM_MAX_TOKENS:
                raise HTTPException(
                    status_code=422,
                    detail="Invalid value for max_tokens: cannot exceed "
                    f"{MLFLOW_AI_GATEWAY_ANTHROPIC_MAXIMUM_MAX_TOKENS}.",
                )

            preped = {**preped, **rename_payload_keys(payload, {"stop": "stop_sequences"})}
            if "stop_sequences" not in preped:
                preped["stop_sequences"] = []

            if "\n\nHuman:" not in preped["stop_sequences"]:
                preped["stop_sequences"].append("\n\nHuman:")
        elif self._underlying_provider is AWSBedrockModelProvider.AI21:
            preped = {
                "prompt": prompt,
                **rename_payload_keys(
                    payload,
                    {
                        "stop": "stopSequences",
                        "candidate_count": "numResults",
                        "max_tokens": "maxTokens",
                    },
                ),
            }

        elif self._underlying_provider is AWSBedrockModelProvider.COHERE:
            preped = {
                "prompt": prompt,
                **rename_payload_keys(
                    payload,
                    {
                        "stop": "stop_sequences",
                        "candidate_count": "num_generations",
                    },
                ),
            }
            # The range of Cohere's temperature is 0-5, but ours is 0-1, so we scale it.
            if "temperature" in preped:
                preped["temperature"] *= 5
        else:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown AWS Bedrock model type {self._underlying_provider}",
            )

        return preped

    def _process_output(self, response: dict):
        if self._underlying_provider is AWSBedrockModelProvider.ANTHROPIC:
            return {
                "candidates": [
                    {
                        "text": response.get("completion"),
                        "metadata": {
                            "finish_reason": "stop"
                            if response.get("stop_reason") == "stop_sequence"
                            else "length"
                        },
                    }
                ]
            }
        elif self._underlying_provider is AWSBedrockModelProvider.AMAZON:
            return {
                "candidates": [
                    {"text": candidate.get("outputText"), "metadata": {}}
                    for candidate in response.get("results", [])
                ]
            }
        elif self._underlying_provider is AWSBedrockModelProvider.AI21:
            return {
                "candidates": [
                    # second .get ensures item only has key "text",
                    # but might be redundant/undesirable
                    {"text": candidate.get("data", {}).get("text"), "metadata": {}}
                    for candidate in response.get("completions", [])
                ]
            }
        elif self._underlying_provider is AWSBedrockModelProvider.COHERE:
            return {
                "candidates": [
                    {"text": candidate.get("text"), "metadata": {}}
                    for candidate in response.get("generations", [])
                ]
            }

        else:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown AWS Bedrock model type {self._underlying_provider}",
            )

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
        # except botocore.exceptions.ValidationException as e:
        #     raise HTTPException(status_code=422, detail=str(e)) from e
        except botocore.exceptions.ReadTimeoutError as e:
            raise HTTPException(status_code=408) from e

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        self.check_for_model_field(payload)

        request_input = self._prepare_input(
            jsonable_encoder(payload, exclude_none=True, exclude_defaults=True)
        )
        response_dict = self._process_output(self._make_request(request_input))

        return completions.ResponsePayload(
            metadata={
                "model": self.config.model.name,
                "route_type": self.config.route_type,
            },
            **response_dict,
        )

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
