from typing import Any, Dict
import json
import time

import boto3
import botocore.config
import botocore.exceptions
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from mlflow.gateway.config import AmazonSageMakerConfig, RouteConfig, AWSIdAndKey, AWSRole
from mlflow.gateway.exceptions import AIGatewayConfigException
from mlflow.gateway.providers.base import BaseProvider

from mlflow.gateway.schemas import chat, completions, embeddings

class AmazonSageMakerProvider(BaseProvider):
    def __init__(self, config: RouteConfig) -> None:
        super().__init__(config)
        if config.model.config is None or not isinstance(
            config.model.config, AmazonSageMakerConfig
        ):
            raise TypeError(f"Unexpected config type {config.model.config}")
        self.sagemaker_config: AmazonSageMakerConfig = config.model.config
        self._content_type = "application/json"
        self._client = None
        self._client_created = 0
        self._accept_eula = "true"


    def _client_expired(self):
        if not isinstance(self.sagemaker_config.aws_config, AWSRole):
            return False

        return (
            (time.monotonic_ns() - self._client_created)
            >= (self.sagemaker_config.aws_config.session_length_seconds) * 1_000_000_000,
        )

    def get_sagemaker_client(self):
        if self._client is not None and not self._client_expired():
            return self._client

        session = boto3.Session(**self._construct_session_args())

        try:
            self._client, self._client_created = (
                session.client(
                    service_name="sagemaker-runtime",
                    **self._construct_client_args(session),
                ),
                time.monotonic_ns(),
            )
            return self._client
        except botocore.exceptions.UnknownServiceError as e:
            raise AIGatewayConfigException(
                "Cannot create Amazon Sagemaker client. likely missing credentials or accessing account permissions"
            ) from e

    def _construct_session_args(self):
        session_args = {
            "region_name": self.sagemaker_config.aws_config.aws_region,
        }

        return {k: v for k, v in session_args.items() if v}

    def _construct_client_args(self, session):
        aws_config = self.sagemaker_config.aws_config

        if isinstance(aws_config, AWSRole):
            role = session.client(service_name="sts").assume_role(
                RoleArn=aws_config.aws_role_arn,
                RoleSessionName="ai-gateway-sagemaker",
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

    def model_to_completions(self, resp):
        print(f"resp: {resp}")

        return completions.ResponsePayload(
            **{
                "candidates": [
                    {"text": resp["generation"], "metadata": {}}
                ],
                "metadata": {
                    "model": self.config.model.name,
                    "route_type": self.config.route_type,
                },
            }
        )

    def _request(self, body):
        print(f"Body:{body}")
        try:
            response = self.get_sagemaker_client().invoke_endpoint(
                EndpointName=self.sagemaker_config.endpoint_name,
                Body=json.dumps(body).encode(),
                ContentType=self._content_type,
                Accept=self._content_type,
                CustomAttributes=f"accept_eula={self._accept_eula}"
            )
            return json.loads(response.get("Body").read())

        # TODO work though botocore.exceptions to make this catchable.
        # except botocore.exceptions.ValidationException as e:
        #     raise HTTPException(status_code=422, detail=str(e)) from e

        except botocore.exceptions.ReadTimeoutError as e:
            raise HTTPException(status_code=408) from e

    async def chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        raise HTTPException(
            status_code=404,
            detail="The chat route is not available for the Text Generation Inference provider.",
        )

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        print(f"payload: {payload}")
        self.check_for_model_field(payload)
        payload = jsonable_encoder(payload, exclude_none=True, exclude_defaults=True)
        prompt = payload.pop("prompt")

        parameters = {
            'temperature': payload['temperature'] if 'temperature' in payload else 0.7,
            'max_new_tokens': payload['max_tokens'] if 'max_tokens' in payload else 64,
            'top_p': payload['top_p'] if 'top_p' in payload else 1,
            'return_full_text': False,
        }

        final_payload = {"inputs": prompt, "parameters": parameters}
        response = self._request(final_payload)[0]
        return self.model_to_completions(response)

    async def embeddings(self, payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        raise HTTPException(
            status_code=404,
            detail=(
                "The embedding route is not available for the Text Generation Inference provider."
            ),
        )
