import time

from pydantic import BaseModel, StrictFloat, StrictStr, ValidationError, validator

from mlflow.gateway.config import MlflowModelServingConfig, RouteConfig
from mlflow.gateway.constants import MLFLOW_SERVING_RESPONSE_KEY
from mlflow.gateway.exceptions import AIGatewayException
from mlflow.gateway.providers.base import BaseProvider
from mlflow.gateway.providers.utils import send_request
from mlflow.gateway.schemas import chat, completions, embeddings


class ServingTextResponse(BaseModel):
    predictions: list[StrictStr]

    @validator("predictions", pre=True)
    def extract_choices(cls, predictions):
        if isinstance(predictions, list) and not predictions:
            raise ValueError("The input list is empty")
        if isinstance(predictions, dict):
            if "choices" not in predictions and len(predictions) > 1:
                raise ValueError(
                    "The dict format is invalid for this route type. Ensure the served model "
                    "returns a dict key containing 'choices'"
                )
            if len(predictions) == 1:
                predictions = next(iter(predictions.values()))
            else:
                predictions = predictions.get("choices", predictions)
            if not predictions:
                raise ValueError("The input list is empty")
        return predictions


class EmbeddingsResponse(BaseModel):
    predictions: list[list[StrictFloat]]

    @validator("predictions", pre=True)
    def validate_predictions(cls, predictions):
        if isinstance(predictions, list) and not predictions:
            raise ValueError("The input list is empty")
        if isinstance(predictions, list) and all(
            isinstance(item, list) and not item for item in predictions
        ):
            raise ValueError("One or more lists in the returned prediction response are empty")
        elif all(isinstance(item, float) for item in predictions):
            return [predictions]
        else:
            return predictions


class MlflowModelServingProvider(BaseProvider):
    NAME = "MLflow Model Serving"
    CONFIG_TYPE = MlflowModelServingConfig

    def __init__(self, config: RouteConfig) -> None:
        super().__init__(config)
        if config.model.config is None or not isinstance(
            config.model.config, MlflowModelServingConfig
        ):
            raise TypeError(f"Invalid config type {config.model.config}")
        self.mlflow_config: MlflowModelServingConfig = config.model.config
        self.headers = {"Content-Type": "application/json"}

    @staticmethod
    def _extract_mlflow_response_key(response):
        if MLFLOW_SERVING_RESPONSE_KEY not in response:
            raise AIGatewayException(
                status_code=502,
                detail=f"The response is missing the required key: {MLFLOW_SERVING_RESPONSE_KEY}.",
            )
        return response[MLFLOW_SERVING_RESPONSE_KEY]

    @staticmethod
    def _process_payload(payload, key):
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)

        input_data = payload.pop(key, None)
        request_payload = {"inputs": input_data if isinstance(input_data, list) else [input_data]}

        if payload:
            request_payload["params"] = payload

        return request_payload

    @staticmethod
    def _process_completions_response_for_mlflow_serving(response):
        try:
            validated_response = ServingTextResponse(**response)
            inference_data = validated_response.predictions
        except ValidationError as e:
            raise AIGatewayException(status_code=502, detail=str(e))

        return [
            completions.Choice(index=idx, text=entry, finish_reason=None)
            for idx, entry in enumerate(inference_data)
        ]

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        # Example request to MLflow REST API server for completions:
        # {
        #     "inputs": ["hi", "hello", "bye"],
        #     "params": {
        #         "temperature": 0.5,
        #         "top_k": 3,
        #     }
        # }

        resp = await send_request(
            headers=self.headers,
            base_url=self.mlflow_config.model_server_url,
            path="invocations",
            payload=self._process_payload(payload, "prompt"),
        )

        # Example response:
        # {"predictions": ["hello", "hi", "goodbye"]}

        return completions.ResponsePayload(
            created=int(time.time()),
            object="text_completion",
            model=self.config.model.name,
            choices=self._process_completions_response_for_mlflow_serving(resp),
            usage=completions.CompletionsUsage(
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
            ),
        )

    def _process_chat_response_for_mlflow_serving(self, response):
        try:
            validated_response = ServingTextResponse(**response)
            inference_data = validated_response.predictions
        except ValidationError as e:
            raise AIGatewayException(status_code=502, detail=str(e))

        return [
            {"message": {"role": "assistant", "content": entry}, "metadata": {}}
            for entry in inference_data
        ]

    async def chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        # Example request to MLflow REST API for chat:
        # {
        #     "inputs": ["question"],
        #     "params": ["temperature": 0.2],
        # }

        payload = self._process_payload(payload, "messages")

        query_count = len(payload["inputs"])
        if query_count > 1:
            raise AIGatewayException(
                status_code=422,
                detail="MLflow chat models are only capable of processing a single query at a "
                f"time. The request submitted consists of {query_count} queries.",
            )

        payload["inputs"] = [payload["inputs"][0]["content"]]

        resp = await send_request(
            headers=self.headers,
            base_url=self.mlflow_config.model_server_url,
            path="invocations",
            payload=payload,
        )

        # Example response:
        # {"predictions": ["answer"]}

        return chat.ResponsePayload(
            created=int(time.time()),
            model=self.config.model.name,
            choices=[
                chat.Choice(
                    index=idx,
                    message=chat.ResponseMessage(
                        role=c["message"]["role"], content=c["message"]["content"]
                    ),
                    finish_reason=None,
                )
                for idx, c in enumerate(self._process_chat_response_for_mlflow_serving(resp))
            ],
            usage=chat.ChatUsage(
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
            ),
        )

    def _process_embeddings_response_for_mlflow_serving(self, response):
        try:
            validated_response = EmbeddingsResponse(**response)
            inference_data = validated_response.predictions
        except ValidationError as e:
            raise AIGatewayException(status_code=502, detail=str(e))

        return inference_data

    async def embeddings(self, payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        # Example request to MLflow REST API server for embeddings:
        # {
        #     "inputs": ["a sentence", "another sentence"],
        #     "params": {
        #         "output_value": "token_embeddings",
        #     }
        # }

        resp = await send_request(
            headers=self.headers,
            base_url=self.mlflow_config.model_server_url,
            path="invocations",
            payload=self._process_payload(payload, "input"),
        )

        # Example response:
        # {"predictions": [[0.100, -0.234, 0.002, ...], [0.222, -0.111, 0.134, ...]]}

        return embeddings.ResponsePayload(
            data=[
                embeddings.EmbeddingObject(
                    embedding=embedding,
                    index=idx,
                )
                for idx, embedding in enumerate(
                    self._process_embeddings_response_for_mlflow_serving(resp)
                )
            ],
            model=self.config.model.name,
            usage=embeddings.EmbeddingsUsage(
                prompt_tokens=None,
                total_tokens=None,
            ),
        )
