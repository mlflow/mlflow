from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from .base import BaseProvider
from .utils import send_request
from ..config import RouteConfig, MlflowModelServingConfig
from ..constants import MLFLOW_SERVING_RESPONSE_KEY
from ..schemas import completions, chat, embeddings


class MlflowModelServingProvider(BaseProvider):
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
            raise HTTPException(
                status_code=502,
                detail=f"The response is missing the required key: {MLFLOW_SERVING_RESPONSE_KEY}.",
            )
        return response[MLFLOW_SERVING_RESPONSE_KEY]

    @staticmethod
    def _process_payload(payload, key):
        payload = jsonable_encoder(payload, exclude_none=True)

        input_data = payload.pop(key, None)
        request_payload = {"inputs": input_data if isinstance(input_data, list) else [input_data]}

        if payload:
            request_payload["params"] = payload

        return request_payload

    def _process_completions_response_for_mlflow_serving(self, response):
        inference_data = self._extract_mlflow_response_key(response)

        if isinstance(inference_data, dict):
            inference_data = inference_data.get("candidates", inference_data)

        is_list_of_strings = isinstance(inference_data, list) and all(
            isinstance(item, str) for item in inference_data
        )

        if not is_list_of_strings:
            raise HTTPException(
                status_code=502,
                detail=(
                    "The response structure from the MLflow serving endpoint is not "
                    "compatible with the completions route type. The value components must be "
                    "a list of strings."
                ),
            )

        return [{"text": entry, "metadata": {}} for entry in inference_data]

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
            **{
                "candidates": self._process_completions_response_for_mlflow_serving(resp),
                "metadata": {
                    "model": self.config.model.name,
                    "route_type": self.config.route_type,
                },
            }
        )

    def _process_chat_response_for_mlflow_serving(self, response):
        inference_data = self._extract_mlflow_response_key(response)

        if isinstance(inference_data, dict):
            inference_data = inference_data.get("candidates", inference_data)

        is_list_of_strings = isinstance(inference_data, list) and all(
            isinstance(item, str) for item in inference_data
        )

        if not is_list_of_strings:
            raise HTTPException(
                status_code=502,
                detail=(
                    "The response structure from the MLflow serving endpoint is not "
                    "compatible with the chat route type. The value components must be "
                    "a list of strings."
                ),
            )

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
            raise HTTPException(
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
            **{
                "candidates": self._process_chat_response_for_mlflow_serving(resp),
                "metadata": {
                    "model": self.config.model.name,
                    "route_type": self.config.route_type,
                },
            }
        )

    def _process_embeddings_response_for_mlflow_serving(self, response):
        inference_data = self._extract_mlflow_response_key(response)

        is_list_of_list_of_floats = isinstance(inference_data, list) and all(
            isinstance(sublist, list)
            and sublist
            and all(isinstance(item, float) for item in sublist)
            for sublist in inference_data
        )

        if not is_list_of_list_of_floats:
            raise HTTPException(
                status_code=502,
                detail=(
                    "The response structure from the MLflow serving endpoint is not "
                    "compatible with the embeddings route type. The value components must be "
                    "a list of lists of floats."
                ),
            )

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
            payload=self._process_payload(payload, "text"),
        )

        # Example response:
        # {"predictions": [[0.100, -0.234, 0.002, ...], [0.222, -0.111, 0.134, ...]]}

        return embeddings.ResponsePayload(
            **{
                "embeddings": self._process_embeddings_response_for_mlflow_serving(resp),
                "metadata": {
                    "model": self.config.model.name,
                    "route_type": self.config.route_type,
                },
            }
        )
