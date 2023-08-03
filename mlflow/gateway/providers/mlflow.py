from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
import json

from .base import BaseProvider
from .utils import send_request
from ..config import RouteConfig, MLflowConfig
from ..schemas import completions, chat, embeddings


class MLflowProvider(BaseProvider):
    def __init__(self, config: RouteConfig) -> None:
        super().__init__(config)
        if config.model.config is None or not isinstance(config.model.config, MLflowConfig):
            raise TypeError(f"Invalid config type {config.model.config}")
        self.mlflow_config: MLflowConfig = config.model.config
        self.headers = {"Content-Type": "application/json"}

    @staticmethod
    def process_payload(payload, key):
        payload = jsonable_encoder(payload, exclude_none=True)

        input_data = payload.pop(key)
        request_payload = {"inputs": input_data if isinstance(input_data, list) else [input_data]}

        if payload:
            request_payload["params"] = payload

        return request_payload

    @staticmethod
    def process_response(response_text):
        return response_text if isinstance(response_text, str) else json.dumps(response_text)

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
            base_url=self.mlflow_config.mlflow_api_base,
            path="invocations",
            payload=self.process_payload(payload, "prompt"),
        )

        # Example response:
        # {"predictions": ["hello", "hi", "goodbye"]}

        return completions.ResponsePayload(
            **{
                "candidates": [
                    {
                        "text": self.process_response(resp["predictions"]),
                        "metadata": {"finish_reason": "length"},
                    }
                ],
                "metadata": {
                    "model": self.config.model.name,
                    "route_type": self.config.route_type,
                },
            }
        )

    async def chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        # Example request to MLflow REST API for chat:
        # {
        #     "inputs": ["question"],
        #     "params": ["temperature": 0.2],
        # }

        payload = self.process_payload(payload, "messages")

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
            base_url=self.mlflow_config.mlflow_api_base,
            path="invocations",
            payload=payload,
        )

        # Example response:
        # {"predictions": "answer"}

        return chat.ResponsePayload(
            **{
                "candidates": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": self.process_response(resp["predictions"]),
                        },
                        "metadata": {"finish_reason": "length"},
                    }
                ],
                "metadata": {
                    "model": self.config.model.name,
                    "route_type": self.config.route_type,
                },
            }
        )

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
            base_url=self.mlflow_config.mlflow_api_base,
            path="invocations",
            payload=self.process_payload(payload, "text"),
        )

        # Example response:
        # {"predictions": [[0.100, -0.234, 0.002, ...], [0.222, -0.111, 0.134, ...]]}

        return embeddings.ResponsePayload(
            **{
                "embeddings": resp["predictions"],
                "metadata": {
                    "model": self.config.model.name,
                    "route_type": self.config.route_type,
                },
            }
        )
