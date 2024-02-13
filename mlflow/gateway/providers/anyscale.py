import json
from typing import AsyncIterable

from mlflow.exceptions import MlflowException
from mlflow.gateway.config import AnyscaleConfig, RouteConfig
from mlflow.gateway.providers.base import BaseProvider
from mlflow.gateway.providers.utils import send_request, send_stream_request
from mlflow.gateway.schemas import chat, embeddings
from mlflow.gateway.utils import handle_incomplete_chunks, strip_sse_prefix


class AnyscaleProvider(BaseProvider):
    NAME = "Anyscale"

    def __init__(self, config: RouteConfig) -> None:
        super().__init__(config)
        if config.model.config is None or not isinstance(config.model.config, AnyscaleConfig):
            # Should be unreachable
            raise MlflowException.invalid_parameter_value(
                "Invalid config type {config.model.config}"
            )
        self.anyscale_config: AnyscaleConfig = config.model.config
        self.base_url = (
            self.anyscale_config.anyscale_api_base or "https://api.endpoints.anyscale.com/v1"
        )

    def _get_object_type(self, object: str) -> str:
        return {"text_completion": "chat.completion"}.get(object)

    def _get_object_type_for_stream(self, object: str) -> str:
        return {"text_completion": "chat.completion.chunk"}.get(object)

    @property
    def _request_headers(self):
        return {
            "Authorization": f"Bearer {self.anyscale_config.anyscale_api_key}",
        }

    def _add_model_to_payload(self, payload):
        # Add model to payload as described in here
        # https://docs.endpoints.anyscale.com/guides/models#chat-models-1
        return {"model": self.config.model.name, **payload}

    async def chat_stream(
        self, payload: chat.RequestPayload
    ) -> AsyncIterable[chat.StreamResponsePayload]:
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)

        stream = send_stream_request(
            headers=self._request_headers,
            base_url=self.base_url,
            path="chat/completions",
            payload=self._add_model_to_payload(payload),
        )

        async for chunk in handle_incomplete_chunks(stream):
            chunk = chunk.strip()
            if not chunk:
                continue

            data = strip_sse_prefix(chunk.decode("utf-8"))
            if data == "[DONE]":
                return

            resp = json.loads(data)
            yield chat.StreamResponsePayload(
                id=resp["id"],
                object=self._get_object_type_for_stream(resp["object"]),
                created=resp["created"],
                model=resp["model"],
                choices=[
                    chat.StreamChoice(
                        index=c["index"],
                        finish_reason=c["finish_reason"],
                        delta=chat.StreamDelta(
                            role=c["delta"].get("role"), content=c["delta"].get("content")
                        ),
                    )
                    for c in resp["choices"]
                ],
            )

    async def chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)

        resp = await send_request(
            headers=self._request_headers,
            base_url=self.base_url,
            path="chat/completions",
            payload=self._add_model_to_payload(payload),
        )

        return chat.ResponsePayload(
            id=resp["id"],
            object=self._get_object_type(resp["object"]),
            created=resp["created"],
            model=resp["model"],
            choices=[
                chat.Choice(
                    index=idx,
                    message=chat.ResponseMessage(
                        role=c["message"]["role"], content=c["message"]["content"]
                    ),
                    finish_reason=c["finish_reason"],
                )
                for idx, c in enumerate(resp["choices"])
            ],
            usage=chat.ChatUsage(
                prompt_tokens=resp["usage"]["prompt_tokens"],
                completion_tokens=resp["usage"]["completion_tokens"],
                total_tokens=resp["usage"]["total_tokens"],
            ),
        )

    async def embeddings(self, payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        resp = await send_request(
            headers=self._request_headers,
            base_url=self.base_url,
            path="embeddings",
            payload=self._add_model_to_payload(payload),
        )

        # ```
        return embeddings.ResponsePayload(
            data=[
                embeddings.EmbeddingObject(
                    embedding=d["embedding"],
                    index=idx,
                )
                for idx, d in enumerate(resp["data"])
            ],
            model=resp["model"],
            usage=embeddings.EmbeddingsUsage(
                prompt_tokens=resp["usage"]["prompt_tokens"],
                total_tokens=resp["usage"]["total_tokens"],
            ),
        )
