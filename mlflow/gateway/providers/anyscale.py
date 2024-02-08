import json
from typing import AsyncIterable

from mlflow.exceptions import MlflowException
from mlflow.gateway.config import  AnyscaleConfig, RouteConfig
from mlflow.gateway.providers.base import BaseProvider
from mlflow.gateway.providers.utils import send_request, send_stream_request
from mlflow.gateway.schemas import chat, completions, embeddings
from mlflow.gateway.utils import handle_incomplete_chunks, strip_sse_prefix
from mlflow.utils.uri import append_to_uri_path, append_to_uri_query_params



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
        self.base_url = self.anyscale_config.anyscale_api_base or "https://api.endpoints.anyscale.com/v1"


    def _get_object_type(self,object:str)->str:
        return {"text_completion":"chat.completion"}.get(object)
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
                object=self._get_object_type(resp["object"]),
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

    def _prepare_completion_request_payload(self, payload):
        payload["messages"] = [{"role": "user", "content": payload.pop("prompt")}]
        return payload

    def _prepare_completion_response_payload(self, resp):
        return completions.ResponsePayload(
            id=resp["id"],
            # The chat models response from OpenAI is of object type "chat.completion". Since
            # we're using the completions response format here, we hardcode the "text_completion"
            # object type in the response instead
            object="text_completion",
            created=resp["created"],
            model=resp["model"],
            choices=[
                completions.Choice(
                    index=idx,
                    text=c["message"]["content"],
                    finish_reason=c["finish_reason"],
                )
                for idx, c in enumerate(resp["choices"])
            ],
            usage=completions.CompletionsUsage(
                prompt_tokens=resp["usage"]["prompt_tokens"],
                completion_tokens=resp["usage"]["completion_tokens"],
                total_tokens=resp["usage"]["total_tokens"],
            ),
        )

    async def completions_stream(
        self, payload: completions.RequestPayload
    ) -> AsyncIterable[completions.StreamResponsePayload]:
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        payload = self._prepare_completion_request_payload(payload)

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
            yield completions.StreamResponsePayload(
                id=resp["id"],
                # The chat models response from OpenAI is of object type "chat.completion.chunk".
                # Since we're using the completions response format here, we hardcode the
                # "text_completion_chunk" object type in the response instead
                object="text_completion_chunk",
                created=resp["created"],
                model=resp["model"],
                choices=[
                    completions.StreamChoice(
                        index=c["index"],
                        finish_reason=c["finish_reason"],
                        delta=completions.StreamDelta(
                            content=c["delta"].get("content"),
                        ),
                    )
                    for c in resp["choices"]
                ],
            )

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        payload = self._prepare_completion_request_payload(payload)

        resp = await send_request(
            headers=self._request_headers,
            base_url=self.base_url,
            path="chat/completions",
            payload=self._add_model_to_payload(payload),
        )
        # ```
        return self._prepare_completion_response_payload(resp)

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
