import time
from typing import Any

from mlflow.gateway.config import GeminiConfig, RouteConfig
from mlflow.gateway.exceptions import AIGatewayException
from mlflow.gateway.providers.base import BaseProvider, ProviderAdapter
from mlflow.gateway.providers.utils import rename_payload_keys, send_request
from mlflow.gateway.schemas import chat, completions, embeddings

GENERATION_CONFIG_KEY_MAPPING = {
    "stop": "stopSequences",
    "n": "candidateCount",
    "max_tokens": "maxOutputTokens",
    "top_k": "topK",
    "top_p": "topP",
}

GENERATION_CONFIGS = [
    "temperature",
    "stopSequences",
    "candidateCount",
    "maxOutputTokens",
    "topK",
    "topP",
]


class GeminiAdapter(ProviderAdapter):
    @classmethod
    def chat_to_model(cls, payload, config):
        # Documentation: https://ai.google.dev/api/generate-content
        # Example payload for the chat API.
        #
        # {
        #     "contents": [
        #         {
        #             "role": "user",
        #             "parts": [
        #                 {
        #                     "text": "System: You are an advanced AI Assistant"
        #                 }
        #             ]
        #         },
        #         {
        #             "role": "user",
        #             "parts": [
        #                 {
        #                     "text": "Please give the code for addition of two numbers"
        #                 }
        #             ]
        #         }
        #     ],
        #     "generationConfig": {
        #         "temperature": 0.1,
        #         "topP": 1,
        #         "stopSequences": ["\n"],
        #         "candidateCount": 1,
        #         "maxOutputTokens": 100,
        #         "topK": 40
        #     }
        # }

        for k1, k2 in GENERATION_CONFIG_KEY_MAPPING.items():
            if k2 in payload:
                raise AIGatewayException(
                    status_code=422, detail=f"Invalid parameter {k2}. Use {k1} instead."
                )

        if "top_p" in payload and payload["top_p"] > 1:
            raise AIGatewayException(
                status_code=422, detail="top_p should be less than or equal to 1"
            )

        payload = rename_payload_keys(payload, GENERATION_CONFIG_KEY_MAPPING)

        contents, system_message = [], None
        for message in payload["messages"]:
            role = message["role"]

            if role == "assistant":
                role = "model"
            elif role == "system":
                if system_message is None:
                    system_message = {"parts": []}
                system_message["parts"].append({"text": message["content"]})
                continue

            contents.append({"role": role, "parts": [{"text": message["content"]}]})

        gemini_payload = {"contents": contents}

        if system_message:
            gemini_payload["system_instruction"] = system_message

        generation_config = {k: v for k, v in payload.items() if k in GENERATION_CONFIGS}

        if generation_config:
            gemini_payload["generationConfig"] = generation_config

        return gemini_payload

    @classmethod
    def model_to_chat(cls, resp, config):
        # Documentation: https://ai.google.dev/api/generate-content
        #
        # Example Response:
        # {
        #   "candidates": [
        #     {
        #       "content": {
        #         "parts": [
        #           {
        #             "text": "Blue is often seen as a calming and soothing color."
        #           }
        #         ]
        #       },
        #       "finishReason": "stop"
        #     }
        #   ],
        #   "usageMetadata": {
        #     "promptTokenCount": 10,
        #     "candidatesTokenCount": 10,
        #     "totalTokenCount": 20
        #   }
        # }
        choices = []
        for idx, candidate in enumerate(resp.get("candidates", [])):
            content = ""
            if parts := candidate.get("content", {}).get("parts", None):
                content = parts[0].get("text", None)
            if not content:
                continue

            finish_reason = candidate.get("finishReason", "stop")
            if finish_reason == "MAX_TOKENS":
                finish_reason = "length"

            choices.append(
                chat.Choice(
                    index=idx,
                    message=chat.ResponseMessage(
                        role="assistant",
                        content=content,
                    ),
                    finish_reason=finish_reason,
                )
            )

        usage_metadata = resp.get("usageMetadata", {})

        return chat.ResponsePayload(
            id=f"gemini-chat-{int(time.time())}",
            created=int(time.time()),
            object="chat.completion",
            model=config.model.name,
            choices=choices,
            usage=chat.ChatUsage(
                prompt_tokens=usage_metadata.get("promptTokenCount", None),
                completion_tokens=usage_metadata.get("candidatesTokenCount", None),
                total_tokens=usage_metadata.get("totalTokenCount", None),
            ),
        )

    @classmethod
    def completions_to_model(cls, payload, config):
        # Documentation: https://ai.google.dev/api/generate-content
        # Example payload for the completions API.
        #
        # {
        #     "prompt": "What is the world record for flapjack consumption in a single sitting?",
        #     "temperature": 0.1,
        #     "topP": 1,
        #     "stop": ["\n"],
        #     "n": 1,
        #     "max_tokens": 100,
        #     "top_k": 40,
        # }

        chat_payload = {"messages": [{"role": "user", "content": payload.pop("prompt")}], **payload}
        if system_message := payload.pop("system_prompt", None):
            chat_payload["messages"].insert(0, {"role": "system", "content": system_message})
        return cls.chat_to_model(chat_payload, config)

    @classmethod
    def model_to_completions(cls, resp, config):
        # Documentation: https://ai.google.dev/api/generate-content
        #
        # Example Response:
        # {
        #   "candidates": [
        #     {
        #       "content": {
        #         "parts": [
        #           {
        #             "text": "Blue is often seen as a calming and soothing color."
        #           }
        #         ]
        #       },
        #       "finishReason": "stop"
        #     }
        #   ],
        #   "usageMetadata": {
        #     "promptTokenCount": 10,
        #     "candidatesTokenCount": 10,
        #     "totalTokenCount": 20
        #   }
        # }
        choices = []

        for idx, candidate in enumerate(resp.get("candidates", [])):
            text = ""
            if parts := candidate.get("content", {}).get("parts", None):
                text = parts[0].get("text", None)
            if not text:
                continue

            finish_reason = candidate.get("finishReason", "stop")
            if finish_reason == "MAX_TOKENS":
                finish_reason = "length"

            choices.append(
                completions.Choice(
                    index=idx,
                    text=text,
                    finish_reason=finish_reason,
                )
            )

        usage_metadata = resp.get("usageMetadata", {})

        return completions.ResponsePayload(
            created=int(time.time()),
            object="text_completion",
            model=config.model.name,
            choices=choices,
            usage=completions.CompletionsUsage(
                prompt_tokens=usage_metadata.get("promptTokenCount", None),
                completion_tokens=usage_metadata.get("candidatesTokenCount", None),
                total_tokens=usage_metadata.get("totalTokenCount", None),
            ),
        )

    @classmethod
    def embeddings_to_model(cls, payload, config):
        # Example payload for the embedding API.
        # Documentation: https://ai.google.dev/api/embeddings#v1beta.ContentEmbedding
        #
        # {
        #     "requests": [
        #         {
        #             "model": "models/text-embedding-004",
        #             "content": {
        #                 "parts": [
        #                     {
        #                         "text": "What is the meaning of life?"
        #                     }
        #                 ]
        #             }
        #         },
        #         {
        #             "model": "models/text-embedding-004",
        #             "content": {
        #                 "parts": [
        #                     {
        #                         "text": "How much wood would a woodchuck chuck?"
        #                     }
        #                 ]
        #             }
        #         },
        #         {
        #             "model": "models/text-embedding-004",
        #             "content": {
        #                 "parts": [
        #                     {
        #                         "text": "How does the brain work?"
        #                     }
        #                 ]
        #             }
        #         }
        #     ]
        # }

        texts = payload["input"]
        if isinstance(texts, str):
            texts = [texts]
        return (
            {"content": {"parts": [{"text": texts[0]}]}}
            if len(texts) == 1
            else {
                "requests": [
                    {"model": f"models/{config.model.name}", "content": {"parts": [{"text": text}]}}
                    for text in texts
                ]
            }
        )

    @classmethod
    def model_to_embeddings(cls, resp, config):
        # Documentation: https://ai.google.dev/api/embeddings#v1beta.ContentEmbedding
        #
        # Example Response:
        # {
        #   "embeddings": [
        #     {
        #       "values": [
        #         3.25,
        #         0.7685547,
        #         2.65625,
        #         ...,
        #         -0.30126953,
        #         -2.3554688,
        #         1.2597656
        #       ]
        #     }
        #   ]
        # }

        data = [
            embeddings.EmbeddingObject(embedding=item.get("values", []), index=i)
            for i, item in enumerate(resp.get("embeddings") or [resp.get("embedding", {})])
        ]

        # Create and return response payload directly
        return embeddings.ResponsePayload(
            data=data,
            model=config.model.name,
            usage=embeddings.EmbeddingsUsage(
                prompt_tokens=None,
                total_tokens=None,
            ),
        )


class GeminiProvider(BaseProvider):
    NAME = "Gemini"
    CONFIG_TYPE = GeminiConfig

    def __init__(self, config: RouteConfig) -> None:
        super().__init__(config)
        if config.model.config is None or not isinstance(config.model.config, GeminiConfig):
            raise TypeError(f"Unexpected config type {config.model.config}")
        self.gemini_config: GeminiConfig = config.model.config

    @property
    def headers(self):
        return {"x-goog-api-key": self.gemini_config.gemini_api_key}

    @property
    def base_url(self):
        return "https://generativelanguage.googleapis.com/v1beta/models"

    @property
    def adapter_class(self):
        return GeminiAdapter

    async def _request(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await send_request(
            headers=self.headers,
            base_url=self.base_url,
            path=path,
            payload=payload,
        )

    async def embeddings(self, payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)

        embedding_payload = self.adapter_class.embeddings_to_model(payload, self.config)
        # Documentation: https://ai.google.dev/api/embeddings

        # Use the batch endpoint if payload contains "requests"
        if "requests" in embedding_payload:
            endpoint_suffix = ":batchEmbedContents"
        else:
            endpoint_suffix = ":embedContent"

        resp = await self._request(
            f"{self.config.model.name}{endpoint_suffix}",
            embedding_payload,
        )
        return self.adapter_class.model_to_embeddings(resp, self.config)

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)

        completions_payload = self.adapter_class.completions_to_model(payload, self.config)
        # Documentation: https://ai.google.dev/api/generate-content

        if payload.get("stream", False):
            # TODO: Implement streaming for completions
            raise AIGatewayException(
                status_code=422,
                detail="Streaming is not yet supported for completions with Gemini AI Gateway",
            )

        resp = await self._request(
            f"{self.config.model.name}:generateContent",
            completions_payload,
        )

        return self.adapter_class.model_to_completions(resp, self.config)

    async def chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)

        chat_payload = self.adapter_class.chat_to_model(payload, self.config)
        # Documentation: https://ai.google.dev/api/generate-content

        if payload.get("stream", False):
            # TODO: Implement streaming for chat completions
            raise AIGatewayException(
                status_code=422,
                detail="Streaming is not yet supported for chat completions with Gemini AI Gateway",
            )

        resp = await self._request(
            f"{self.config.model.name}:generateContent",
            chat_payload,
        )

        return self.adapter_class.model_to_chat(resp, self.config)
