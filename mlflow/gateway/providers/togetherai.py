
from mlflow.gateway.providers.base import BaseProvider, ProviderAdapter
from mlflow.gateway.config import TogetherAIConfig, RouteConfig
from mlflow.gateway.providers.utils import rename_payload_keys, send_request, send_stream_request
from mlflow.gateway.schemas import chat, completions, embeddings


from typing import Dict, Any, AsyncGenerator, AsyncIterable

class TogetherAIAdapter(ProviderAdapter): 


    def _scale_repetition_penalty(openai_frequency_penalty: float, min_repetion_penalty=-3.402823669209385e+38, max_repetion_penalty=3.402823669209385e+38): 

        # Normalize OpenAI penalty to a 0-1 range
        normalized_penalty = (openai_penalty + 2) / 4  # Transforming [-2, 2] to [0, 1] = (repetition_penalty - 1)/(max_repetion_penalty - 1) 


        scale_penalty = normalized_penalty * (max_repetion_penalty - min_repetion_penalty) + min_repetion_penalty


        return scale_penalty
    
    @classmethod
    def model_to_embeddings(cls, resp, config):
        #Response example: (https://docs.together.ai/docs/embeddings-rest)
        #```
        #{
        #  "object": "list",
        #  "data": [
        #    {
        #      "object": "embedding",
        #      "embedding": [
        #        0.44990748,
        #        -0.2521129,
        #        ...
        #        -0.43091708,
        #        0.214978
        #      ],
        #      "index": 0
        #    }
        #  ],
        #  "model": "togethercomputer/m2-bert-80M-8k-retrieval",
        #  "request_id": "840fc1b5bb2830cb-SEA"
        #}
        #```
        return embeddings.ResponsePayload(
            data=[
                embeddings.EmbeddingObject(
                    embedding=item['embedding'],
                    index=item['index'],
                )
                for item in resp['data']
            ], 
            model=config.model.name, 
            usage=embeddings.EmbeddingsUsage(
                prompt_tokens=None,  
                total_tokens=None
            )
        )

    @classmethod
    def model_to_completions(cls, resp, config):
        # Example response (https://docs.together.ai/reference/completions): 
        #{
        #  "id": "8447f286bbdb67b3-SJC",
        #  "choices": [
        #    {
        #      "text": "The capital of France is Paris. It's located in the north-central part of the country and is one of the most famous cities in the world, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and more. Paris is also the cultural, political, and economic center of France."
        #    }
        #  ],
        #  "usage": {
        #    "prompt_tokens": 16,
        #    "completion_tokens": 78,
        #    "total_tokens": 94
        #  },
        #  "created": 1705089226,
        #  "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        #  "object": "text_completion"
        #}


        return completions.ResponsePayload(
            id=resp["id"],
            created=resp["created"],
            model=config.model.name,
            choices=[ 
                completions.Choice(
                    index=idx,
                    text=c["text"],
                    finish_reason=None,
                )
                for idx, c in enumerate(resp["choices"])
            ],
            usage=completions.CompletionsUsage(
                prompt_tokens=resp["usage"]["prompt_tokens"],
                completion_tokens=resp["usage"]["completion_tokens"],
                total_tokens=resp["usage"]["total_tokens"]
            )

        )

    @classmethod
    def model_to_completions_streaming(cls, resp, config):
        raise NotImplementedError

    @classmethod
    def completions_to_model(cls, payload, config):
        key_mapping = {
            # repetition_penalty (-3.402823669209385e+38, 3.402823669209385e+38)
            # frequency_penalty (-2.0, 2.0)
            # presence_penalty (-2.0, 2.0)
            "frequency_penalty": "repetition_penalty",
            # top_logprobs (0, 20)
            # logprobs (-2147483648,2147483648) 
            "top_logprobs": "logprobs"
        }

        payload = rename_payload_keys(payload, key_mapping)

        if "logprobs" in payload and payload["logprobs"] and not "logprobs" in payload:
            raise HTTPException(
                status_code=422, 
                detail="Missing paramater in payload. If flag logprobs parameter is set to True then logsprobs parameter must contain a value."
            )

        # TODO maybe let the api service handle these internally?

        if "logitbias" in payload: 
            raise HTTPException(
                status_code=422, 
                detail="Invalid parameter in payload. Parameter logitbias is not supported for togetherai."
            )

        if "seed" in payload: 
            raise HTTPException(
                status_code=422,
                detail="Invalid parameter in payload. Parameter seed is not suuported of togetherai."
            )

        if "presence_penalty" in payload: 
            raise HTTPException(
                status_code=422,
                detail="Invalid parameter in payload. Parameter presence_penalty is not supported for togetherai."
            )


        return payload

    @classmethod
    def completions_streaming_to_model(cls, payload, config):
        raise NotImplementedError

    @classmethod
    def model_to_chat(cls, resp, config):
        raise NotImplementedError

    @classmethod
    def model_to_chat_streaming(cls, resp, config):
        raise NotImplementedError

    @classmethod
    def chat_to_model(cls, payload, config):
        raise NotImplementedError

    @classmethod
    def chat_streaming_to_model(cls, payload, config):
        raise NotImplementedError

    @classmethod
    def embeddings_to_model(cls, payload, config):
        #Example request (https://docs.together.ai/reference/embeddings): 
        #curl --request POST \
        #   --url https://api.together.xyz/v1/embeddings \
        #   --header 'accept: application/json' \
        #   --header 'content-type: application/json' \
        #   --data '
        #   {
        #     "model": "togethercomputer/m2-bert-80M-8k-retrieval",
        #     "input": "Our solar system orbits the Milky Way galaxy at about 515,000 mph"
        #   }

        # This is just to keep the interface consistent the adapter
        # class is not needed here as the togetherai request similar
        # to the openAI one.

        return payload

class TogetherAIProvider(BaseProvider):
    NAME = "TogetherAI"

    def __init__(self, config: RouteConfig) -> None: 
        super().__init__(config)
        if config.model.config is None or not isinstance(config.model.config, TogetherAIConfig):
            # Should be unreachable
            raise MlflowException.invalid_parameter_value(
                "Invalid config type {config.model.config}"
            )
        self.togetherai_config:  TogetherAIConfig = config.model.config

    @property
    def base_url(self): 
        #togetherai seems to support only this url
        return "https://api.together.xyz/v1"

    @property
    def auth_headers(self): 
        return {"Authorization": f"Bearer {self.togetherai_config.togetherai_api_key}"} 


    def _add_model_to_payload(self, payload: chat.RequestPayload):
        return {"model":self.config.model.name, **payload}

    async def _request(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await send_request(
            headers=self.auth_headers,
            base_url=self.base_url,
            path=path,
            payload=payload,
        )

    async def _stream_request(self, path: str, payload: Dict[str, Any]) -> AsyncGenerator[bytes, None]:
        return send_stream_request(
            headers=self.auth_headers,
            base_url=self.base_url,
            path=path,
            payload=payload,
        )


    def _add_model_to_payload(self, payload: chat.RequestPayload):
        return {"model":self.config.model.name, **payload}

    async def embeddings(self, payload: chat.RequestPayload):
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)
        #self.check_for_model_field(payload)

        resp = await self._request(
            path="embeddings",
            payload={
                "model": self.config.model.name, 
                **TogetherAIAdapter.embeddings_to_model(payload, self.config)
            }
        )

        return TogetherAIAdapter.model_to_embeddings(resp, self.config)

    async def completions(self, payload: chat.RequestPayload) -> completions.ResponsePayload:
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)

        resp = await self._request(
            path="completions",
            payload={
                "model": self.config.model.name, 
                **TogetherAIAdapter.completions_to_model(payload, self.config)
            }
        )

        return TogetherAIAdapter.model_to_completions(resp, self.config)

    async def chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        raise NotImplementedError 

