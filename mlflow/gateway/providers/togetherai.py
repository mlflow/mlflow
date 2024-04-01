import json 

from typing import Dict, Any, AsyncGenerator, AsyncIterable

from mlflow.gateway.schemas import chat, completions, embeddings

from mlflow.gateway.providers.base import BaseProvider, ProviderAdapter
from mlflow.gateway.providers.utils import rename_payload_keys, send_request, send_stream_request

from mlflow.gateway.config import TogetherAIConfig, RouteConfig
from mlflow.gateway.utils import handle_incomplete_chunks, strip_sse_prefix


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

            #Response example (after manually calling API): 

            #{'id': '86d8d6e06df86f61-ATH', 'object': 'completion.chunk', 'created': 1711977238, 'choices': [{'index': 0, 'text': ' ', 'logprobs': None, 'finish_reason': None, 'delta': {'token_id': 28705, 'content': ' '}}], 'model': 'mistralai/Mixtral-8x7B-v0.1', 'usage': None}
            # {'id': '86d8d6e06df86f61-ATH', 'object': 'completion.chunk', 'created': 1711977238, 'choices': [{'index': 0, 'text': ' },', 'logprobs': None, 'finish_reason': None, 'delta': {'token_id': 1630, 'content': ' },'}}], 'model': 'mistralai/Mixtral-8x7B-v0.1', 'usage': None}
            # },{'id': '86d8d6e06df86f61-ATH', 'object': 'completion.chunk', 'created': 1711977238, 'choices': [{'index': 0, 'text': '\n', 'logprobs': None, 'finish_reason': None, 'delta': {'token_id': 13, 'content': '\n'}}], 'model': 'mistralai/Mixtral-8x7B-v0.1', 'usage': None}
            #
            #{'id': '86d8d6e06df86f61-ATH', 'object': 'completion.chunk', 'created': 1711977238, 'choices': [{'index': 0, 'text': ' ', 'logprobs': None, 'finish_reason': None, 'delta': {'token_id': 28705, 'content': ' '}}], 'model': 'mistralai/Mixtral-8x7B-v0.1', 'usage': None}
            # {'id': '86d8d6e06df86f61-ATH', 'object': 'completion.chunk', 'created': 1711977238, 'choices': [{'index': 0, 'text': ' {', 'logprobs': None, 'finish_reason': None, 'delta': {'token_id': 371, 'content': ' {'}}], 'model': 'mistralai/Mixtral-8x7B-v0.1', 'usage': None}
            # {{'id': '86d8d6e06df86f61-ATH', 'object': 'completion.chunk', 'created': 1711977238, 'choices': [{'index': 0, 'text': '\n', 'logprobs': None, 'finish_reason': None, 'delta': {'token_id': 13, 'content': '\n'}}], 'model': 'mistralai/Mixtral-8x7B-v0.1', 'usage': None}
            #
            #{'id': '86d8d6e06df86f61-ATH', 'object': 'completion.chunk', 'created': 1711977238, 'choices': [{'index': 0, 'text': '   ', 'logprobs': None, 'finish_reason': None, 'delta': {'token_id': 2287, 'content': '   '}}], 'model': 'mistralai/Mixtral-8x7B-v0.1', 'usage': None}
            #   {'id': '86d8d6e06df86f61-ATH', 'object': 'completion.chunk', 'created': 1711977238, 'choices': [{'index': 0, 'text': ' "', 'logprobs': None, 'finish_reason': None, 'delta': {'token_id': 345, 'content': ' "'}}], 'model': 'mistralai/Mixtral-8x7B-v0.1', 'usage': None}
            # "{'id': '86d8d6e06df86f61-ATH', 'object': 'completion.chunk', 'created': 1711977238, 'choices': [{'index': 0, 'text': 'name', 'logprobs': None, 'finish_reason': None, 'delta': {'token_id': 861, 'content': 'name'}}], 'model': 'mistralai/Mixtral-8x7B-v0.1', 'usage': None}

            ## LAST CHUNK 
            #name{'id': '86d8d6e06df86f61-ATH', 'object': 'completion.chunk', 'created': 1711977238, 'choices': [{'index': 0, 'text': '":', 'logprobs': None, 'finish_reason': 'length', 'delta': {'token_id': 1264, 'content': '":'}}], 'model': 'mistralai/Mixtral-8x7B-v0.1', 'usage': {'prompt_tokens': 17, 'completion_tokens': 200, 'total_tokens': 217}}
            #":[DONE]

        return completions.StreamResponsePayload(
            id=resp.get("id"), 
            created=resp.get("created"), 
            model=config.model.name,
            choices=[
                completions.StreamChoice(
                    index=idx,
                    finish_reason=choice.get("finish_reason"), #TODO this is questionable since the finish reason comes from togetherai api
                    delta=completions.StreamDelta(
                        role=None,
                        content=choice.get("text")
                    )
                )

                for idx, choice in enumerate(resp.get("choices"))
            ],
            # usage is not included in OpenAI StreamResponsePayload
            #usage=completions.CompletionsUsage(
            #    prompt_tokens=usage.get("prompt_tokens") if usage else None,
            #    input_tokens=usage.get("completion_tokens") if usage else None,
            #    total_tokens=usage.get("total_tokens") if usage else None,
            #)
        )

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
        # parameters for streaming completions are the same as the standard completions
        return TogetherAIAdapter.completions_to_model(payload, config)

    @classmethod
    def model_to_chat(cls, resp, config):
        # Example response (https://docs.together.ai/reference/chat-completions): 
        #{
        #   "id": "8448080b880415ea-SJC",
        #   "choices": [
        #    {
        #        "message": {
        #           "role": "assistant",
        #           "content": "San Francisco is a vibrant and culturally rich city located in Northern California. It is known for its steep rolling hills, iconic landmarks, and cool, foggy weather. Here are some of the top attractions in San Francisco:\n\n1. The Golden Gate Bridge: This iconic red suspension bridge spans the Golden Gate Strait and is one of the most famous landmarks in the world.\n2. Alcatraz Island: Once a federal prison, Alcatraz Island is now a popular tourist destination. Visitors can take a ferry to the island and explore the prison buildings and learn about its infamous inmates.\n3. Fisherman's Wharf: This popular tourist area is known for its seafood restaurants, shopping, and attractions such as the sea lion colony at Pier 39 and the USS Hornet Museum.\n4. Chinatown: San Francisco's Chinatown is one of the oldest and largest in North America, and is a great place to explore Chinese culture, cuisine, and shopping.\n5. The Presidio: This national park is located at the northern tip of the San Francisco Peninsula and offers hiking trails, beaches, and stunning views of the Golden Gate Bridge.\n6. The Painted Ladies: These colorful Victorian homes are one of the most photographed sites in San Francisco.\n7. Cable Cars: These historic cable cars offer a unique way to get around the city's steep hills and are a fun way to see the sights.\n8. Union Square: This central plaza is surrounded by shops, restaurants, and theaters, and is a popular gathering place for both locals and tourists.\n9. The Ferry Building Marketplace: This historic building houses a food hall featuring local artisanal food and drink vendors.\n10. The Exploratorium: This interactive science museum is located on the Embarcadero and offers hands-on exhibits and educational programs for all ages.\n\nSan Francisco is also known for its cultural diversity, progressive values, and vibrant arts scene, making it a must-visit destination for any traveler."
        #         }
        #     }
        #   ],
        #   "usage": {
        #     "prompt_tokens": 31,
        #     "completion_tokens": 455,
        #     "total_tokens": 486
        #   },
        #   "created": 1705090115,
        #   "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        #   "object": "chat.completion"
        #}
        return chat.ResponsePayload(
            id=resp["id"],
            object="chat.completion",
            created=resp["created"],
            model=config.model.name,
            choices=[
                chat.Choice(
                    index=idx,
                    message=chat.ResponseMessage(
                        role="assistant",
                        content=c["message"]["content"],
                    ),
                    finish_reason=None,
                )
                for idx, c in enumerate(resp["choices"])
            ],
            usage=chat.ChatUsage(
                prompt_tokens=resp["usage"]["prompt_tokens"],
                completion_tokens=resp["usage"]["completion_tokens"],
                total_tokens=resp["usage"]["total_tokens"],
            ),
        )

    @classmethod
    def model_to_chat_streaming(cls, resp, config):
        raise NotImplementedError

    @classmethod
    def chat_to_model(cls, payload, config):

        if "prompt" in payload: 
            raise HTTPException(
                status_code=422,
                detail="Parameter prompt used in chat endpoint. You should provide a list of messages, meaning the conversation so far.",
            )

        # completions and chat endpoint contain the same parameters
        return TogetherAIAdapter.completions_to_model(payload, config) 

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


    async def embeddings(self, payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)

        resp = await self._request(
            path="embeddings",
            payload={
                "model": self.config.model.name, 
                **TogetherAIAdapter.embeddings_to_model(payload, self.config)
            }
        )

        return TogetherAIAdapter.model_to_embeddings(resp, self.config)
    
    # WARNING CHANGING THE ORDER OF METHODS HERE FOR SOME REASON CAUSES AN ERROR
    # completions MODULE IS INTERPERTED AS THE completions function
    async def completions_stream(self, payload: completions.RequestPayload) -> AsyncIterable[completions.StreamResponsePayload]:
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)

        stream = await self._stream_request(
            path="completions",
            payload={
                "model": self.config.model.name, 
                **TogetherAIAdapter.completions_streaming_to_model(payload, self.config)
            }
        )

        async for chunk in stream:
            chunk = chunk.strip()
            if not chunk:
                continue

            
            chunk = strip_sse_prefix(chunk.decode("utf-8"))
            if chunk == "[DONE]":
                return

            resp = json.loads(chunk)
            yield TogetherAIAdapter.model_to_completions_streaming(resp, self.config)


    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
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
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)

        resp = await self._request(
            path="chat/completions",
            payload={
                "model": self.config.model.name, 
                **TogetherAIAdapter.chat_to_model(payload, self.config)
            }
        )

        return TogetherAIAdapter.model_to_chat(resp, self.config)

