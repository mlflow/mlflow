import time
from typing import Any, Dict

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from mlflow.gateway.config import HuggingFaceTextGenerationInferenceConfig, RouteConfig
from mlflow.gateway.providers.base import BaseProvider
from mlflow.gateway.providers.utils import (
    rename_payload_keys,
    send_request,
)
from mlflow.gateway.schemas import completions


class HFTextGenerationInferenceServerProvider(BaseProvider):
    NAME = "Hugging Face Text Generation Inference"

    def __init__(self, config: RouteConfig) -> None:
        super().__init__(config)
        if config.model.config is None or not isinstance(
            config.model.config, HuggingFaceTextGenerationInferenceConfig
        ):
            raise TypeError(f"Unexpected config type {config.model.config}")
        self.huggingface_config: HuggingFaceTextGenerationInferenceConfig = config.model.config
        self.headers = {"Content-Type": "application/json"}

    async def _request(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await send_request(
            headers=self.headers,
            base_url=self.huggingface_config.hf_server_url,
            path=path,
            payload=payload,
        )

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        key_mapping = {
            "max_tokens": "max_new_tokens",
        }
        for k1, k2 in key_mapping.items():
            if k2 in payload:
                raise HTTPException(
                    status_code=422, detail=f"Invalid parameter {k2}. Use {k1} instead."
                )

        # HF TGI does not support generating multiple candidates.
        n = payload.pop("n", 1)
        if n != 1:
            raise HTTPException(
                status_code=422,
                detail="'n' must be '1' for the Text Generation Inference provider."
                f"Received value: '{n}'.",
            )
        prompt = payload.pop("prompt")
        parameters = rename_payload_keys(payload, key_mapping)

        # The range of HF TGI's temperature is 0-100, but ours is 0-2, so we multiply
        # by 50
        payload["temperature"] = 50 * payload["temperature"]
        # HF TGI does not support 0 temperature
        parameters["temperature"] = max(payload["temperature"], 1e-3)
        parameters["details"] = True
        parameters["decoder_input_details"] = True

        final_payload = {"inputs": prompt, "parameters": parameters}

        resp = await self._request(
            "generate",
            final_payload,
        )

        # Example Response:
        # Documentation: https://huggingface.github.io/text-generation-inference/#/Text%20Generation%20Inference/compat_generate
        # {'details': {'best_of_sequences': [{'finish_reason': 'length',
        #     'generated_text': 'test',
        #     'generated_tokens': 1,
        #     'prefill': [{'id': 0, 'logprob': -0.34, 'text': 'test'}],
        #     'seed': 42,
        #     'tokens': [{'id': 0, 'logprob': -0.34, 'special': False, 'text': 'test'}],
        #     'top_tokens': [[{'id': 0,
        #     'logprob': -0.34,
        #     'special': False,
        #     'text': 'test'}]]}],
        # 'finish_reason': 'length',
        # 'generated_tokens': 1,
        # 'prefill': [{'id': 0, 'logprob': -0.34, 'text': 'test'}],
        # 'seed': 42,
        # 'tokens': [{'id': 0, 'logprob': -0.34, 'special': False, 'text': 'test'}],
        # 'top_tokens': [[{'id': 0,
        #     'logprob': -0.34,
        #     'special': False,
        #     'text': 'test'}]]},
        # 'generated_text': 'test'}
        output_tokens = resp["details"]["generated_tokens"]
        input_tokens = len(resp["details"]["prefill"])
        return completions.ResponsePayload(
            created=int(time.time()),
            object="text_completion",
            model=self.config.model.name,
            choices=[
                completions.Choice(
                    index=0,
                    text=resp["generated_text"],
                    finish_reason=resp["details"]["finish_reason"],
                )
            ],
            usage=completions.CompletionsUsage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
        )
