from typing import Any, Dict, List, Union

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from mlflow.exceptions import MlflowException
from mlflow.gateway.config import MosaicMLConfig, RouteConfig
from mlflow.gateway.constants import (
    MLFLOW_AI_GATEWAY_MOSAICML_MAX_TOKENS_THRESHOLD,
    MLFLOW_AI_GATEWAY_MOSAICML_TOKEN_ESTIMATION_FACTOR,
)
from mlflow.gateway.providers.base import BaseProvider
from mlflow.gateway.providers.utils import rename_payload_keys, send_request
from mlflow.gateway.schemas import chat, completions, embeddings


class MosaicMLProvider(BaseProvider):
    def __init__(self, config: RouteConfig) -> None:
        super().__init__(config)
        if config.model.config is None or not isinstance(config.model.config, MosaicMLConfig):
            raise TypeError(f"Unexpected config type {config.model.config}")
        self.mosaicml_config: MosaicMLConfig = config.model.config

    async def _request(self, model: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"Authorization": f"{self.mosaicml_config.mosaicml_api_key}"}
        return await send_request(
            headers=headers,
            base_url=self.mosaicml_config.mosaicml_api_base
            or "https://models.hosted-on.mosaicml.hosting",
            path=model + "/v1/predict",
            payload=payload,
        )

    @staticmethod
    def _get_used_tokens_estimate(payload):
        """
        Computes the token count estimate for a given payload.
        """
        if isinstance(payload, chat.RequestPayload):
            return sum(len(msg.content.split()) for msg in payload.messages)
        elif isinstance(payload, completions.RequestPayload):
            return len(payload.prompt.split())
        return 0

    # NB: This validation logic can be removed iff the token allocation validation error moves from
    # a 5xx error to a 4xx error.
    def _is_token_count_over_threshold(
        self, payload: Union[chat.RequestPayload, completions.RequestPayload]
    ):
        """
        Performs a conservative estimate of submitted token counts and a requested max_tokens value
        to preempt a server-side retry based on a 5xx return exception from MosaicML for token
        count violating the maximum threshold.
        Since we don't have access to the tokenizer that is used for a given model, the
        conservative estimation of user-provided tokens errs on the side of caution by counting
        each word supplied as consuming 2 tokens.
        """
        used_tokens = self._get_used_tokens_estimate(payload)
        max_tokens = payload.max_tokens or 0
        return (
            used_tokens * MLFLOW_AI_GATEWAY_MOSAICML_TOKEN_ESTIMATION_FACTOR + max_tokens
            > MLFLOW_AI_GATEWAY_MOSAICML_MAX_TOKENS_THRESHOLD
        )

    # NB: as this parser performs no blocking operations, we are intentionally not defining it
    # as async due to the overhead of spawning an additional thread if we did.
    @staticmethod
    def _parse_chat_messages_to_prompt(messages: List[chat.RequestMessage]) -> str:
        """
        This parser is based on the format described in
        https://huggingface.co/blog/llama2#how-to-prompt-llama-2 .
        The expected format is:
            "<s>[INST] <<SYS>>
            {{ system_prompt }}
            <</SYS>>

            {{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s>
            <s>[INST] {{ user_msg_2 }} [/INST]"
        """
        prompt = "<s>"  # Always start with an opening <s> tag
        for m in messages:
            if m.role == "system" or m.role == "user":
                inst = m.content

                # Wrap system messages in <<SYS>> tags
                if m.role == "system":
                    inst = f"<<SYS>> {inst} <</SYS>>"

                # Close the [INST] tag
                inst += " [/INST]"

                # If the previous message was a system/user message,
                # remove previous closing [/INST] tag
                if prompt.endswith("[/INST]"):
                    prompt = prompt[:-7]
                # Otherwise, add an opening [INST] tag
                else:
                    inst = f"[INST] {inst}"
                prompt += inst
            elif m.role == "assistant":
                # Add statement closing/opening tags by default
                prompt += f" {m.content} </s><s>"
            else:
                raise MlflowException.invalid_parameter_value(
                    f"Invalid role {m.role} inputted. Must be one of 'system', "
                    "'user', or 'assistant'.",
                )

        # Remove the last </s><s> tags if they exist to allow for
        # assistant completion prompts.
        if prompt.endswith("</s><s>"):
            prompt = prompt[:-7]
        return prompt

    async def chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        # Validate the payload and raise if token consumption will be violated
        if self._is_token_count_over_threshold(payload):
            raise HTTPException(
                status_code=422,
                detail="The input content entries and the requested max_tokens value are in "
                "excess of the supported total token length of "
                f"{MLFLOW_AI_GATEWAY_MOSAICML_MAX_TOKENS_THRESHOLD}",
            )

        # Extract the List[RequestMessage] from the RequestPayload
        messages = payload.messages
        payload = jsonable_encoder(payload, exclude_none=True)
        # remove the messages from the remaining configuration items
        payload.pop("messages", None)
        self.check_for_model_field(payload)
        key_mapping = {
            "max_tokens": "max_new_tokens",
        }
        for k1, k2 in key_mapping.items():
            if k2 in payload:
                raise HTTPException(
                    status_code=422, detail=f"Invalid parameter {k2}. Use {k1} instead."
                )
        payload = rename_payload_keys(payload, key_mapping)

        # Handle 'prompt' field in payload
        try:
            prompt = [self._parse_chat_messages_to_prompt(messages)]
        except MlflowException as e:
            raise HTTPException(
                status_code=422, detail=f"An invalid request structure was submitted. {e.message}"
            )
        # Construct final payload structure
        final_payload = {"inputs": prompt, "parameters": payload}

        # Input data structure for Mosaic Text Completion endpoint
        #
        # {"inputs": [prompt],
        #  {
        #    "parameters": {
        #      "temperature": 0.2
        #    }
        #   }
        # }

        resp = await self._request(
            self.config.model.name,
            final_payload,
        )
        # Response example
        # (https://docs.mosaicml.com/en/latest/inference.html#text-completion-models)
        # ```
        # {
        #   "outputs": [
        #     "string",
        #   ],
        # }
        # ```
        return chat.ResponsePayload(
            **{
                "candidates": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": c,
                        },
                        "metadata": {},
                    }
                    for c in resp["outputs"]
                ],
                "metadata": {
                    "model": self.config.model.name,
                    "route_type": self.config.route_type,
                },
            }
        )

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        # Validate the payload and raise if token consumption will be violated
        if self._is_token_count_over_threshold(payload):
            raise HTTPException(
                status_code=422,
                detail="The input prompt word length and the requested max_tokens value are in "
                "excess of the supported total token length of "
                f"{MLFLOW_AI_GATEWAY_MOSAICML_MAX_TOKENS_THRESHOLD}",
            )

        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        key_mapping = {
            "max_tokens": "max_new_tokens",
        }
        for k1, k2 in key_mapping.items():
            if k2 in payload:
                raise HTTPException(
                    status_code=400, detail=f"Invalid parameter {k2}. Use {k1} instead."
                )
        payload = rename_payload_keys(payload, key_mapping)

        # Handle 'prompt' field in payload
        prompt = payload.pop("prompt")
        if isinstance(prompt, str):
            prompt = [prompt]

        # Construct final payload structure
        final_payload = {"inputs": prompt, "parameters": payload}

        # Input data structure for Mosaic Text Completion endpoint
        #
        # {"inputs": [prompt],
        #  {
        #    "parameters": {
        #      "temperature": 0.2
        #    }
        #   }
        # }

        resp = await self._request(
            self.config.model.name,
            final_payload,
        )
        # Response example
        # (https://docs.mosaicml.com/en/latest/inference.html#text-completion-models)
        # ```
        # {
        #   "outputs": [
        #     "string",
        #   ],
        # }
        # ```
        return completions.ResponsePayload(
            **{
                "candidates": [
                    {
                        "text": c,
                        "metadata": {},
                    }
                    for c in resp["outputs"]
                ],
                "metadata": {
                    "model": self.config.model.name,
                    "route_type": self.config.route_type,
                },
            }
        )

    async def embeddings(self, payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        key_mapping = {"text": "inputs"}
        for k1, k2 in key_mapping.items():
            if k2 in payload:
                raise HTTPException(
                    status_code=400, detail=f"Invalid parameter {k2}. Use {k1} instead."
                )
        payload = rename_payload_keys(payload, key_mapping)

        # Ensure 'inputs' is a list of strings
        if isinstance(payload["inputs"], str):
            payload["inputs"] = [payload["inputs"]]

        resp = await self._request(
            self.config.model.name,
            payload,
        )
        # Response example
        # (https://docs.mosaicml.com/en/latest/inference.html#text-embedding-models):
        # ```
        # {
        #   "outputs": [
        #     [
        #       3.25,
        #       0.7685547,
        #       2.65625,
        #       ...
        #       -0.30126953,
        #       -2.3554688,
        #       1.2597656
        #     ]
        #   ]
        # }
        # ```
        return embeddings.ResponsePayload(
            **{
                "embeddings": resp["outputs"],
                "metadata": {
                    "model": self.config.model.name,
                    "route_type": self.config.route_type,
                },
            }
        )
