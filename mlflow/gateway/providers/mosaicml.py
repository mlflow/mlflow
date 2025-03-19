import time
from contextlib import contextmanager
from typing import Any

from mlflow.exceptions import MlflowException
from mlflow.gateway.config import MosaicMLConfig, RouteConfig
from mlflow.gateway.exceptions import AIGatewayException
from mlflow.gateway.providers.base import BaseProvider
from mlflow.gateway.providers.utils import rename_payload_keys, send_request
from mlflow.gateway.schemas import chat, completions, embeddings


class MosaicMLProvider(BaseProvider):
    NAME = "MosaicML"
    CONFIG_TYPE = MosaicMLConfig

    def __init__(self, config: RouteConfig) -> None:
        super().__init__(config)
        if config.model.config is None or not isinstance(config.model.config, MosaicMLConfig):
            raise TypeError(f"Unexpected config type {config.model.config}")
        self.mosaicml_config: MosaicMLConfig = config.model.config

    async def _request(self, model: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Authorization": f"{self.mosaicml_config.mosaicml_api_key}"}
        return await send_request(
            headers=headers,
            base_url=self.mosaicml_config.mosaicml_api_base
            or "https://models.hosted-on.mosaicml.hosting",
            path=model + "/v1/predict",
            payload=payload,
        )

    # NB: as this parser performs no blocking operations, we are intentionally not defining it
    # as async due to the overhead of spawning an additional thread if we did.
    @staticmethod
    def _parse_chat_messages_to_prompt(messages: list[chat.RequestMessage]) -> str:
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
        from fastapi.encoders import jsonable_encoder

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
                raise AIGatewayException(
                    status_code=422, detail=f"Invalid parameter {k2}. Use {k1} instead."
                )
        payload = rename_payload_keys(payload, key_mapping)

        # Handle 'prompt' field in payload
        try:
            prompt = [self._parse_chat_messages_to_prompt(messages)]
        except MlflowException as e:
            raise AIGatewayException(
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
        with custom_token_allowance_exceeded_handling():
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
            created=int(time.time()),
            model=self.config.model.name,
            choices=[
                chat.Choice(
                    index=idx,
                    message=chat.ResponseMessage(role="assistant", content=c),
                    finish_reason=None,
                )
                for idx, c in enumerate(resp["outputs"])
            ],
            usage=chat.ChatUsage(
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
            ),
        )

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        key_mapping = {
            "max_tokens": "max_new_tokens",
        }
        for k1, k2 in key_mapping.items():
            if k2 in payload:
                raise AIGatewayException(
                    status_code=422, detail=f"Invalid parameter {k2}. Use {k1} instead."
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

        with custom_token_allowance_exceeded_handling():
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
            created=int(time.time()),
            object="text_completion",
            model=self.config.model.name,
            choices=[
                completions.Choice(
                    index=idx,
                    text=c,
                    finish_reason=None,
                )
                for idx, c in enumerate(resp["outputs"])
            ],
            usage=completions.CompletionsUsage(
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
            ),
        )

    async def embeddings(self, payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        key_mapping = {"input": "inputs"}
        for k1, k2 in key_mapping.items():
            if k2 in payload:
                raise AIGatewayException(
                    status_code=422, detail=f"Invalid parameter {k2}. Use {k1} instead."
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
            data=[
                embeddings.EmbeddingObject(
                    embedding=output,
                    index=idx,
                )
                for idx, output in enumerate(resp["outputs"])
            ],
            model=self.config.model.name,
            usage=embeddings.EmbeddingsUsage(
                prompt_tokens=None,
                total_tokens=None,
            ),
        )


@contextmanager
def custom_token_allowance_exceeded_handling():
    """
    Context manager handler for specific error messages that are incorrectly set as server-side
    errors, but are in actuality an issue with the request sent to the external provider.
    """
    from fastapi import HTTPException

    try:
        yield
    except HTTPException as e:
        status_code = e.status_code
        detail = e.detail or {}

        if (
            status_code == 500
            and detail
            and any(
                detail.get("message", "").startswith(x)
                for x in (
                    "Error: max output tokens is limited to",
                    "Error: prompt token count",
                )
            )
        ):
            raise HTTPException(status_code=422, detail=detail)
        else:
            raise
