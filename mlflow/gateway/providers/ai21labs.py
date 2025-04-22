import time

from mlflow.gateway.config import AI21LabsConfig, RouteConfig
from mlflow.gateway.exceptions import AIGatewayException
from mlflow.gateway.providers.base import BaseProvider
from mlflow.gateway.providers.utils import rename_payload_keys, send_request
from mlflow.gateway.schemas import completions


class AI21LabsProvider(BaseProvider):
    NAME = "AI21Labs"
    CONFIG_TYPE = AI21LabsConfig

    def __init__(self, config: RouteConfig) -> None:
        super().__init__(config)
        if config.model.config is None or not isinstance(config.model.config, AI21LabsConfig):
            raise TypeError(f"Unexpected config type {config.model.config}")
        self.ai21labs_config: AI21LabsConfig = config.model.config
        self.headers = {"Authorization": f"Bearer {self.ai21labs_config.ai21labs_api_key}"}
        self.base_url = f"https://api.ai21.com/studio/v1/{self.config.model.name}/"

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        key_mapping = {
            "stop": "stopSequences",
            "n": "numResults",
            "max_tokens": "maxTokens",
        }
        for k1, k2 in key_mapping.items():
            if k2 in payload:
                raise AIGatewayException(
                    status_code=422, detail=f"Invalid parameter {k2}. Use {k1} instead."
                )
        if payload.get("stream", False):
            raise AIGatewayException(
                status_code=422,
                detail="Setting the 'stream' parameter to 'true' is not supported with the MLflow "
                "Gateway.",
            )
        payload = rename_payload_keys(payload, key_mapping)
        resp = await send_request(
            headers=self.headers,
            base_url=self.base_url,
            path="complete",
            payload=payload,
        )
        # Response example (https://docs.ai21.com/reference/j2-complete-ref)
        # ```
        # {
        #   "id": "7921a78e-d905-c9df-27e3-88e4831e3c3b",
        #   "prompt": {
        #     "text": "I will"
        #   },
        #   "completions": [
        #     {
        #       "data": {
        #         "text": " complete this"
        #       },
        #       "finishReason": {
        #         "reason": "length",
        #         "length": 2
        #       }
        #     }
        #   ]
        # }
        # ```
        return completions.ResponsePayload(
            created=int(time.time()),
            object="text_completion",
            model=self.config.model.name,
            choices=[
                completions.Choice(
                    index=idx,
                    text=c["data"]["text"],
                    finish_reason=c["finishReason"]["reason"],
                )
                for idx, c in enumerate(resp["completions"])
            ],
            usage=completions.CompletionsUsage(
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
            ),
        )
