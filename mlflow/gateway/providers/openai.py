import json
import os
from typing import TYPE_CHECKING, AsyncIterable, Dict

from fastapi import HTTPException

from mlflow.environment_variables import MLFLOW_ENABLE_UC_FUNCTIONS
from mlflow.exceptions import MlflowException
from mlflow.gateway.config import OpenAIAPIType, OpenAIConfig, RouteConfig
from mlflow.gateway.providers.base import BaseProvider
from mlflow.gateway.providers.utils import send_request, send_stream_request
from mlflow.gateway.schemas import chat, completions, embeddings
from mlflow.gateway.uc_function_utils import (
    _UC_FUNCTION,
    TokenUsageAccumulator,
    execute_function,
    get_func_schema,
    join_uc_functions,
    parse_uc_functions,
    prepend_uc_functions,
)
from mlflow.gateway.utils import handle_incomplete_chunks, strip_sse_prefix
from mlflow.utils.uri import append_to_uri_path, append_to_uri_query_params

if TYPE_CHECKING:
    from databricks.sdk import FunctionInfo


# To mock the WorkspaceClient in tests
def _get_workspace_client():
    try:
        from databricks.sdk import WorkspaceClient

        return WorkspaceClient()
    except ImportError:
        raise HTTPException(
            message="Databricks SDK is required to use Unity Catalog integration",
            error_code=404,
        )


class OpenAIProvider(BaseProvider):
    NAME = "OpenAI"
    CONFIG_TYPE = OpenAIConfig

    def __init__(self, config: RouteConfig) -> None:
        super().__init__(config)
        if config.model.config is None or not isinstance(config.model.config, OpenAIConfig):
            # Should be unreachable
            raise MlflowException.invalid_parameter_value(
                "Invalid config type {config.model.config}"
            )
        self.openai_config: OpenAIConfig = config.model.config

    @property
    def _request_base_url(self):
        api_type = self.openai_config.openai_api_type
        if api_type == OpenAIAPIType.OPENAI:
            base_url = self.openai_config.openai_api_base or "https://api.openai.com/v1"
            if api_version := self.openai_config.openai_api_version is not None:
                return append_to_uri_query_params(base_url, ("api-version", api_version))
            else:
                return base_url
        elif api_type in (OpenAIAPIType.AZURE, OpenAIAPIType.AZUREAD):
            openai_url = append_to_uri_path(
                self.openai_config.openai_api_base,
                "openai",
                "deployments",
                self.openai_config.openai_deployment_name,
            )
            return append_to_uri_query_params(
                openai_url,
                ("api-version", self.openai_config.openai_api_version),
            )
        else:
            raise MlflowException.invalid_parameter_value(
                f"Invalid OpenAI API type '{self.openai_config.openai_api_type}'"
            )

    @property
    def _request_headers(self):
        api_type = self.openai_config.openai_api_type
        if api_type == OpenAIAPIType.OPENAI:
            headers = {
                "Authorization": f"Bearer {self.openai_config.openai_api_key}",
            }
            if org := self.openai_config.openai_organization:
                headers["OpenAI-Organization"] = org
            return headers
        elif api_type == OpenAIAPIType.AZUREAD:
            return {
                "Authorization": f"Bearer {self.openai_config.openai_api_key}",
            }
        elif api_type == OpenAIAPIType.AZURE:
            return {
                "api-key": self.openai_config.openai_api_key,
            }
        else:
            raise MlflowException.invalid_parameter_value(
                f"Invalid OpenAI API type '{self.openai_config.openai_api_type}'"
            )

    def _add_model_to_payload_if_necessary(self, payload):
        # NB: For Azure OpenAI, the deployment name (which is included in the URL) specifies
        # the model; it is not specified in the payload. For OpenAI outside of Azure, the
        # model is always specified in the payload
        if self.openai_config.openai_api_type not in (OpenAIAPIType.AZURE, OpenAIAPIType.AZUREAD):
            return {"model": self.config.model.name, **payload}
        else:
            return payload

    async def chat_stream(
        self, payload: chat.RequestPayload
    ) -> AsyncIterable[chat.StreamResponsePayload]:
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)

        stream = send_stream_request(
            headers=self._request_headers,
            base_url=self._request_base_url,
            path="chat/completions",
            payload=self._add_model_to_payload_if_necessary(payload),
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
                object=resp["object"],
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

    async def _chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)

        return await send_request(
            headers=self._request_headers,
            base_url=self._request_base_url,
            path="chat/completions",
            payload=self._add_model_to_payload_if_necessary(payload),
        )

    async def _chat_uc_function(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        workspace_client = _get_workspace_client()
        warehouse_id = os.environ.get("DATABRICKS_WAREHOUSE_ID")
        if warehouse_id is None:
            raise HTTPException(
                status_code=400,
                detail="DATABRICKS_WAREHOUSE_ID environment variable is not set",
            )

        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)

        token_usage_accumulator = TokenUsageAccumulator()
        user_tool_messages = [m for m in payload["messages"] if m["role"] == "tool"]
        user_tool_calls = next(
            (m["tool_calls"] for m in payload["messages"] if "tool_calls" in m), None
        )
        if (
            user_tool_messages
            and user_tool_calls
            and (result := parse_uc_functions(payload["messages"][0]["content"]))
        ):
            uc_func_calls, uc_func_messages = result
            messages = [
                *[m for m in payload["messages"] if m["role"] == "tool" or "tool_calls" in m],
                # Join UC function calls and user tool calls
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": uc_func_calls + user_tool_calls,
                },
                *uc_func_messages,
                *user_tool_messages,
            ]
            resp = await send_request(
                headers=self._request_headers,
                base_url=self._request_base_url,
                path="chat/completions",
                payload=self._add_model_to_payload_if_necessary(
                    {
                        **payload,
                        "messages": messages,
                    }
                ),
            )
            token_usage_accumulator.update(resp.get("usage", {}))
        elif any(t["type"] == _UC_FUNCTION for t in payload.get("tools", [])):
            updated_tools = []
            uc_func_mapping: Dict[str, "FunctionInfo"] = {}
            for tool in payload.get("tools", []):
                if tool["type"] == _UC_FUNCTION:
                    function_name = tool[_UC_FUNCTION]["name"]
                    function = workspace_client.functions.get(function_name)
                    param_metadata = get_func_schema(function)
                    t = {
                        "type": "function",
                        "function": param_metadata,
                    }
                    uc_func_mapping[t["function"]["name"]] = function
                    updated_tools.append(t)
                else:
                    updated_tools.append(tool)

            payload["tools"] = updated_tools

            messages = payload.pop("messages", [])
            uc_func_calls = []
            user_tool_calls = []
            resp = None
            for _ in range(20):  # loop until we get a response without tool_calls
                resp = await send_request(
                    headers=self._request_headers,
                    base_url=self._request_base_url,
                    path="chat/completions",
                    payload=self._add_model_to_payload_if_necessary(
                        {
                            **payload,
                            "messages": messages,
                        }
                    ),
                )
                token_usage_accumulator.update(resp.get("usage", {}))
                # TODO to support n > 1.
                assistant_msg = resp["choices"][0]["message"]
                tool_calls = assistant_msg.get("tool_calls")
                if tool_calls is None:
                    if uc_func_calls:
                        original_content = resp["choices"][0]["message"]["content"]
                        resp["choices"][0]["message"]["content"] = prepend_uc_functions(
                            original_content, uc_func_calls
                        )

                    if user_tool_calls:
                        # Is this line unreachable?
                        resp["choices"][0]["message"]["tool_calls"] = user_tool_calls

                    break

                tool_messages = []
                for tool_call in tool_calls:
                    func = tool_call["function"]
                    parameters = json.loads(func["arguments"])
                    if func_info := uc_func_mapping.get(func["name"]):
                        result = execute_function(
                            ws=workspace_client,
                            warehouse_id=warehouse_id,
                            function=function,
                            parameters=parameters,
                        )
                        tool_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call["id"],
                                "content": result.to_json(),
                            }
                        )

                        uc_func_calls.append(
                            (
                                {
                                    "id": tool_call["id"],
                                    "name": func_info.full_name,
                                    "arguments": func["arguments"],
                                },
                                {
                                    "tool_call_id": tool_call["id"],
                                    "content": result.to_json(),
                                },
                            )
                        )
                    else:
                        user_tool_calls.append(
                            {
                                "id": tool_call["id"],
                                "type": "function",
                                "function": {
                                    "name": func["name"],
                                    "arguments": func["arguments"],
                                },
                            }
                        )

                if message_content := assistant_msg.pop("content", None):
                    messages.append({"role": "assistant", "content": message_content})
                messages += [assistant_msg, *tool_messages]

                if user_tool_calls:
                    # We can't go on without a response from the user, so we break here
                    if uc_func_calls:
                        resp["choices"][0]["message"]["content"] = join_uc_functions(uc_func_calls)

                    resp["choices"][0]["message"]["tool_calls"] = user_tool_calls
                    break
            else:
                raise HTTPException(
                    status_code=500,
                    detail="Max iterations reached",
                )
        else:
            # No UC functions to execute
            resp = await send_request(
                headers=self._request_headers,
                base_url=self._request_base_url,
                path="chat/completions",
                payload=self._add_model_to_payload_if_necessary(payload),
            )
            token_usage_accumulator.update(resp.get("usage", {}))

        # Update the token usage
        resp["usage"].update(token_usage_accumulator.dict())

        return resp

    async def chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        if MLFLOW_ENABLE_UC_FUNCTIONS.get():
            resp = await self._chat_uc_function(payload)
        else:
            resp = await self._chat(payload)

        # Response example (https://platform.openai.com/docs/api-reference/chat/create)
        # ```
        # {
        #    "id":"chatcmpl-abc123",
        #    "object":"chat.completion",
        #    "created":1677858242,
        #    "model":"gpt-4o-mini",
        #    "usage":{
        #       "prompt_tokens":13,
        #       "completion_tokens":7,
        #       "total_tokens":20
        #    },
        #    "choices":[
        #       {
        #          "message":{
        #             "role":"assistant",
        #             "content":"\n\nThis is a test!"
        #          },
        #          "finish_reason":"stop",
        #          "index":0
        #       }
        #    ]
        # }
        # ```
        return chat.ResponsePayload(
            id=resp["id"],
            object=resp["object"],
            created=resp["created"],
            model=resp["model"],
            choices=[
                chat.Choice(
                    index=idx,
                    message=chat.ResponseMessage(
                        role=c["message"]["role"],
                        content=c["message"].get("content"),
                        tool_calls=(
                            (calls := c["message"].get("tool_calls"))
                            and [chat.ToolCall(**c) for c in calls]
                        ),
                    ),
                    finish_reason=c.get("finish_reason"),
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
            base_url=self._request_base_url,
            path="chat/completions",
            payload=self._add_model_to_payload_if_necessary(payload),
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
            base_url=self._request_base_url,
            path="chat/completions",
            payload=self._add_model_to_payload_if_necessary(payload),
        )
        # Response example (https://platform.openai.com/docs/api-reference/completions/create)
        # ```
        # {
        #   "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
        #   "object": "text_completion",
        #   "created": 1589478378,
        #   "model": "text-davinci-003",
        #   "choices": [
        #     {
        #       "text": "\n\nThis is indeed a test",
        #       "index": 0,
        #       "logprobs": null,
        #       "finish_reason": "length"
        #     }
        #   ],
        #   "usage": {
        #     "prompt_tokens": 5,
        #     "completion_tokens": 7,
        #     "total_tokens": 12
        #   }
        # }
        # ```
        return self._prepare_completion_response_payload(resp)

    async def embeddings(self, payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        from fastapi.encoders import jsonable_encoder

        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        resp = await send_request(
            headers=self._request_headers,
            base_url=self._request_base_url,
            path="embeddings",
            payload=self._add_model_to_payload_if_necessary(payload),
        )
        # Response example (https://platform.openai.com/docs/api-reference/embeddings/create):
        # ```
        # {
        #   "object": "list",
        #   "data": [
        #     {
        #       "object": "embedding",
        #       "embedding": [
        #         0.0023064255,
        #         -0.009327292,
        #         .... (1536 floats total for ada-002)
        #         -0.0028842222,
        #       ],
        #       "index": 0
        #     }
        #   ],
        #   "model": "text-embedding-ada-002",
        #   "usage": {
        #     "prompt_tokens": 8,
        #     "total_tokens": 8
        #   }
        # }
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
