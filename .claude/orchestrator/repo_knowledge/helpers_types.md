---
name: helpers_types
description: Auto-generated public-symbol reference for `mlflow/types/`. Use this before suggesting a new helper.
applies_to: any PR that touches mlflow/types/, defines a chat / agent / response BaseModel, calls validate_compat / model_dump_compat / model_validate, or threads ChatMessage / ChatTool / ResponsesAgent types.
last_verified: 2026-05-05
citation_policy: each `path:line` is the `def` / `class` line. If the snippet drifts, search by symbol name.
generated_by: .claude/orchestrator/scripts/generate_helpers_md.py (refreshed weekly by .github/workflows/refresh-helpers.yml).
---

# Helpers: `mlflow/types/`

Auto-generated. Walks `mlflow/types/` and lists every public symbol with its signature and first docstring sentence.

## How to use this file

- **Before suggesting a new utility function in a review**, grep this file for the area you're touching. If a helper already exists, point at its `path:line` instead of asking for a new one.
- **Class entries** list public methods in the same row group (`ClassName.method` form).
- **Search by symbol name**, not by line number: line numbers drift after reformats.

## `mlflow/types/agent.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `ChatAgentMessage` | class | `(BaseModel)` | A message in a ChatAgent model request or response. | 22 |
| `ChatAgentMessage.check_content_and_tool_calls` | method | `(self)` | Ensure at least one of 'content' or 'tool_calls' is set. | 51 |
| `ChatAgentMessage.check_tool_messages` | method | `(self)` | Ensure that the 'name' and 'tool_call_id' fields are set for tool messages. | 60 |
| `ChatContext` | class | `(BaseModel)` | Context to be used in a ChatAgent endpoint. | 69 |
| `ChatAgentRequest` | class | `(BaseModel)` | Format of a ChatAgent interface request. | 82 |
| `ChatAgentResponse` | class | `(BaseModel)` | Represents the response of a ChatAgent. | 103 |
| `ChatAgentResponse.check_message_ids` | method | `(self)` | Ensure that all messages have an ID and it is unique. | 125 |
| `ChatAgentChunk` | class | `(BaseModel)` | Represents a single chunk within the streaming response of a ChatAgent. | 143 |
| `ChatAgentChunk.check_message_id` | method | `(self)` | Ensure that the message ID is unique. | 169 |

## `mlflow/types/chat.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `TextContentPart` | class | `(BaseModel)` |  | 9 |
| `ImageUrl` | class | `(BaseModel)` | Represents an image URL. | 14 |
| `ImageContentPart` | class | `(BaseModel)` |  | 32 |
| `InputAudio` | class | `(BaseModel)` |  | 37 |
| `AudioContentPart` | class | `(BaseModel)` |  | 42 |
| `Function` | class | `(BaseModel)` |  | 55 |
| `Function.to_tool_call` | method | `(self, id) -> ToolCall` |  | 59 |
| `ToolCall` | class | `(BaseModel)` |  | 65 |
| `ChatMessage` | class | `(BaseModel)` | A chat request. | 71 |
| `ParamType` | class | `(BaseModel)` |  | 103 |
| `ParamProperty` | class | `(ParamType)` | OpenAI uses JSON Schema (https://json-schema.org/) for function parameters. | 107 |
| `FunctionParams` | class | `(BaseModel)` |  | 122 |
| `FunctionToolDefinition` | class | `(BaseModel)` |  | 129 |
| `ChatTool` | class | `(BaseModel)` | A tool definition passed to the chat completion API. | 136 |
| `ResponseFormat` | class | `(BaseModel)` | Response format configuration for structured outputs. | 147 |
| `ToolChoiceFunction` | class | `(BaseModel)` | Specifies a tool the model should use. | 160 |
| `ToolChoice` | class | `(BaseModel)` | Specifies a particular tool to use. | 166 |
| `BaseRequestPayload` | class | `(BaseModel)` | Common parameters used for chat completions and completion endpoints. | 177 |
| `ChatChoice` | class | `(BaseModel)` |  | 202 |
| `PromptTokensDetails` | class | `(BaseModel)` |  | 208 |
| `ChatUsage` | class | `(BaseModel)` |  | 214 |
| `ToolCallDelta` | class | `(BaseModel)` |  | 230 |
| `ChatChoiceDelta` | class | `(BaseModel)` |  | 237 |
| `ChatChunkChoice` | class | `(BaseModel)` |  | 243 |
| `ChatCompletionChunk` | class | `(BaseModel)` | A chunk of a chat completion stream response. | 249 |
| `ChatCompletionRequest` | class | `(BaseRequestPayload)` | A request to the chat completion API. | 260 |
| `ChatCompletionResponse` | class | `(BaseModel)` | A response from the chat completion API. | 273 |

## `mlflow/types/llm.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `FunctionToolCallArguments` | class | `(_BaseDataclass)` | The arguments of a function tool call made by the model. | 132 |
| `FunctionToolCallArguments.to_tool_call` | method | `(self, id)` |  | 148 |
| `ToolCall` | class | `(_BaseDataclass)` | A tool call made by the model. | 155 |
| `ChatMessage` | class | `(_BaseDataclass)` | A message in a chat request or response. | 176 |
| `ChatChoiceDelta` | class | `(_BaseDataclass)` | A streaming message delta in a chat response. | 219 |
| `ParamType` | class | `(_BaseDataclass)` |  | 258 |
| `ParamProperty` | class | `(ParamType)` | A single parameter within a function definition. | 266 |
| `ToolParamsSchema` | class | `(_BaseDataclass)` | A tool parameter definition. | 293 |
| `FunctionToolDefinition` | class | `(_BaseDataclass)` | Definition for function tools (currently the only supported type of tool). | 319 |
| `FunctionToolDefinition.to_tool_definition` | method | `(self)` | Convenience function for wrapping this in a ToolDefinition | 345 |
| `ToolDefinition` | class | `(_BaseDataclass)` | Definition for tools that can be called by the model. | 353 |
| `ChatParams` | class | `(_BaseDataclass)` | Common parameters used for chat inference  Args:     temperature (float): A param used to control randomness and creativity dur... | 371 |
| `ChatParams.keys` | classmethod | `(cls) -> set[str]` | Return the keys of the dataclass | 451 |
| `ChatCompletionRequest` | class | `(ChatParams)` | Format of the request object expected by the chat endpoint. | 459 |
| `TopTokenLogProb` | class | `(_BaseDataclass)` | Token and its log probability. | 507 |
| `TokenLogProb` | class | `(_BaseDataclass)` | Message content token with log probability information. | 534 |
| `ChatChoiceLogProbs` | class | `(_BaseDataclass)` | Log probability information for the choice. | 566 |
| `ChatChoice` | class | `(_BaseDataclass)` | A single chat response generated by the model. | 581 |
| `ChatChunkChoice` | class | `(_BaseDataclass)` | A single chat response chunk generated by the model. | 609 |
| `TokenUsageStats` | class | `(_BaseDataclass)` | Stats about the number of tokens used during inference. | 637 |
| `ChatCompletionResponse` | class | `(_BaseDataclass)` | The full response object returned by the chat endpoint. | 661 |
| `ChatCompletionChunk` | class | `(_BaseDataclass)` | The streaming chunk returned by the chat endpoint. | 698 |

## `mlflow/types/responses.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `ResponsesAgentRequest` | class | `(BaseRequestPayload)` | Request object for ResponsesAgent. | 35 |
| `ResponsesAgentResponse` | class | `(Response)` | Response object for ResponsesAgent. | 55 |
| `ResponsesAgentStreamEvent` | class | `(BaseModel)` | Stream event for ResponsesAgent. | 71 |
| `ResponsesAgentStreamEvent.check_type` | method | `(self) -> 'ResponsesAgentStreamEvent'` |  | 87 |
| `responses_agent_output_reducer` | function | `(chunks: list[ResponsesAgentStreamEvent \| dict[str, Any]])` | Output reducer for ResponsesAgent streaming. | 145 |
| `create_text_delta` | function | `(delta: str, item_id: str) -> dict[str, Any]` | Helper method to create a dictionary conforming to the text delta schema for streaming. | 164 |
| `create_annotation_added` | function | `(item_id: str, annotation: dict[str, Any], annotation_index: int \| None) -> dict[str, Any]` | Helper method to create annotation added event. | 177 |
| `create_text_output_item` | function | `(text: str, id: str, annotations: list[dict[str, Any]] \| None) -> dict[str, Any]` | Helper method to create a dictionary conforming to the text output item schema. | 189 |
| `create_reasoning_item` | function | `(id: str, reasoning_text: str) -> dict[str, Any]` | Helper method to create a dictionary conforming to the reasoning item schema. | 214 |
| `create_function_call_item` | function | `(id: str, call_id: str, name: str, arguments: str) -> dict[str, Any]` | Helper method to create a dictionary conforming to the function call item schema. | 231 |
| `create_function_call_output_item` | function | `(call_id: str, output: str) -> dict[str, Any]` | Helper method to create a dictionary conforming to the function call output item schema. | 251 |
| `create_mcp_approval_request_item` | function | `(id: str, arguments: str, name: str, server_label: str) -> dict[str, Any]` | Helper method to create a dictionary conforming to the MCP approval request item schema. | 268 |
| `create_mcp_approval_response_item` | function | `(id: str, approval_request_id: str, approve: bool, reason: str \| None) -> dict[str, Any]` | Helper method to create a dictionary conforming to the MCP approval response item schema. | 290 |
| `responses_to_cc` | function | `(message: dict[str, Any]) -> list[dict[str, Any]]` | Convert from a Responses API output item to a list of ChatCompletion messages. | 315 |
| `to_chat_completions_input` | function | `(responses_input: Sequence[dict[str, Any] \| Message \| OutputItem]) -> list[dict[str, Any]]` | Convert from Responses input items to ChatCompletion dictionaries. | 386 |
| `output_to_responses_items_stream` | function | `(chunks: Iterator[dict[str, Any]], aggregator: list[dict[str, Any]] \| None) -> Generator[ResponsesAgentStr...` | For streaming, convert from various message format dicts to Responses output items, returning a generator of ResponsesAgentStre... | 399 |

## `mlflow/types/responses_helpers.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `Status` | class | `(BaseModel)` |  | 17 |
| `Status.check_status` | method | `(self) -> 'Status'` |  | 21 |
| `ResponseError` | class | `(BaseModel)` |  | 34 |
| `AnnotationFileCitation` | class | `(BaseModel)` |  | 39 |
| `AnnotationURLCitation` | class | `(BaseModel)` |  | 45 |
| `AnnotationFilePath` | class | `(BaseModel)` |  | 53 |
| `Annotation` | class | `(BaseModel)` |  | 59 |
| `Annotation.check_type` | method | `(self) -> 'Annotation'` |  | 64 |
| `ResponseOutputText` | class | `(BaseModel)` |  | 76 |
| `ResponseOutputRefusal` | class | `(BaseModel)` |  | 82 |
| `Content` | class | `(BaseModel)` |  | 87 |
| `Content.check_type` | method | `(self) -> 'Content'` |  | 92 |
| `ResponseOutputMessage` | class | `(Status)` |  | 102 |
| `ResponseOutputMessage.check_role` | method | `(self) -> 'ResponseOutputMessage'` |  | 109 |
| `ResponseOutputMessage.check_content` | method | `(self) -> 'ResponseOutputMessage'` |  | 115 |
| `ResponseFunctionToolCall` | class | `(Status)` |  | 123 |
| `Summary` | class | `(BaseModel)` |  | 131 |
| `ResponseReasoningItem` | class | `(Status)` |  | 136 |
| `McpApprovalRequest` | class | `(Status)` |  | 142 |
| `McpApprovalResponse` | class | `(Status)` |  | 150 |
| `OutputItem` | class | `(BaseModel)` |  | 158 |
| `OutputItem.check_type` | method | `(self) -> 'OutputItem'` |  | 163 |
| `IncompleteDetails` | class | `(BaseModel)` |  | 185 |
| `IncompleteDetails.check_reason` | method | `(self) -> 'IncompleteDetails'` |  | 189 |
| `ToolChoiceFunction` | class | `(BaseModel)` |  | 195 |
| `FunctionTool` | class | `(BaseModel)` |  | 200 |
| `Tool` | class | `(BaseModel)` |  | 208 |
| `Tool.check_type` | method | `(self) -> 'Tool'` |  | 213 |
| `ToolChoice` | class | `(BaseModel)` |  | 221 |
| `ToolChoice.check_tool_choice` | method | `(self) -> 'ToolChoice'` |  | 225 |
| `ReasoningParams` | class | `(BaseModel)` |  | 235 |
| `ReasoningParams.check_generate_summary` | method | `(self) -> 'ReasoningParams'` |  | 240 |
| `ReasoningParams.check_effort` | method | `(self) -> 'ReasoningParams'` |  | 246 |
| `InputTokensDetails` | class | `(BaseModel)` |  | 252 |
| `OutputTokensDetails` | class | `(BaseModel)` |  | 256 |
| `ResponseUsage` | class | `(BaseModel)` |  | 260 |
| `Truncation` | class | `(BaseModel)` |  | 268 |
| `Truncation.check_truncation` | method | `(self) -> 'Truncation'` |  | 272 |
| `Response` | class | `(Truncation, ToolChoice)` |  | 278 |
| `Response.check_status` | method | `(self) -> 'Response'` |  | 317 |
| `ResponseInputTextParam` | class | `(BaseModel)` |  | 334 |
| `Message` | class | `(Status)` |  | 339 |
| `Message.check_content` | method | `(self) -> 'Message'` |  | 346 |
| `Message.check_role` | method | `(self) -> 'Message'` |  | 363 |
| `FunctionCallOutput` | class | `(Status)` |  | 371 |
| `BaseRequestPayload` | class | `(Truncation, ToolChoice)` |  | 377 |
| `ResponseTextDeltaEvent` | class | `(BaseModel)` |  | 397 |
| `ResponseTextAnnotationDeltaEvent` | class | `(BaseModel)` |  | 405 |
| `ResponseOutputItemDoneEvent` | class | `(BaseModel)` |  | 414 |
| `ResponseErrorEvent` | class | `(BaseModel)` |  | 420 |
| `ResponseCompletedEvent` | class | `(BaseModel)` |  | 427 |

## `mlflow/types/schema.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `DataType` | class | `(Enum)` | MLflow data types. | 39 |
| `DataType.to_numpy` | method | `(self) -> np.dtype` | Get equivalent numpy data type. | 83 |
| `DataType.to_pandas` | method | `(self) -> np.dtype` | Get equivalent pandas data type. | 87 |
| `DataType.to_spark` | method | `(self)` |  | 91 |
| `DataType.to_python` | method | `(self)` | Get equivalent python data type. | 101 |
| `DataType.check_type` | classmethod | `(cls, data_type, value)` |  | 106 |
| `DataType.all_types` | classmethod | `(cls)` |  | 119 |
| `DataType.get_spark_types` | classmethod | `(cls)` |  | 123 |
| `DataType.from_numpy_type` | classmethod | `(cls, np_type)` |  | 127 |
| `BaseType` | class | `(ABC)` |  | 131 |
| `BaseType.to_dict` | method | `(self) -> dict[str, Any]` | Dictionary representation of the object. | 145 |
| `Property` | class | `(BaseType)` | Specification used to represent a json-convertible object property. | 157 |
| `Property.required` | method | `(self, value: bool) -> None` |  | 208 |
| `Property.to_dict` | method | `(self)` |  | 227 |
| `Property.from_json_dict` | classmethod | `(cls, **kwargs)` | Deserialize from a json loaded dictionary. | 233 |
| `Object` | class | `(BaseType)` | Specification used to represent a json-convertible object. | 320 |
| `Object.properties` | method | `(self, value: list[Property]) -> None` |  | 362 |
| `Object.to_dict` | method | `(self)` |  | 375 |
| `Object.from_json_dict` | classmethod | `(cls, **kwargs)` | Deserialize from a json loaded dictionary. | 385 |
| `Array` | class | `(BaseType)` | Specification used to represent a json-convertible array. | 476 |
| `Array.to_dict` | method | `(self)` |  | 507 |
| `Array.from_json_dict` | classmethod | `(cls, **kwargs)` | Deserialize from a json loaded dictionary. | 514 |
| `SparkMLVector` | class | `(Array)` | Specification used to represent a vector type in Spark ML. | 569 |
| `SparkMLVector.to_dict` | method | `(self)` |  | 577 |
| `SparkMLVector.from_json_dict` | classmethod | `(cls, **kwargs)` |  | 581 |
| `Map` | class | `(BaseType)` | Specification used to represent a json-convertible map with string type keys. | 596 |
| `Map.to_dict` | method | `(self)` |  | 626 |
| `Map.from_json_dict` | classmethod | `(cls, **kwargs)` | Deserialize from a json loaded dictionary. | 635 |
| `AnyType` | class | `(BaseType)` |  | 683 |
| `AnyType.to_dict` | method | `(self)` |  | 710 |
| `ColSpec` | class | `` | Specification of name and type of a single column in a dataset. | 727 |
| `ColSpec.name` | method | `(self, value: str \| None) -> None` |  | 762 |
| `ColSpec.to_dict` | method | `(self) -> dict[str, Any]` |  | 770 |
| `ColSpec.from_json_dict` | classmethod | `(cls, **kwargs)` | Deserialize from a json loaded dictionary. | 790 |
| `TensorInfo` | class | `` | Representation of the shape and type of a Tensor. | 818 |
| `TensorInfo.to_dict` | method | `(self) -> dict[str, Any]` |  | 857 |
| `TensorInfo.from_json_dict` | classmethod | `(cls, **kwargs)` | Deserialize from a json loaded dictionary. | 861 |
| `TensorSpec` | class | `` | Specification used to represent a dataset stored as a Tensor. | 878 |
| `TensorSpec.to_dict` | method | `(self) -> dict[str, Any]` |  | 915 |
| `TensorSpec.from_json_dict` | classmethod | `(cls, **kwargs)` | Deserialize from a json loaded dictionary. | 922 |
| `Schema` | class | `` | Specification of a dataset. | 951 |
| `Schema.is_tensor_spec` | method | `(self) -> bool` | Return true iff this schema is specified using TensorSpec | 1014 |
| `Schema.input_names` | method | `(self) -> list[str \| int]` | Get list of data names or range of indices if the schema has no names. | 1018 |
| `Schema.required_input_names` | method | `(self) -> list[str \| int]` | Get list of required data names or range of indices if schema has no names. | 1022 |
| `Schema.optional_input_names` | method | `(self) -> list[str \| int]` | Get list of optional data names or range of indices if schema has no names. | 1026 |
| `Schema.has_input_names` | method | `(self) -> bool` | Return true iff this schema declares names, false otherwise. | 1030 |
| `Schema.input_types` | method | `(self) -> list[DataType \| np.dtype \| Array \| Object]` | Get types for each column in the schema. | 1034 |
| `Schema.input_types_dict` | method | `(self) -> dict[str, DataType \| np.dtype \| Array \| Object]` | Maps column names to types, iff this schema declares names. | 1038 |
| `Schema.input_dict` | method | `(self) -> dict[str, ColSpec \| TensorSpec]` | Maps column names to inputs, iff this schema declares names. | 1044 |
| `Schema.numpy_types` | method | `(self) -> list[np.dtype]` | Convenience shortcut to get the datatypes as numpy types. | 1050 |
| `Schema.pandas_types` | method | `(self) -> list[np.dtype]` | Convenience shortcut to get the datatypes as pandas types. | 1060 |
| `Schema.as_spark_schema` | method | `(self)` | Convert to Spark schema. | 1070 |
| `Schema.to_json` | method | `(self) -> str` | Serialize into json string. | 1089 |
| `Schema.to_dict` | method | `(self) -> list[dict[str, Any]]` | Serialize into a jsonable dictionary. | 1093 |
| `Schema.from_json` | classmethod | `(cls, json_str: str)` | Deserialize from a json string. | 1098 |
| `ParamSpec` | class | `` | Specification used to represent parameters for the model. | 1120 |
| `ParamSpec.validate_param_spec` | classmethod | `(cls, value: Any, param_spec: 'ParamSpec')` |  | 1156 |
| `ParamSpec.validate_type_and_shape` | classmethod | `(cls, spec: str, value: Any, value_type: DataType \| Object, shape: tuple[int, ...] \| None)` | Validate that the value has the expected type and shape. | 1162 |
| `ParamSpecTypedDict` | class | `(TypedDict)` |  | 1231 |
| `ParamSpec.to_dict` | method | `(self) -> ParamSpecTypedDict` |  | 1237 |
| `ParamSpec.from_json_dict` | classmethod | `(cls, **kwargs)` | Deserialize from a json loaded dictionary. | 1276 |
| `ParamSchema` | class | `` | Specification of parameters applicable to the model. | 1301 |
| `ParamSchema.to_json` | method | `(self) -> str` | Serialize into json string. | 1341 |
| `ParamSchema.from_json` | classmethod | `(cls, json_str: str)` | Deserialize from a json string. | 1346 |
| `ParamSchema.to_dict` | method | `(self) -> list[dict[str, Any]]` | Serialize into a jsonable dictionary. | 1350 |
| `convert_dataclass_to_schema` | function | `(dataclass)` | Converts a given dataclass into a Schema object. | 1401 |

## `mlflow/types/type_hints.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `type_hints_no_signature_inference` | function | `()` | This function returns a tuple of types that can be used as type hints, but no schema can be inferred from them. | 72 |
| `ColSpecType` | class | `(NamedTuple)` |  | 108 |
| `UnsupportedTypeHintException` | class | `(MlflowException)` |  | 113 |
| `InvalidTypeHintException` | class | `(MlflowException)` |  | 121 |
| `model_fields` | function | `(model: pydantic.BaseModel) -> dict[str, type[FIELD_TYPE]]` |  | 317 |
| `model_validate` | function | `(model: pydantic.BaseModel, values: Any) -> None` |  | 323 |
| `field_required` | function | `(field: type[FIELD_TYPE]) -> bool` |  | 329 |
| `ValidationResult` | class | `(NamedTuple)` |  | 475 |

## `mlflow/types/utils.py`

| Symbol | Kind | Signature | One-line docstring | Line |
|---|---|---|---|---|
| `TensorsNotSupportedException` | class | `(MlflowException)` |  | 37 |
| `clean_tensor_type` | function | `(dtype: np.dtype)` | This method strips away the size information stored in flexible datatypes such as np.str_ and np.bytes_. | 74 |
| `InvalidDataForSignatureInferenceError` | class | `(MlflowException)` |  | 118 |

