import json
import logging
from typing import TYPE_CHECKING

from mlflow.types.chat import (
    ChatTool,
    Function,
    FunctionParams,
    FunctionToolDefinition,
    ParamProperty,
    ToolCall,
)

if TYPE_CHECKING:
    from google import genai

_logger = logging.getLogger(__name__)


def convert_gemini_func_to_mlflow_chat_tool(
    function_def: "genai.types.FunctionDeclaration",
) -> ChatTool:
    """
    Convert Gemini function definition into MLflow's standard format (OpenAI compatible).
    Ref: https://ai.google.dev/gemini-api/docs/function-calling

    Args:
        function_def: A genai.types.FunctionDeclaration or genai.protos.FunctionDeclaration object
                      representing a function definition. google-genai >= 2.18 may expose
                      parameters via ``parameters_json_schema`` (a plain dict) instead of the
                      ``parameters`` Schema object; both representations are handled.

    Returns:
        ChatTool: MLflow's standard tool definition object.
    """
    if function_def.parameters is not None:
        params = _convert_gemini_function_param_to_mlflow_function_param(function_def.parameters)
    elif (json_schema := getattr(function_def, "parameters_json_schema", None)) is not None:
        # google-genai >= 2.18 uses parameters_json_schema (plain dict) instead of parameters
        params = _convert_json_schema_to_mlflow_function_param(json_schema)
    else:
        params = None
    return ChatTool(
        type="function",
        function=FunctionToolDefinition(
            name=function_def.name,
            description=function_def.description,
            parameters=params,
        ),
    )


def convert_gemini_func_call_to_mlflow_tool_call(
    func_call: "genai.types.FunctionCall",
) -> ToolCall:
    """
    Convert Gemini function call into MLflow's standard format (OpenAI compatible).
    Ref: https://ai.google.dev/gemini-api/docs/function-calling

    Args:
        func_call: A genai.types.FunctionCall or genai.protos.FunctionCall object
                   representing a single func call.

    Returns:
        ToolCall: MLflow's standard tool call object.
    """
    # original args object is not json serializable
    args = func_call.args or {}

    return ToolCall(
        # Gemini does not have func call id
        id=func_call.name,
        type="function",
        function=Function(name=func_call.name, arguments=json.dumps(dict(args))),
    )


def _convert_gemini_param_property_to_mlflow_param_property(param_property) -> ParamProperty:
    """
    Convert Gemini parameter property definition into MLflow's standard format (OpenAI compatible).
    Ref: https://ai.google.dev/gemini-api/docs/function-calling

    Args:
        param_property: A genai.types.Schema or genai.protos.Schema object
                        representing a parameter property.

    Returns:
        ParamProperty: MLflow's standard param property object.
    """
    type_name = param_property.type
    type_name = type_name.name.lower() if hasattr(type_name, "name") else type_name.lower()
    return ParamProperty(
        description=param_property.description,
        enum=param_property.enum,
        type=type_name,
    )


def _convert_gemini_function_param_to_mlflow_function_param(
    function_params: "genai.types.Schema",
) -> FunctionParams:
    """
    Convert Gemini function parameter definition into MLflow's standard format (OpenAI compatible).
    Ref: https://ai.google.dev/gemini-api/docs/function-calling

    Args:
        function_params: A genai.types.Schema or genai.protos.Schema object
                         representing function parameters.

    Returns:
        FunctionParams: MLflow's standard function parameter object.
    """
    return FunctionParams(
        properties={
            k: _convert_gemini_param_property_to_mlflow_param_property(v)
            for k, v in function_params.properties.items()
        },
        required=function_params.required,
    )


def _json_schema_prop_to_param_property(prop: dict[str, object]) -> ParamProperty:
    # ParamProperty has no `properties` field, so object-typed array items lose their
    # sub-schema; this is acceptable for the current use case (tool definitions display).
    kwargs: dict[str, object] = {
        "type": prop.get("type"),
        "description": prop.get("description"),
        "enum": prop.get("enum"),
    }
    if "items" in prop:
        kwargs["items"] = _json_schema_prop_to_param_property(prop["items"])
    return ParamProperty(**kwargs)


def _convert_json_schema_to_mlflow_function_param(json_schema: dict[str, object]) -> FunctionParams:
    """
    Convert a JSON Schema dict (FunctionDeclaration.parameters_json_schema in google-genai >= 2.18)
    into MLflow's FunctionParams format.
    """
    return FunctionParams(
        properties={
            name: _json_schema_prop_to_param_property(prop)
            for name, prop in json_schema.get("properties", {}).items()
        },
        required=json_schema.get("required"),
    )
