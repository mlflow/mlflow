# TODO: Move this in mlflow/gateway/utils/uc_functions.py

import json
import re
from dataclasses import dataclass
from io import StringIO
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.catalog import FunctionInfo, FunctionParameterInfo
    from databricks.sdk.service.sql import StatementParameterListItem


_UC_FUNCTION = "uc_function"


def uc_type_to_json_schema_type(uc_type_json: str | dict[str, Any]) -> dict[str, Any]:
    """
    Converts the JSON representation of a Unity Catalog data type to the corresponding JSON schema
    type. The conversion is lossy because we do not need to convert it back.
    """
    # See https://docs.databricks.com/en/sql/language-manual/sql-ref-datatypes.html
    # The actual type name in type_json is different from the corresponding SQL type name.
    spark_struct_field_mapping = {
        "long": {"type": "integer"},
        "binary": {"type": "string"},
        "boolean": {"type": "boolean"},
        "date": {"type": "string", "format": "date"},
        "double": {"type": "number"},
        "float": {"type": "number"},
        "integer": {"type": "integer"},
        "void": {"type": "null"},
        "short": {"type": "integer"},
        "string": {"type": "string"},
        "timestamp": {"type": "string", "format": "date-time"},
        "timestamp_ntz": {"type": "string", "format": "date-time"},
        "byte": {"type": "integer"},
    }
    if isinstance(uc_type_json, str):
        if t := spark_struct_field_mapping.get(uc_type_json):
            return t
        else:
            if uc_type_json.startswith("decimal"):
                return {"type": "number"}
            elif uc_type_json.startswith("interval"):
                raise TypeError(f"Type {uc_type_json} is not supported.")
            else:
                raise TypeError(f"Unknown type {uc_type_json}. Try upgrading this package.")
    else:
        assert isinstance(uc_type_json, dict)
        type = uc_type_json["type"]
        if type == "array":
            element_type = uc_type_to_json_schema_type(uc_type_json["elementType"])
            return {"type": "array", "items": element_type}
        elif type == "map":
            key_type = uc_type_json["keyType"]
            if key_type != "string":
                raise TypeError(f"Only support STRING key type for MAP but got {key_type}.")
            value_type = uc_type_to_json_schema_type(uc_type_json["valueType"])
            return {
                "type": "object",
                "additionalProperties": value_type,
            }
        elif type == "struct":
            properties = {}
            for field in uc_type_json["fields"]:
                properties[field["name"]] = uc_type_to_json_schema_type(field["type"])
            return {"type": "object", "properties": properties}
        else:
            raise TypeError(f"Unknown type {uc_type_json}. Try upgrading this package.")


def extract_param_metadata(p: "FunctionParameterInfo") -> dict[str, Any]:
    type_json = json.loads(p.type_json)["type"]
    json_schema_type = uc_type_to_json_schema_type(type_json)
    json_schema_type["name"] = p.name
    json_schema_type["description"] = (
        (p.comment or "") + f" (default: {p.parameter_default})" if p.parameter_default else ""
    )
    return json_schema_type


def get_func_schema(func: "FunctionInfo") -> dict[str, Any]:
    parameters = func.input_params.parameters if func.input_params else []
    return {
        "description": func.comment,
        "name": _get_tool_name(func),
        "parameters": {
            "type": "object",
            "properties": {p.name: extract_param_metadata(p) for p in parameters},
            "required": [p.name for p in parameters if p.parameter_default is None],
        },
    }


@dataclass
class ParameterizedStatement:
    statement: str
    parameters: list["StatementParameterListItem"]


@dataclass
class FunctionExecutionResult:
    """
    Result of executing a function.
    We always use a string to present the result value for AI model to consume.
    """

    error: str | None = None
    format: Literal["SCALAR", "CSV"] | None = None
    value: str | None = None
    truncated: bool | None = None

    def to_json(self) -> str:
        data = {k: v for (k, v) in self.__dict__.items() if v is not None}
        return json.dumps(data)


def is_scalar(function: "FunctionInfo") -> bool:
    """
    Returns True if the function returns a single row instead of a table.
    """
    from databricks.sdk.service.catalog import ColumnTypeName

    return function.data_type != ColumnTypeName.TABLE_TYPE


def _quote_identifier(identifier: str) -> str:
    """
    Quotes a SQL identifier to prevent SQL injection.
    Databricks SQL uses backticks for quoting identifiers.

    For multi-part identifiers (e.g., catalog.schema.function), each part is quoted separately.
    Existing backticks around parts are stripped before re-quoting.

    Raises:
        ValueError: If any identifier part contains embedded backticks.
    """
    parts = identifier.split(".")
    stripped_parts = [part.strip("`") for part in parts]
    for part in stripped_parts:
        if "`" in part:
            raise ValueError(
                f"Invalid identifier: {identifier}. "
                "Backticks are not allowed within Unity Catalog identifier names."
            )
    quoted_parts = [f"`{part}`" for part in stripped_parts]
    return ".".join(quoted_parts)


def get_execute_function_sql_stmt(
    function: "FunctionInfo",
    json_params: dict[str, Any],
) -> ParameterizedStatement:
    from databricks.sdk.service.catalog import ColumnTypeName
    from databricks.sdk.service.sql import StatementParameterListItem

    parts = []
    output_params = []
    quoted_function_name = _quote_identifier(function.full_name)
    if is_scalar(function):
        parts.append(f"SELECT {quoted_function_name}(")
    else:
        parts.append(f"SELECT * FROM {quoted_function_name}(")
    if function.input_params is None or function.input_params.parameters is None:
        assert not json_params, "Function has no parameters but parameters were provided."
    else:
        args = []
        use_named_args = False
        for p in function.input_params.parameters:
            if p.name not in json_params:
                if p.parameter_default is not None:
                    use_named_args = True
                else:
                    raise ValueError(f"Parameter {p.name} is required but not provided.")
            else:
                arg_clause = ""
                if use_named_args:
                    quoted_param_name = _quote_identifier(p.name)
                    arg_clause += f"{quoted_param_name} => "
                json_value = json_params[p.name]
                if p.type_name in (
                    ColumnTypeName.ARRAY,
                    ColumnTypeName.MAP,
                    ColumnTypeName.STRUCT,
                ):
                    # Use from_json to restore values of complex types.
                    json_value_str = json.dumps(json_value)
                    # TODO: parametrize type
                    arg_clause += f"from_json(:{p.name}, '{p.type_text}')"
                    output_params.append(
                        StatementParameterListItem(name=p.name, value=json_value_str)
                    )
                elif p.type_name == ColumnTypeName.BINARY:
                    # Use ubbase64 to restore binary values.
                    arg_clause += f"unbase64(:{p.name})"
                    output_params.append(StatementParameterListItem(name=p.name, value=json_value))
                else:
                    arg_clause += f":{p.name}"
                    output_params.append(
                        StatementParameterListItem(name=p.name, value=json_value, type=p.type_text)
                    )
                args.append(arg_clause)
        parts.append(",".join(args))
    parts.append(")")
    # TODO: check extra params in kwargs
    statement = "".join(parts)
    return ParameterizedStatement(statement=statement, parameters=output_params)


def execute_function(
    ws: "WorkspaceClient",
    warehouse_id: str,
    function: "FunctionInfo",
    parameters: dict[str, Any],
) -> FunctionExecutionResult:
    """
    Execute a function with the given arguments and return the result.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "Could not import pandas python package. Please install it with `pip install pandas`."
        ) from e
    from databricks.sdk.service.sql import StatementState

    # TODO: async so we can run functions in parallel
    parameterized_statement = get_execute_function_sql_stmt(function, parameters)
    # TODO: make limits and wait timeout configurable
    response = ws.statement_execution.execute_statement(
        statement=parameterized_statement.statement,
        warehouse_id=warehouse_id,
        parameters=parameterized_statement.parameters,
        wait_timeout="30s",
        row_limit=100,
        byte_limit=4096,
    )
    status = response.status
    assert status is not None, f"Statement execution failed: {response}"
    if status.state != StatementState.SUCCEEDED:
        error = status.error
        assert error is not None, "Statement execution failed but no error message was provided."
        return FunctionExecutionResult(error=f"{error.error_code}: {error.message}")
    manifest = response.manifest
    assert manifest is not None
    truncated = manifest.truncated
    result = response.result
    assert result is not None, "Statement execution succeeded but no result was provided."
    data_array = result.data_array
    if is_scalar(function):
        value = None
        if data_array and len(data_array) > 0 and len(data_array[0]) > 0:
            value = str(data_array[0][0])  # type: ignore
        return FunctionExecutionResult(format="SCALAR", value=value, truncated=truncated)
    else:
        schema = manifest.schema
        assert schema is not None and schema.columns is not None, (
            "Statement execution succeeded but no schema was provided."
        )
        columns = [c.name for c in schema.columns]
        if data_array is None:
            data_array = []
        pdf = pd.DataFrame.from_records(data_array, columns=columns)
        csv_buffer = StringIO()
        pdf.to_csv(csv_buffer, index=False)
        return FunctionExecutionResult(
            format="CSV", value=csv_buffer.getvalue(), truncated=truncated
        )


def join_uc_functions(uc_functions: list[dict[str, Any]]):
    calls = [
        f"""
<uc_function_call>
{json.dumps(request, indent=2)}
</uc_function_call>

<uc_function_result>
{json.dumps(result, indent=2)}
</uc_function_result>
""".strip()
        for (request, result) in uc_functions
    ]
    return "\n\n".join(calls)


def _get_tool_name(function: "FunctionInfo") -> str:
    # The maximum function name length OpenAI supports is 64 characters.
    return f"{function.catalog_name}__{function.schema_name}__{function.name}"[-64:]


@dataclass
class ParseResult:
    tool_calls: list[dict[str, Any]]
    tool_messages: list[dict[str, Any]]


_UC_REGEX = re.compile(
    r"""
<uc_function_call>
(?P<uc_function_call>.*?)
</uc_function_call>

<uc_function_result>
(?P<uc_function_result>.*?)
</uc_function_result>
""",
    re.DOTALL,
)


def parse_uc_functions(content) -> ParseResult | None:
    tool_calls = []
    tool_messages = []
    for m in _UC_REGEX.finditer(content):
        c = m.group("uc_function_call")
        g = m.group("uc_function_result")
        tool_calls.append(json.loads(c))
        tool_messages.append(json.loads(g))

    return ParseResult(tool_calls, tool_messages) if tool_calls else None


@dataclass
class TokenUsageAccumulator:
    prompt_tokens: int = 0
    completions_tokens: int = 0
    total_tokens: int = 0

    def update(self, usage_dict):
        self.prompt_tokens += usage_dict.get("prompt_tokens", 0)
        self.completions_tokens += usage_dict.get("completion_tokens", 0)
        self.total_tokens += usage_dict.get("total_tokens", 0)

    def dict(self):
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completions_tokens,
            "total_tokens": self.total_tokens,
        }


def prepend_uc_functions(content, uc_functions):
    return join_uc_functions(uc_functions) + "\n\n" + content
