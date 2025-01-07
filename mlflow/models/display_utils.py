import jinja2

from mlflow.models.model import ModelInfo
from mlflow.models.signature import ModelSignature
from mlflow.types import schema
from mlflow.utils import databricks_utils


def _is_input_string(inputs: schema.Schema) -> bool:
    return (
        not inputs.has_input_names()
        and len(inputs.input_types()) == 1
        and inputs.input_types()[0] == schema.DataType.string
    )


def _is_input_agent_compatible(inputs: schema.Schema) -> bool:
    if _is_input_string(inputs):
        return True
    if not inputs.has_input_names():
        return False
    messages = inputs.input_dict().get("messages")
    if not messages:
        return False
    if not isinstance(messages.type, schema.Array):
        return False
    items = messages.type.dtype
    if not isinstance(items, schema.Object):
        return False
    properties = items.properties
    content = next(filter(lambda prop: prop.name == "content", properties), None)
    role = next(filter(lambda prop: prop.name == "role", properties), None)
    return (
        content
        and content.dtype == schema.DataType.string
        and role
        and role.dtype == schema.DataType.string
    )


def _is_output_string_response(outputs: schema.Schema) -> bool:
    if not outputs.has_input_names():
        return False
    content = outputs.input_dict().get("content")
    if not content:
        return False
    return content.type == schema.DataType.string


def _is_output_string(outputs: schema.Schema) -> bool:
    return (
        not outputs.has_input_names()
        and len(outputs.input_types()) == 1
        and outputs.input_types()[0] == schema.DataType.string
    )


def _is_output_chat_completion_response(outputs: schema.Schema) -> bool:
    if not outputs.has_input_names():
        return False
    choices = outputs.input_dict().get("choices")
    if not choices:
        return False
    if not isinstance(choices.type, schema.Array):
        return False
    items = choices.type.dtype
    if not isinstance(items, schema.Object):
        return False
    properties = items.properties
    message = next(filter(lambda prop: prop.name == "message", properties), None)
    if not message:
        return False
    if not isinstance(message.dtype, schema.Object):
        return False
    message_properties = message.dtype.properties
    content = next(filter(lambda prop: prop.name == "content", message_properties), None)
    role = next(filter(lambda prop: prop.name == "role", message_properties), None)
    return (
        content
        and content.dtype == schema.DataType.string
        and role
        and role.dtype == schema.DataType.string
    )


def _is_output_agent_compatible(outputs: schema.Schema) -> bool:
    return (
        _is_output_string_response(outputs)
        or _is_output_string(outputs)
        or _is_output_chat_completion_response(outputs)
    )


def _is_signature_agent_compatible(signature: ModelSignature) -> bool:
    """Determines whether the given signature is compatible with the agent eval schema.

    See https://docs.databricks.com/en/generative-ai/agent-evaluation/evaluation-schema.html.
    The schema accepts the OpenAI spec, as well as simpler formats such as vanilla string response
    and `StringResponse`.
    """
    return _is_input_agent_compatible(signature.inputs) and _is_output_agent_compatible(
        signature.outputs
    )


def should_render_agent_eval_template(signature: ModelSignature) -> bool:
    if not databricks_utils.is_in_databricks_runtime():
        return False
    from IPython import get_ipython

    if get_ipython() is None:
        return False
    return _is_signature_agent_compatible(signature)


def maybe_render_agent_eval_recipe(model_info: ModelInfo) -> None:
    # For safety, we wrap in try/catch to make sure we don't break `mlflow.*.log_model`.
    try:
        if not should_render_agent_eval_template(model_info.signature):
            return
        # Create a Jinja2 environment and load the template
        env = jinja2.Environment(
            loader=jinja2.PackageLoader("mlflow.models", "resources"),
            autoescape=jinja2.select_autoescape(["html"]),
        )
        pip_install_command = """%pip install -U databricks-agents
dbutils.library.restartPython()
## Run the above in a separate cell ##"""
        eval_with_synthetic_code = env.get_template("eval_with_synthetic_example.py").render(
            {"pipInstall": pip_install_command, "modelUri": model_info.model_uri}
        )
        eval_with_dataset_code = env.get_template("eval_with_dataset_example.py").render(
            {"pipInstall": pip_install_command, "modelUri": model_info.model_uri}
        )

        # Remove the ruff noqa comments.
        ruff_line = "# ruff: noqa: F821, I001\n"
        eval_with_synthetic_code = eval_with_synthetic_code.replace(ruff_line, "")
        eval_with_dataset_code = eval_with_dataset_code.replace(ruff_line, "")

        rendered_html = env.get_template("agent_evaluation_template.html").render(
            {
                "eval_with_synthetic_code": eval_with_synthetic_code,
                "eval_with_dataset_code": eval_with_dataset_code,
            }
        )
        from IPython.display import HTML, display

        display(HTML(rendered_html))
    except Exception:
        pass
