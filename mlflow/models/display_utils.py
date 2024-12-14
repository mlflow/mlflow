import json

import jinja2

from mlflow.models.model import ModelInfo
from mlflow.models.signature import ModelSignature
from mlflow.utils import databricks_utils


def _is_input_agent_compatible(inputs: list[dict]) -> bool:
    messages = next(filter(lambda col: col.get("name") == "messages", inputs), None)
    if not messages:
        return False
    if not messages.get("type") == "array":
        return False
    items = messages.get("items")
    if not items:
        return False
    if items.get("type") != "object":
        return False
    properties = items.get("properties")
    if not properties:
        return False
    if properties.get("content", {}).get("type") != "string":
        return False
    if properties.get("role", {}).get("type") != "string":
        return False
    return True


def _is_output_string_response(outputs: list[dict]) -> bool:
    content = next(filter(lambda col: col.get("name") == "content", outputs), None)
    if not content:
        return False
    if content.get("type") != "string":
        return False
    return True


def _is_output_chat_completion_response(outputs: list[dict]) -> bool:
    choices = next(filter(lambda col: col.get("name") == "choices", outputs), None)
    if not choices:
        return False
    if not choices.get("type") == "array":
        return False
    items = choices.get("items")
    if not items:
        return False
    if items.get("type") != "object":
        return False
    properties = items.get("properties")
    if not properties:
        return False
    message = properties.get("message")
    if not message:
        return False
    if message.get("type") != "object":
        return False
    message_properties = message.get("properties")
    if not message_properties:
        return False
    if message_properties.get("content", {}).get("type") != "string":
        return False
    if message_properties.get("role", {}).get("type") != "string":
        return False
    return True


def _is_output_agent_compatible(outputs: list[dict]) -> bool:
    return _is_output_string_response(outputs) or _is_output_chat_completion_response(outputs)


def _is_signature_agent_compatible(signature: ModelSignature) -> bool:
    signature = signature.to_dict()
    inputs = json.loads(signature.get("inputs") or "[]")
    outputs = json.loads(signature.get("outputs") or "[]")
    return _is_input_agent_compatible(inputs) and _is_output_agent_compatible(outputs)


def should_render_agent_eval_template(signature: ModelSignature) -> bool:
    if not databricks_utils.is_in_databricks_runtime():
        return False
    try:
        from IPython import get_ipython

        if get_ipython() is None:
            return False
        # For safety, we do this in try/catch to make sure we don't break
        # `mlflow.*.log_model`.
        return _is_signature_agent_compatible(signature)
    except Exception:
        return False


def maybe_render_agent_eval_recipe(model_info: ModelInfo) -> None:
    if not should_render_agent_eval_template(model_info.signature):
        return
    # Create a Jinja2 environment and load the template
    env = jinja2.Environment(
        loader=jinja2.PackageLoader("mlflow.recipes", "resources"),
        autoescape=jinja2.select_autoescape(["html"]),
    )
    template = env.get_template("agent_evaluation_template.html")
    rendered_html = template.render({"modelUri": model_info.model_uri})
    from IPython.display import HTML, display

    display(HTML(rendered_html))
