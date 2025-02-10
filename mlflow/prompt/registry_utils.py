import urllib.parse
from typing import Optional

from mlflow.entities.model_registry.prompt import IS_PROMPT_TAG_KEY
from mlflow.exceptions import MlflowException


def add_prompt_filter_string(
    filter_string: Optional[str], is_prompt: bool = False
) -> Optional[str]:
    """
    Additional filter string to include/exclude prompts from the result.
    By default, exclude prompts from the result.
    """
    if IS_PROMPT_TAG_KEY not in (filter_string or ""):
        prompt_filter_query = (
            f"tag.`{IS_PROMPT_TAG_KEY}` = 'true'"
            if is_prompt
            else f"tag.`{IS_PROMPT_TAG_KEY}` != 'true'"
        )
        if filter_string:
            filter_string = f"{filter_string} AND {prompt_filter_query}"
        else:
            filter_string = prompt_filter_query
    return filter_string


def parse_prompt_uri(uri: str) -> tuple[str, str]:
    """
    Parse prompt URI into prompt name and prompt version.
    'prompt:/<name>/<version>' -> ('<name>', '<version>')
    """
    parsed = urllib.parse.urlparse(uri)

    if parsed.scheme != "prompts":
        raise MlflowException.invalid_parameter_value(
            f"Invalid prompt URI: {uri}. Expected schema 'prompts:/<name>/<version>'"
        )

    path = parsed.path
    if path.count("/") != 2:
        raise MlflowException.invalid_parameter_value(
            f"Invalid prompt URI: {uri}. Expected schema 'prompts:/<name>/<version>'"
        )

    return path.split("/")[1:]
