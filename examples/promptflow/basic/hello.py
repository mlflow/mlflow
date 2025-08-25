import os
from typing import Any

from _utils import parse_chat
from openai import OpenAI
from openai.version import VERSION as OPENAI_VERSION
from promptflow import tool
from render_template import render_template

# The inputs section will change based on the arguments of the tool function, after you save the code
# Adding type to arguments and return value will help the system show the types properly
# Please update the function name/signature per need


def to_bool(value) -> bool:
    return str(value).lower() == "true"


def get_client():
    if OPENAI_VERSION.startswith("0."):
        raise Exception(
            "Please upgrade your OpenAI package to version >= 1.0.0 or using the command: pip install --upgrade openai."
        )
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


@tool
def my_python_tool(
    prompt: str,
    deployment_name: str,
    max_tokens: int = 120,
    temperature: float = 1.0,
    top_p: float = 1.0,
    n: int = 1,
    stop: list[str] | None = None,
    presence_penalty: float = 0,
    frequency_penalty: float = 0,
    logit_bias: dict[str, Any] | None = None,
    user: str = "",
    **kwargs,
) -> str:
    if "OPENAI_API_KEY" not in os.environ:
        raise Exception("Please specify environment variables: OPENAI_API_KEY")

    chat_str = render_template(prompt, **kwargs)
    messages = parse_chat(chat_str)

    response = get_client().chat.completions.create(
        messages=messages,
        model=deployment_name,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        n=n,
        stop=stop if stop else None,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        # Logit bias must be a dict if we passed it to openai api.
        logit_bias=logit_bias if logit_bias else {},
        user=user,
    )

    return response.choices[0].message.content
