import os
from typing import Optional

from openai import OpenAI
from openai.version import VERSION as OPENAI_VERSION
from promptflow import tool

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
    suffix: Optional[str] = None,
    max_tokens: int = 120,
    temperature: float = 1.0,
    top_p: float = 1.0,
    n: int = 1,
    logprobs: Optional[int] = None,
    echo: bool = False,
    stop: Optional[list] = None,
    presence_penalty: float = 0,
    frequency_penalty: float = 0,
    best_of: int = 1,
    logit_bias: Optional[dict] = None,
    user: str = "",
    **kwargs,
) -> str:
    if "OPENAI_API_KEY" not in os.environ:
        raise Exception("Please specify environment variables: OPENAI_API_KEY")

    echo = to_bool(echo)

    response = get_client().completions.create(
        prompt=prompt,
        model=deployment_name,
        # empty string suffix should be treated as None.
        suffix=suffix if suffix else None,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        n=n,
        logprobs=logprobs if logprobs else None,
        echo=echo,
        stop=stop if stop else None,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        best_of=best_of,
        # Logit bias must be a dict if we passed it to openai api.
        logit_bias=logit_bias if logit_bias else {},
        user=user,
    )

    # get first element because prompt is single.
    return response.choices[0].text
