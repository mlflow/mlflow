import re

import pytest

from mlflow.exceptions import MlflowException
from mlflow.metrics.genai.prompt_template import PromptTemplate


def test_prompt_template_formatting():
    prompt = PromptTemplate(template_str="Say {foo}")
    assert prompt.format(foo="bar") == "Say bar"

    prompt = PromptTemplate(template_str="Say {foo} {baz}")
    with pytest.raises(KeyError, match="baz"):
        prompt.format(foo="bar")

    prompt = PromptTemplate(template_str="Say foo")
    assert prompt.format() == "Say foo"

    with pytest.raises(
        MlflowException,
        match=re.escape(
            "The provided variables ['foo'] are not a subset of "
            "the extracted variables ['foo', 'baz'] from the template string"
        ),
    ):
        prompt = PromptTemplate(template_str="Say {foo} {baz}", variables=["foo"])

    prompt = PromptTemplate(template_str="Say {foo} {foo}")
    assert prompt.format(foo="bar") == "Say bar bar"


def test_prompt_template_partial_formatting():
    prompt = PromptTemplate(template_str="Say {foo} {baz}")
    partial_prompt = prompt.partial_fill(foo="bar")
    assert partial_prompt.format(baz="qux") == "Say bar qux"
