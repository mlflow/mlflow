import pytest

from mlflow.metrics.utils.prompt_template import PromptTemplate


def test_prompt_template_formatting():
    prompt = PromptTemplate(template_str="Say {foo}")
    assert prompt.format_prompt(foo="bar") == "Say bar"

    prompt = PromptTemplate(template_str="Say {foo} {baz}")
    with pytest.raises(KeyError, match="baz"):
        prompt.format_prompt(foo="bar")


def test_prompt_template_partial_formatting():
    prompt = PromptTemplate(template_str="Say {foo} {baz}")
    partial_prompt = prompt.partial_fill(foo="bar")
    assert partial_prompt.format_prompt(baz="qux") == "Say bar qux"
