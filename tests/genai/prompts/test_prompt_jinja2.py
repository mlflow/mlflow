from mlflow.entities.model_registry.prompt_version import PromptVersion
from mlflow.prompt.constants import (
    PROMPT_TYPE_JINJA2,
    PROMPT_TYPE_TEXT,
)


def test_jinja2_prompt_basic_rendering():
    p = PromptVersion("jinja-basic", 1, "Hello {% if name %}{{ name }}{% else %}Guest{% endif %}")
    assert p.format(name="Alice") == "Hello Alice"
    assert p.format() == "Hello Guest"


def test_jinja2_prompt_loop_rendering():
    template = "Fruits: {% for f in fruits %}{{ f }} {% endfor %}"
    p = PromptVersion("jinja-loop", 1, template)
    result = p.format(fruits=["apple", "banana", "cherry"])
    assert "apple" in result
    assert "banana" in result
    assert "cherry" in result


def test_jinja2_prompt_no_sandbox():
    # Jinja2 is detected by {% %} syntax
    p = PromptVersion("jinja-nosandbox", 1, "{% set x = 2 * 3 %}{{ x }}")
    assert p.format(use_jinja_sandbox=False) == "6"


def test_jinja2_prompt_type_detection():
    jinja_prompt = PromptVersion(
        "jinja-detect", 1, "Hello {% if name %}{{ name }}{% else %}World{% endif %}"
    )
    assert jinja_prompt._prompt_type == PROMPT_TYPE_JINJA2


def test_text_prompt_type_detection():
    text_prompt = PromptVersion("text-detect", 1, "Hello {{name}}")
    assert text_prompt._prompt_type == PROMPT_TYPE_TEXT


def test_plain_text_formatting():
    p = PromptVersion("plain-text", 1, "Hello {{name}}!")
    assert p.format(name="Alice") == "Hello Alice!"


def test_jinja2_filters():
    p = PromptVersion("jinja-filter", 1, "{% set name = 'alice' %}{{ name | upper }}")
    assert p.format() == "ALICE"
