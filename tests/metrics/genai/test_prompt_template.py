from mlflow.metrics.genai.prompt_template import PromptTemplate


def test_prompt_template_flat_str_no_variables():
    prompt = PromptTemplate(template_str="Say {foo}")
    assert prompt.format(foo="bar") == "Say bar"

    prompt = PromptTemplate(template_str="Say {foo} {baz}")
    assert prompt.format(foo="bar") == ""

    prompt = PromptTemplate(template_str="Say foo")
    assert prompt.format() == "Say foo"

    prompt = PromptTemplate(template_str="Say {foo} {foo}")
    assert prompt.format(foo="bar") == "Say bar bar"

    prompt = PromptTemplate(template_str="Say {foo}")
    assert prompt.format(foo=None) == ""


def test_prompt_template_arr_str_no_variables():
    prompt = PromptTemplate(template_str=["Say {foo}"])
    assert prompt.format(foo="bar") == "Say bar"

    prompt = PromptTemplate(template_str=["Say {foo} {baz}"])
    assert prompt.format(foo="bar") == ""

    prompt = PromptTemplate(template_str=["Say {foo}", " {baz}"])
    assert prompt.format(foo="bar") == "Say bar"

    prompt = PromptTemplate(template_str=["Say {foo}", " {baz}"])
    assert prompt.format(baz="qux") == " qux"

    prompt = PromptTemplate(template_str=["Say foo", ", and say bar"])
    assert prompt.format() == "Say foo, and say bar"

    prompt = PromptTemplate(template_str=["Say {foo} {foo}", ", and {foo}", ", and {foo} again"])
    assert prompt.format(foo="bar") == "Say bar bar, and bar, and bar again"

    prompt = PromptTemplate(template_str=["Say {foo}", " {baz}"])
    assert prompt.format(foo="bar", baz="qux") == "Say bar qux"

    prompt = PromptTemplate(template_str=["Say {foo}", " {baz}"])
    assert prompt.format(foo="bar", baz=None) == "Say bar"


def test_prompt_template_partial_formatting():
    prompt = PromptTemplate(template_str="Say {foo} {baz}")
    partial_prompt = prompt.partial_fill(foo="bar")
    assert partial_prompt.format(baz="qux") == "Say bar qux"
    assert partial_prompt.format(baz=None) == ""

    prompt = PromptTemplate(template_str=["Say {foo}", " {baz}"])
    partial_prompt = prompt.partial_fill(foo="bar")
    assert partial_prompt.format(baz="qux") == "Say bar qux"
    assert partial_prompt.format(baz=None) == "Say bar"
    assert partial_prompt.format() == "Say bar"
