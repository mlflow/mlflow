from mlflow.genai.judges.utils.parsing_utils import (
    _sanitize_justification,
    _strip_markdown_code_blocks,
)


def test_strip_markdown_no_markdown_returns_unchanged():
    response = '{"result": "yes", "rationale": "looks good"}'
    assert _strip_markdown_code_blocks(response) == response


def test_strip_markdown_json_code_block():
    response = """```json
{"result": "yes", "rationale": "looks good"}
```"""
    expected = '{"result": "yes", "rationale": "looks good"}'
    assert _strip_markdown_code_blocks(response) == expected


def test_strip_markdown_code_block_without_language():
    response = """```
{"result": "yes", "rationale": "looks good"}
```"""
    expected = '{"result": "yes", "rationale": "looks good"}'
    assert _strip_markdown_code_blocks(response) == expected


def test_strip_markdown_code_block_with_whitespace():
    response = """  ```json
{"result": "yes", "rationale": "looks good"}
```  """
    expected = '{"result": "yes", "rationale": "looks good"}'
    assert _strip_markdown_code_blocks(response) == expected


def test_strip_markdown_multiline_json():
    response = """```json
{
  "result": "yes",
  "rationale": "looks good"
}
```"""
    expected = """{
  "result": "yes",
  "rationale": "looks good"
}"""
    assert _strip_markdown_code_blocks(response) == expected


def test_strip_markdown_empty_string():
    assert _strip_markdown_code_blocks("") == ""


def test_strip_markdown_only_opening_backticks():
    response = '```json\n{"result": "yes"}'
    expected = '{"result": "yes"}'
    assert _strip_markdown_code_blocks(response) == expected


def test_strip_markdown_backticks_in_middle_not_stripped():
    response = '{"result": "yes", "rationale": "use ``` for code"}'
    assert _strip_markdown_code_blocks(response) == response


def test_strip_markdown_nested_backticks_inside_code_block():
    response = """```json
{"result": "yes", "code": "use ``` for code blocks"}
```"""
    expected = '{"result": "yes", "code": "use ``` for code blocks"}'
    assert _strip_markdown_code_blocks(response) == expected


def test_strip_markdown_multiple_lines_before_closing():
    response = """```json
{"result": "yes"}
{"another": "line"}
{"more": "data"}
```"""
    expected = """{"result": "yes"}
{"another": "line"}
{"more": "data"}"""
    assert _strip_markdown_code_blocks(response) == expected


def test_strip_markdown_python_language():
    response = """```python
print("hello")
```"""
    expected = 'print("hello")'
    assert _strip_markdown_code_blocks(response) == expected


def test_strip_markdown_closing_with_trailing_content():
    response = """```json
{"result": "yes"}
```
This text should not be included"""
    expected = '{"result": "yes"}'
    assert _strip_markdown_code_blocks(response) == expected


def test_sanitize_removes_step_by_step_prefix():
    justification = "Let's think step by step. The answer is correct."
    expected = "The answer is correct."
    assert _sanitize_justification(justification) == expected


def test_sanitize_no_prefix_unchanged():
    justification = "The answer is correct."
    assert _sanitize_justification(justification) == justification


def test_sanitize_empty_string():
    assert _sanitize_justification("") == ""


def test_sanitize_only_prefix():
    justification = "Let's think step by step. "
    assert _sanitize_justification(justification) == ""


def test_sanitize_prefix_in_middle():
    justification = "First, Let's think step by step. Then continue."
    expected = "First, Then continue."
    assert _sanitize_justification(justification) == expected


def test_sanitize_multiple_occurrences():
    justification = "Let's think step by step. A. Let's think step by step. B."
    expected = "A. B."
    assert _sanitize_justification(justification) == expected


def test_sanitize_case_sensitive():
    justification = "let's think step by step. The answer is correct."
    assert _sanitize_justification(justification) == justification


def test_sanitize_prefix_without_trailing_space():
    justification = "Let's think step by step.The answer is correct."
    assert _sanitize_justification(justification) == justification


def test_sanitize_with_newlines():
    justification = "Let's think step by step. First point.\nSecond point."
    expected = "First point.\nSecond point."
    assert _sanitize_justification(justification) == expected
