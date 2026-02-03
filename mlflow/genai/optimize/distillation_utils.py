"""
Utilities for creating distillation datasets from traces.

This module provides functions to extract training data from MLflow traces,
supporting both single-turn and multi-turn prompt templates. The extracted
data can be used for knowledge distillation where a student model learns
to mimic a teacher model's responses.

Key capabilities:
- Two-stage matching: fast vague match + LLM-based verification and extraction
- Support for multi-turn chat templates (system + user + assistant messages)
- Filter LLM spans in complex traces to find those matching a specific prompt
- Handle traces with multiple LLM calls (e.g., agent workflows)
"""

import json
import logging
from typing import Any

from mlflow.prompt.constants import PROMPT_TEMPLATE_VARIABLE_PATTERN

_logger = logging.getLogger(__name__)

# Model used for variable extraction
_EXTRACTION_MODEL = "openai/gpt-4o-mini"


def _get_template_variables(template: str | list[dict[str, Any]]) -> list[str]:
    """Extract all variable names from a template (text or chat)."""
    if isinstance(template, str):
        return PROMPT_TEMPLATE_VARIABLE_PATTERN.findall(template)
    elif isinstance(template, list):
        # Chat template - extract from all message contents
        all_vars = []
        for msg in template:
            content = msg.get("content", "")
            if content:
                all_vars.extend(PROMPT_TEMPLATE_VARIABLE_PATTERN.findall(content))
        return all_vars
    return []


def _get_static_fragments(template: str | list[dict[str, Any]]) -> list[str]:
    """
    Extract static (non-variable) text fragments from a template.

    These fragments are used for fast vague matching to filter out
    obviously non-matching spans before calling the LLM.
    """
    import re

    # Convert chat template to single string for fragment extraction
    if isinstance(template, list):
        template_str = " ".join(msg.get("content", "") or "" for msg in template)
    else:
        template_str = template

    # Remove variable placeholders and split into fragments
    # Pattern matches {{var}} with optional whitespace
    no_vars = re.sub(r"\{\{\s*\w+\s*\}\}", " ", template_str)

    # Split on whitespace and filter short/empty fragments
    fragments = [f.strip() for f in no_vars.split() if len(f.strip()) >= 4]

    # Return unique fragments, keeping order
    seen = set()
    unique = []
    for f in fragments:
        if f not in seen:
            seen.add(f)
            unique.append(f)

    return unique[:10]  # Limit to 10 fragments for efficiency


def _compute_vague_match_score(
    template: str | list[dict[str, Any]],
    messages: list[dict[str, Any]],
) -> float:
    """
    Compute a vague match score between template and rendered messages.

    Returns a score between 0 and 1 indicating how likely the messages
    were generated from this template. Uses static fragment matching.
    """
    # Get static fragments from template
    fragments = _get_static_fragments(template)
    if not fragments:
        # No static fragments - can't do vague matching
        return 0.5  # Neutral score

    # Convert messages to searchable string
    messages_str = " ".join(msg.get("content", "") or "" for msg in messages).lower()

    # Count how many fragments appear in the messages
    matches = sum(1 for f in fragments if f.lower() in messages_str)

    return matches / len(fragments)


def _check_structure_match(
    template: str | list[dict[str, Any]],
    messages: list[dict[str, Any]],
) -> bool:
    """
    Check if the basic structure matches (message count and roles).

    For text templates, just checks there's a user message.
    For chat templates, checks message count and role sequence.
    """
    if isinstance(template, str):
        # Text template - just need at least one user message
        return any(msg.get("role") == "user" for msg in messages)

    # Chat template - check structure
    if len(template) != len(messages):
        return False

    for tmpl_msg, msg in zip(template, messages):
        if tmpl_msg.get("role") != msg.get("role"):
            return False

    return True


def _extract_variables_with_llm(
    template: str | list[dict[str, Any]],
    messages: list[dict[str, Any]],
    variable_names: list[str],
) -> dict[str, str] | None:
    """
    Use LLM to verify template match and extract variable values.

    Args:
        template: The prompt template (text or chat).
        messages: The rendered messages from a span.
        variable_names: List of variable names to extract.

    Returns:
        Dictionary of extracted variables, or None if not a match.
    """
    try:
        import litellm
    except ImportError:
        _logger.warning("litellm not installed, cannot extract variables with LLM")
        return None

    # Format template for display
    template_str = json.dumps(template, indent=2) if isinstance(template, list) else template

    # Format messages for display
    messages_str = json.dumps(messages, indent=2)

    # Build the extraction prompt
    system_prompt = """You are a template matching assistant. Your task is to:
1. Determine if the given RENDERED_MESSAGES were generated by formatting the TEMPLATE
2. If yes, extract the exact values that were substituted for each variable

The template uses {{variable_name}} syntax for placeholders.

Respond with a JSON object:
- If match: {"match": true, "variables": {"var1": "value1", "var2": "value2"}}
- If no match: {"match": false, "variables": {}}

Be precise - the template structure (roles, static text) must match exactly.
Only the {{variable}} placeholders should have different content."""

    user_prompt = f"""TEMPLATE:
{template_str}

RENDERED_MESSAGES:
{messages_str}

VARIABLES TO EXTRACT: {variable_names}

Analyze if RENDERED_MESSAGES matches TEMPLATE and extract variable values:"""

    try:
        response = litellm.completion(
            model=_EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        result_text = response.choices[0].message.content
        result = json.loads(result_text)

        if result.get("match"):
            variables = result.get("variables", {})
            # Verify all expected variables are present
            if all(var in variables for var in variable_names):
                return variables
            missing = set(variable_names) - set(variables.keys())
            _logger.debug(f"LLM extraction missing variables: {missing}")

        return None

    except Exception as e:
        _logger.warning(f"LLM extraction failed: {e}")
        return None


def extract_variables_from_messages(
    template: str | list[dict[str, Any]],
    messages: list[dict[str, Any]],
    vague_match_threshold: float = 0.5,
) -> dict[str, str] | None:
    """
    Extract template variables from rendered messages using two-stage matching.

    Stage 1: Fast vague match using static fragment matching
    Stage 2: LLM-based verification and extraction for candidates

    Args:
        template: The prompt template (text string or list of message dicts).
        messages: The rendered messages from an LLM span.
        vague_match_threshold: Minimum vague match score to proceed to LLM extraction.

    Returns:
        Dictionary mapping variable names to extracted values,
        or None if the messages don't match the template.
    """
    # Get variable names from template
    variable_names = _get_template_variables(template)
    if not variable_names:
        # No variables - just check structure matches
        if _check_structure_match(template, messages):
            return {}
        return None

    # Stage 1: Check basic structure
    if not _check_structure_match(template, messages):
        _logger.debug("Structure mismatch - skipping")
        return None

    # Stage 1: Fast vague match
    score = _compute_vague_match_score(template, messages)
    if score < vague_match_threshold:
        _logger.debug(f"Vague match score {score:.2f} below threshold {vague_match_threshold}")
        return None

    _logger.debug(f"Vague match score {score:.2f} - proceeding to LLM extraction")

    # Stage 2: LLM extraction
    return _extract_variables_with_llm(template, messages, variable_names)


def find_llm_spans(trace) -> list:
    """
    Find all LLM and CHAT_MODEL spans in a trace.

    Args:
        trace: MLflow trace object.

    Returns:
        List of spans with LLM or CHAT_MODEL type.
    """
    from mlflow.entities.span import SpanType

    spans = []
    for span_type in [SpanType.LLM, SpanType.CHAT_MODEL]:
        try:
            found = trace.search_spans(span_type=span_type)
            if found:
                spans.extend(found)
        except Exception as e:
            _logger.debug(f"Error searching for {span_type} spans: {e}")
    return spans


def extract_output_from_span(span) -> str | None:
    """
    Extract the output text from an LLM span.

    Handles various output formats:
    - Direct string output
    - Dict with "content" or "text" key
    - OpenAI-style chat completion response with "choices"
    - OpenAI responses API format (list with "message" type containing "output_text")

    Args:
        span: An MLflow span object.

    Returns:
        The extracted output text, or None if not found.
    """
    outputs = span.outputs
    if outputs is None:
        return None

    if isinstance(outputs, str):
        return outputs

    if isinstance(outputs, dict):
        # Direct content
        if "content" in outputs:
            return outputs["content"]
        if "text" in outputs:
            return outputs["text"]

        # OpenAI-style chat completion response
        if "choices" in outputs:
            choices = outputs.get("choices", [])
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message", {})
                if isinstance(message, dict):
                    return message.get("content")

        # OpenAI responses API format - output is a list in the dict
        if "output" in outputs:
            return _extract_from_responses_api_output(outputs["output"])

    # OpenAI responses API format - outputs is a list directly
    if isinstance(outputs, list):
        return _extract_from_responses_api_output(outputs)

    return None


def _extract_from_responses_api_output(output_list: list) -> str | None:
    """
    Extract text from OpenAI responses API output format.

    The format is a list of items, where message items have:
    {"type": "message", "content": [{"type": "output_text", "text": "..."}]}

    Args:
        output_list: List of output items from responses API.

    Returns:
        The extracted output text, or None if not found.
    """
    if not isinstance(output_list, list):
        return None

    for item in output_list:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            content = item.get("content", [])
            if isinstance(content, list):
                for content_item in content:
                    if isinstance(content_item, dict) and content_item.get("type") == "output_text":
                        return content_item.get("text")
    return None


def _get_messages_from_span(span) -> list[dict[str, Any]] | None:
    """Extract messages list from span inputs."""
    span_inputs = span.inputs
    if span_inputs is None:
        return None

    if isinstance(span_inputs, dict):
        return span_inputs.get("messages")
    elif isinstance(span_inputs, list):
        # Inputs might be the messages directly
        return span_inputs

    return None


def extract_from_span(
    span,
    prompt_template: str | list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """
    Extract inputs and output from a single LLM span.

    If a prompt_template is provided, uses two-stage matching (vague match + LLM)
    to verify and extract variable values. Otherwise, returns raw span inputs.

    Args:
        span: An MLflow span object (LLM or CHAT_MODEL type).
        prompt_template: Optional template to match against and extract variables.

    Returns:
        Tuple of (inputs_dict, output_string). Either may be None.
    """
    output = extract_output_from_span(span)
    messages = _get_messages_from_span(span)

    if prompt_template and messages:
        # Use two-stage matching
        extracted = extract_variables_from_messages(prompt_template, messages)
        if extracted is not None:
            return extracted, output
        # Template didn't match
        return None, None

    # No template - return raw inputs if available
    span_inputs = span.inputs
    if span_inputs is None:
        return None, output

    if isinstance(span_inputs, dict) and "messages" not in span_inputs:
        return span_inputs, output
    elif isinstance(span_inputs, str):
        return {"input": span_inputs}, output

    return None, output


def extract_matching_spans(
    trace,
    prompt_template: str | list[dict[str, Any]],
) -> list[tuple[dict[str, Any], str]]:
    """
    Extract data from all LLM spans matching the prompt template.

    Uses two-stage matching:
    1. Fast vague match to filter candidates
    2. LLM-based verification and extraction

    Args:
        trace: MLflow trace object.
        prompt_template: The prompt template to match (string or message list).

    Returns:
        List of (inputs_dict, output_string) tuples for matching spans.
    """
    llm_spans = find_llm_spans(trace)
    if not llm_spans:
        _logger.debug("No LLM spans found in trace")
        return []

    results = []
    for span in llm_spans:
        inputs, output = extract_from_span(span, prompt_template)
        if inputs is not None and output is not None:
            results.append((inputs, output))

    return results


def load_prompt_template(prompt_uri: str) -> str | list[dict[str, Any]]:
    """
    Load a prompt template from a URI.

    Args:
        prompt_uri: Prompt URI (e.g., "prompts:/my_prompt/1").

    Returns:
        The prompt template (string for text prompts, list for chat prompts).
    """
    from mlflow.genai.prompts import load_prompt

    prompt = load_prompt(prompt_uri)
    return prompt.template


def create_distillation_records_from_traces(
    traces: list,
    prompt_template: str | list[dict[str, Any]] | None = None,
    fallback_inputs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Create distillation dataset records from a list of traces.

    Each record contains:
    - inputs: The extracted input variables
    - expectations: {"expected_response": <teacher_output>}

    Args:
        traces: List of MLflow trace objects.
        prompt_template: Optional template for variable extraction and span filtering.
        fallback_inputs: Optional list of fallback inputs (same length as traces)
            to use when extraction fails.

    Returns:
        List of record dicts ready for dataset creation.
    """
    from mlflow.genai.utils.trace_utils import extract_response_from_trace

    records = []

    for i, trace in enumerate(traces):
        if trace is None:
            _logger.warning(f"Trace {i} is None, skipping")
            continue

        try:
            if prompt_template:
                # Extract all matching spans using two-stage matching
                matches = extract_matching_spans(trace, prompt_template)
                for inputs, output in matches:
                    records.append(
                        {
                            "inputs": inputs,
                            "expectations": {"expected_response": output},
                        }
                    )
            else:
                # No template - extract from first LLM span with fallbacks
                llm_spans = find_llm_spans(trace)
                inputs = None
                output = None

                if llm_spans:
                    inputs, output = extract_from_span(llm_spans[0], None)

                # Fallbacks
                if inputs is None and fallback_inputs and i < len(fallback_inputs):
                    inputs = fallback_inputs[i]
                if output is None:
                    output = extract_response_from_trace(trace)

                if inputs is not None and output is not None:
                    records.append(
                        {
                            "inputs": inputs,
                            "expectations": {"expected_response": output},
                        }
                    )
                else:
                    _logger.warning(f"Could not extract data from trace {i}")

        except Exception as e:
            _logger.warning(f"Error extracting from trace {i}: {e}")
            continue

    _logger.info(f"Created {len(records)} distillation records from {len(traces)} traces")
    return records
