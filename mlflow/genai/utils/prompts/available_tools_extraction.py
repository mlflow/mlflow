from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlflow.types.llm import ChatMessage

AVAILABLE_TOOLS_EXTRACTION_SYSTEM_PROMPT = """You are an expert in analyzing agent execution traces.
Your task is to examine an MLflow trace and identify all tools or functions that were
available to the LLM, not which tools were actually called.

CRITICAL: You MUST return ONLY valid JSON matching the schema below.
Do NOT return explanations, comments, or natural language. Return ONLY the JSON object.

## How You Should Analyze the Trace
Use the tools available to you to thoroughly inspect the trace:
1. Use list_spans
Retrieve the list of spans in the trace. Each span may contain tool definitions in
attributes or inputs.

2. Use GetSpanTool
For any span returned by list_spans, inspect its content—especially:
- inputs
- attributes
- metadata
These may include tool definitions or schemas given to the LLM.

3. Use SearchTraceRegexTool
Search the trace for keywords commonly associated with tool definitions, such as:
- "definition"
- "schema"
- "tool"
- "parameters"
- "functions"
Use these results to locate spans likely to contain tool schemas.

You must base your findings only on information contained in the trace.
Do not rely on or confuse the tools that you can use (like list_spans or GetSpanTool)
with the tools that were available to the LLM inside the trace. Only identify tools that
the trace itself shows were provided to the LLM.

## What to Look For
Search the trace for tool definitions or schemas that were provided to the LLM,
regardless of where they appear (span attributes, inputs, metadata, etc.).

A "tool definition" includes:
- The tool/function name
- An optional description
- An optional JSON schema for parameters

## Required Output Format
You MUST return a valid JSON object in exactly this format:

{output_example}

For every tool definition found, extract and return:
- type — Always "function"
- function.name — The tool's name (required)
- function.description — A description of the tool (use empty string "" if not available)
- function.parameters — The JSON parameter schema (use empty object {{}} if not available)

## Rules
- Return ONLY valid JSON. No explanations, no markdown, no comments.
- Return only unique tools. Two tool definitions should be treated as duplicates if
  any of the following are true: Their tool names/descriptions/parameter schemas are
  identical or nearly identical.
- If no tool definitions are present in the trace, return: {{"tools": []}}
- Only identify tools that the trace explicitly provides; do not infer or invent tools.
"""

AVAILABLE_TOOLS_EXTRACTION_USER_PROMPT = """
Please analyze the trace with the tools available to you and return the tools that were
available to the LLM in the trace.

Remember: respond with ONLY valid JSON matching the schema provided.
- Use double quotes for all property names and string values.
- Do not include comments, trailing commas, or explanatory text.
- Do not wrap the JSON in markdown (no ``` blocks).
- Do not include any text before or after the JSON.
- The response must be directly parseable by json.loads().

If no tools are found, return {"tools": []}.
"""


def get_available_tools_extraction_prompts(
    output_example: str,
) -> list["ChatMessage"]:
    """
    Generate system and user prompts for extracting available tools from a trace.

    Args:
        output_example: JSON string example of the expected output format.

    Returns:
        A list of chat messages [system_message, user_message] for tool extraction.
    """
    from mlflow.types.llm import ChatMessage

    system_prompt = AVAILABLE_TOOLS_EXTRACTION_SYSTEM_PROMPT.format(output_example=output_example)
    user_prompt = AVAILABLE_TOOLS_EXTRACTION_USER_PROMPT

    system_message = ChatMessage(role="system", content=system_prompt)
    user_message = ChatMessage(role="user", content=user_prompt)

    return [system_message, user_message]
