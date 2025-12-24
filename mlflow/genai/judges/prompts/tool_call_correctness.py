import json

from mlflow.genai.judges.utils.formatting_utils import (
    format_available_tools,
    format_tools_called,
)
from mlflow.genai.prompts.utils import format_prompt
from mlflow.genai.utils.type import FunctionCall
from mlflow.types.chat import ChatTool

TOOL_CALL_CORRECTNESS_FEEDBACK_NAME = "tool_call_correctness"

# Shared output format for all prompt variants
_OUTPUT_FORMAT = """
Please evaluate whether the agent's tool calls and their arguments are correct and reasonable using only the following json format. Return "yes" if the tool calls and arguments are correct and reasonable, otherwise return "no".
Do not use any markdown formatting or output additional lines.
{
  "rationale": "Reason for the assessment. If incorrect or unreasonable tool calls are found, identify which specific calls or arguments are problematic and explain why. If all tool calls and arguments are correct, explain why they are appropriate. Start each rationale with `Let's think step by step`",
  "result": "yes|no"
}\
"""  # noqa: E501

# Ordering instruction variants
ORDERING_INSTRUCTION_CHECK = (
    "3) Ordering\n- Consider whether the order of tool calls matches the expected order."
)
ORDERING_INSTRUCTION_IGNORE = "Note: The order of tool calls does not need to match exactly."

# Evaluation criteria for ground-truth-free mode
_GROUND_TRUTH_FREE_CRITERIA = """\
1) Need for tools
- Was using any tool necessary or helpful for this request?
- Did the agent fail to use an obviously appropriate tool that was available?

2) Tool selection
- For each step, is the chosen tool a good match for the subtask, given the tool descriptions?
- Did the agent avoid tools that are clearly irrelevant, overpowered, or disallowed for the request?

3) Arguments and intent alignment
- Do the arguments match the tool's schema?
- Are the arguments clearly grounded in the user's request and the tool's documented purpose?
- Across calls, are key parameters provided in ways that logically follow from prior tool outputs
or user messages, rather than arbitrary changes?

4) Tool flow and combinations
- When multiple tools are used, is the overall sequence of tool choices logically sound?
- Are follow-up tool calls justified by what the agent appears to be trying to achieve with respect
to the user's request?

{{ordering_instruction}}"""

# Evaluation criteria for full expectations (names + arguments)
_FULL_EXPECTATIONS_CRITERIA = """\
1) Tool selection match
- Are the same tools being called (or semantically equivalent alternatives)?
- Are there any missing or extra tool calls compared to expectations?

2) Argument match
- Do the actual arguments convey the same intent as the expected arguments, even if phrased
differently?
- Are there any significant differences in argument values that would change the outcome?

{{ordering_instruction}}"""

# Evaluation criteria for partial expectations (names only)
_PARTIAL_EXPECTATIONS_CRITERIA = """\
1) Tool selection match
- Do the actual tool calls match the expected tool names?
- Are there any missing or extra tools compared to expectations?

2) Argument reasonableness
- Are the arguments provided reasonable given the user request and tool definitions?
- Do the arguments match the tool's schema and documented purpose?

{{ordering_instruction}}"""

# Unified prompt template
_PROMPT_TEMPLATE = """\
{{preamble}}

Focus only on the choice of tools and the arguments passed to them. Do NOT judge whether the tools'
outputs or implementations are correct.

Evaluate:

{{evaluation_criteria}}

<request>
{{request}}
</request>

{{expected_section}}<actual_tool_calls>
{{tools_called}}
</actual_tool_calls>

<available_tools>
{{available_tools}}
</available_tools>"""

# Preamble variants
_GROUND_TRUTH_FREE_PREAMBLE = """\
Consider whether the agent selected appropriate tools and called with the correct arguments for the
task.

Given the user's request, the available tools (including their described capabilities/constraints),
and the sequence of tool calls made by the agent, evaluate if the agent chose suitable tools and
used them in a reasonable way."""

_FULL_EXPECTATIONS_PREAMBLE = """\
Compare the actual tool calls against the expected tool calls to determine if they are correct.

Given the user's request, the expected tool calls (ground truth), and the actual tool calls made
by the agent, evaluate whether the actual tool calls semantically match the expected ones."""

_PARTIAL_EXPECTATIONS_PREAMBLE = """\
Evaluate tool call correctness by comparing tool selection against expected tool names, and
evaluating whether arguments are reasonable.

Given the user's request, the expected tool names (ground truth), and the actual tool calls made
by the agent, evaluate whether the agent selected the correct tools and used reasonable arguments.\
"""

# Legacy exports for backward compatibility
TOOL_CALL_CORRECTNESS_PROMPT_INSTRUCTIONS = (
    _GROUND_TRUTH_FREE_PREAMBLE
    + "\n\nFocus only on the choice of tools and the arguments passed to them. Do NOT judge "
    "whether the tools'\noutputs or implementations are correct.\n\nEvaluate:\n\n"
    + _GROUND_TRUTH_FREE_CRITERIA
    + "\n\n<request>\n{{request}}\n</request>\n\n<available_tools>\n{{available_tools}}\n"
    "</available_tools>\n\n<tools_called>\n{{tools_called}}\n</tools_called>"
)
TOOL_CALL_CORRECTNESS_PROMPT_OUTPUT = _OUTPUT_FORMAT
TOOL_CALL_CORRECTNESS_PROMPT = TOOL_CALL_CORRECTNESS_PROMPT_INSTRUCTIONS + _OUTPUT_FORMAT


def _format_expected_calls(expected_calls: list[FunctionCall], compare_arguments: bool) -> str:
    lines = []
    for i, call in enumerate(expected_calls, 1):
        if compare_arguments and call.arguments:
            lines.append(f"Expected Tool Call {i}: {call.name}")
            lines.append(f"  Arguments: {json.dumps(call.arguments)}")
        else:
            lines.append(f"Expected Tool {i}: {call.name}")
    return "\n".join(lines) if lines else "No expected tool calls provided."


def get_prompt(
    request: str,
    tools_called: list[FunctionCall],
    available_tools: list[ChatTool],
    expected_calls: list[FunctionCall] | None = None,
    compare_arguments: bool = True,
    check_order: bool = False,
) -> str:
    """
    Generate tool call correctness evaluation prompt.

    Args:
        request: The original user request that the agent is trying to fulfill
        tools_called: The sequence of tools that were called by the agent
        available_tools: The set of available tools
        expected_calls: Optional list of expected tool calls for ground-truth comparison.
            If None, uses ground-truth-free evaluation.
        compare_arguments: If True, compare both tool names and arguments (full expectations).
            If False, compare only tool names (partial expectations).
        check_order: If True, ask LLM to consider ordering of tool calls.
    """
    available_tools_str = format_available_tools(available_tools)
    tools_called_str = format_tools_called(tools_called)

    if check_order:
        ordering_instruction = ORDERING_INSTRUCTION_CHECK
    else:
        ordering_instruction = ORDERING_INSTRUCTION_IGNORE

    if expected_calls is None:
        # Ground-truth-free mode
        criteria = format_prompt(
            _GROUND_TRUTH_FREE_CRITERIA, ordering_instruction=ordering_instruction
        )
        return format_prompt(
            _PROMPT_TEMPLATE + _OUTPUT_FORMAT,
            preamble=_GROUND_TRUTH_FREE_PREAMBLE,
            evaluation_criteria=criteria,
            expected_section="",
            request=request,
            available_tools=available_tools_str,
            tools_called=tools_called_str,
        )

    # With expectations mode
    expected_calls_str = _format_expected_calls(expected_calls, compare_arguments)

    if compare_arguments:
        preamble = _FULL_EXPECTATIONS_PREAMBLE
        criteria = format_prompt(
            _FULL_EXPECTATIONS_CRITERIA, ordering_instruction=ordering_instruction
        )
        expected_section = (
            f"<expected_tool_calls>\n{expected_calls_str}\n</expected_tool_calls>\n\n"
        )
    else:
        preamble = _PARTIAL_EXPECTATIONS_PREAMBLE
        criteria = format_prompt(
            _PARTIAL_EXPECTATIONS_CRITERIA, ordering_instruction=ordering_instruction
        )
        expected_section = (
            f"<expected_tool_names>\n{expected_calls_str}\n</expected_tool_names>\n\n"
        )

    return format_prompt(
        _PROMPT_TEMPLATE + _OUTPUT_FORMAT,
        preamble=preamble,
        evaluation_criteria=criteria,
        expected_section=expected_section,
        request=request,
        available_tools=available_tools_str,
        tools_called=tools_called_str,
    )
