"""
Constants for the InstructionsJudge module.

This module contains constant values used by the InstructionsJudge class,
including the augmented prompt template for trace-based evaluation.
"""

# Augmented prompt template for trace-based evaluation
INSTRUCTIONS_JUDGE_TRACE_PROMPT_TEMPLATE = """
You have access to tools to analyze the trace. You *must* identify *all* the tools you'll need to
use to complete the task instructions. Then, methodically use these tools to complete the task
instructions.

You MUST follow this methodology:

REQUIRED STEPS (Fetch and analyze the results in order!):
1. ALWAYS fetch the trace metadata to understand the overall context, timing, and execution details
2. ALWAYS list all spans to see the complete trace structure and understand the flow of execution
3. ALWAYS retrieve the root span to understand the top-level inputs and outputs of the interaction.
   The root span typically contains the overall inputs to the agent and the final outputs.

After completing these required steps, use more tools *if and only if* needed. For example:
- Retrieve specific spans by ID to examine their details
- Search for patterns or specific text across the entire trace
- Continue using tools until you have gathered sufficient information

IMPORTANT: It's recommended to call tools in parallel to save time. For example, you can
retrieve multiple spans using separate tool calls with different IDs.

Task Instructions
-----------------
{task_instructions}
"""