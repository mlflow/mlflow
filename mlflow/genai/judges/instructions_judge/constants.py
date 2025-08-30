"""Constants for InstructionsJudge."""

INSTRUCTIONS_JUDGE_EVALUATION_PROMPT_TEMPLATE = """
You are evaluating the provided data according to specific task instructions.

You MUST follow this methodology:

EVALUATION APPROACH:
1. Carefully review all provided data (inputs, outputs, and/or expectations as available)
2. Apply the task instructions systematically to evaluate the data
3. Consider the context and relationships between different data elements
4. Make your evaluation based on the criteria specified in the task instructions

IMPORTANT GUIDELINES:
- Base your evaluation strictly on the data provided
- Be objective and consistent in your assessment
- If expectations are provided, use them as ground truth for comparison
- Focus on the specific aspects mentioned in the task instructions

Task Instructions
-----------------
{{task_instructions}}
"""
