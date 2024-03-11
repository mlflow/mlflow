from promptflow import tool


@tool
def my_python_tool(prompt: str) -> str:
    # Echo the prompt and return directly
    return prompt
