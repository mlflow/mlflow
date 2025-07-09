from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import dspy


def format_optimized_prompt(program: "dspy.Predict", input_fields: dict[str, type]) -> str:
    import dspy

    messages = dspy.settings.adapter.format(
        signature=program.signature,
        demos=program.demos,
        inputs={key: "{{" + key + "}}" for key in input_fields.keys()},
    )

    return "\n\n".join(
        [f"<{message['role']}>\n{message['content']}\n</{message['role']}>" for message in messages]
    )
