from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import dspy


def format_optimized_prompt(
    program: "dspy.Predict",
    input_fields: dict[str, type],
    convert_to_single_text: bool,
) -> dict[str, Any] | str:
    import dspy

    messages = dspy.settings.adapter.format(
        signature=program.signature,
        demos=program.demos,
        inputs={key: "{{" + key + "}}" for key in input_fields.keys()},
    )

    if convert_to_single_text:
        messages = "\n\n".join(
            [
                f"<{message['role']}>\n{message['content']}\n</{message['role']}>"
                for message in messages
            ]
        )

    return messages
