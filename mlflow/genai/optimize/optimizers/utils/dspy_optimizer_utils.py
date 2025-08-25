from typing import TYPE_CHECKING, Any

from mlflow.utils.annotations import experimental

if TYPE_CHECKING:
    import dspy


@experimental(version="3.3.0")
def format_dspy_prompt(
    program: "dspy.Predict",
    convert_to_single_text: bool,
) -> dict[str, Any] | str:
    import dspy

    signature = program.signature
    messages = dspy.settings.adapter.format(
        signature=signature,
        demos=program.demos,
        inputs={key: "{{" + key + "}}" for key in signature.input_fields.keys()},
    )

    if convert_to_single_text:
        messages = "\n\n".join(
            [
                f"<{message['role']}>\n{message['content']}\n</{message['role']}>"
                for message in messages
            ]
        )

    return messages
