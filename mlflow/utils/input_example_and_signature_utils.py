from mlflow.models import infer_signature

FAILED_INPUT_EXAMPLE_PREFIX_TEXT = "Failed to gather input example: "
FAILED_MODEL_SIGNATURE_PREFIX_TEXT = "Failed to infer model signature: "
ENSURE_AUTOLOGGING_ENABLED_TEXT = (
    "please ensure that autologging is enabled before constructing the dataset."
)


class _InputExampleInfo:
    def __init__(self, input_example=None, error_msg=None):
        self.input_example = input_example
        self.error_msg = error_msg


def handle_input_example_and_signature(
    input_example_info, log_input_example, log_model_signature, model_predict_fn
):
    if input_example_info is None:
        return (
            None,
            None,
            (FAILED_INPUT_EXAMPLE_PREFIX_TEXT + ENSURE_AUTOLOGGING_ENABLED_TEXT)
            if log_input_example
            else None,
            (FAILED_MODEL_SIGNATURE_PREFIX_TEXT + ENSURE_AUTOLOGGING_ENABLED_TEXT)
            if log_model_signature
            else None,
        )

    (
        input_example,
        signature,
        input_example_user_msg,
        model_signature_user_msg,
    ) = _process_and_build_user_facing_messages(
        input_example_info, model_predict_fn, log_model_signature
    )

    return (
        input_example if log_input_example else None,
        signature if log_model_signature else None,
        input_example_user_msg if log_input_example else None,
        model_signature_user_msg if log_model_signature else None,
    )


def _process_and_build_user_facing_messages(
    input_example_info, model_predict_fn, log_model_signature
):
    input_example = None
    input_example_user_msg = None
    try:
        input_example = input_example_info.input_example
        if input_example is None:
            raise Exception(input_example_info.error_msg)
    except Exception as e:  # pylint: disable=broad-except
        input_example_user_msg = FAILED_INPUT_EXAMPLE_PREFIX_TEXT + str(e)

    model_signature = None
    model_signature_user_msg = None
    if log_model_signature:
        try:
            if input_example is None:
                raise Exception(
                    "could not sample data to infer model signature: "
                    + input_example_info.error_msg
                )

            model_output = model_predict_fn(input_example)

            model_signature = infer_signature(input_example, model_output)
        except Exception as e:  # pylint: disable=broad-except
            model_signature_user_msg = FAILED_MODEL_SIGNATURE_PREFIX_TEXT + str(e)

    return input_example, model_signature, input_example_user_msg, model_signature_user_msg
