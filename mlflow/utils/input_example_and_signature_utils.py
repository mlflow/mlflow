FAILED_INPUT_EXAMPLE_PREFIX_TEXT = "Failed to gather input example: "
FAILED_MODEL_SIGNATURE_PREFIX_TEXT = "Failed to infer model signature: "
ENSURE_AUTOLOGGING_ENABLED_TEXT = (
    "please ensure that autologging is enabled before constructing the dataset."
)


class _InputExampleInfo:
    def __init__(self, input_example=None, error_msg=None):
        self.input_example = input_example
        self.error_msg = error_msg

def handle_input_example_and_signature(get_input_example, get_model_signature, log_input_example, log_model_signature, logger):
    input_example = None
    input_example_user_msg = None
    input_example_failure_msg = None
    if log_input_example or log_model_signature:
        try:
            input_example = get_input_example()
        except Exception as e:  # pylint: disable=broad-except
            input_example_failure_msg = str(e)
            input_example_user_msg = FAILED_INPUT_EXAMPLE_PREFIX_TEXT + str(e)

    model_signature = None
    model_signature_user_msg = None
    if log_model_signature:
        try:
            if input_example is None:
                raise Exception("could not sample data to infer model signature: " + input_example_failure_msg)
            model_signature = get_model_signature(input_example)
        except Exception as e:
            model_signature_user_msg = FAILED_MODEL_SIGNATURE_PREFIX_TEXT + str(e)

    if log_input_example and input_example_user_msg is not None:
        logger.warning(input_example_user_msg)
    if log_model_signature and model_signature_user_msg is not None:
        logger.warning(model_signature_user_msg)

    return input_example if log_input_example else None, model_signature if log_model_signature else None