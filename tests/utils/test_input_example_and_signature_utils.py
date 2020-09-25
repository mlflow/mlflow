import pytest

import mlflow.utils.input_example_and_signature_utils as utils


@pytest.fixture(autouse=True)
def infer_signature_mock(monkeypatch):
    monkeypatch.setattr(
        "mlflow.utils.input_example_and_signature_utils.infer_signature", lambda _1, _2: "signature"
    )


@pytest.fixture()
def successful_input_example_info():
    return utils._InputExampleInfo("data", None)


def some_fn(some_arg):
    return "signature"


# Testing the wrapper function
@pytest.mark.parametrize("log_input_example", [True, False])
@pytest.mark.parametrize("log_model_signature", [True, False])
def test_filters_out_data_and_messages(log_input_example, log_model_signature, monkeypatch):
    monkeypatch.setattr(
        "tests.utils.test_input_example_and_signature_utils.utils._process_and_build_user_facing_messages",
        lambda _1, _2, _3: (
            "data",
            "signature",
            "input example user msg",
            "model signature user msg",
        ),
    )
    (
        input_example,
        signature,
        input_example_user_msg,
        model_signature_user_msg,
    ) = utils.handle_input_example_and_signature(
        successful_input_example_info, log_input_example, log_model_signature, some_fn
    )

    assert (input_example == "data") if log_input_example else (input_example is None)
    assert (signature == "signature") if log_model_signature else (signature is None)
    assert (
        (input_example_user_msg == "input example user msg")
        if log_input_example
        else (input_example_user_msg is None)
    )
    assert (
        (model_signature_user_msg == "model signature user msg")
        if log_model_signature
        else (model_signature_user_msg is None)
    )


# Testing the wrapper function
@pytest.mark.parametrize("log_input_example", [True, False])
@pytest.mark.parametrize("log_model_signature", [True, False])
def test_returns_nothing_if_no_input_example_info(
    log_input_example, log_model_signature, monkeypatch
):
    monkeypatch.setattr(
        "tests.utils.test_input_example_and_signature_utils.utils._process_and_build_user_facing_messages",
        lambda _1, _2, _3: (
            "data",
            "signature",
            "input example user msg",
            "model signature user msg",
        ),
    )
    (
        input_example,
        signature,
        input_example_user_msg,
        model_signature_user_msg,
    ) = utils.handle_input_example_and_signature(
        None, log_input_example, log_model_signature, some_fn
    )

    assert input_example is None
    assert signature is None
    assert (
        (
            input_example_user_msg
            == utils.FAILED_INPUT_EXAMPLE_PREFIX_TEXT + utils.ENSURE_AUTOLOGGING_ENABLED_TEXT
        )
        if log_input_example
        else (input_example_user_msg is None)
    )
    assert (
        (
            model_signature_user_msg
            == utils.FAILED_MODEL_SIGNATURE_PREFIX_TEXT + utils.ENSURE_AUTOLOGGING_ENABLED_TEXT
        )
        if log_model_signature
        else (model_signature_user_msg is None)
    )


# Testing the main logic
def test_if_input_example_fails():
    failed_input_example_info = utils._InputExampleInfo(None, "exceptiontext")

    (
        input_example,
        signature,
        input_example_user_msg,
        model_signature_user_msg,
    ) = utils._process_and_build_user_facing_messages(failed_input_example_info, some_fn, True)

    assert input_example is None
    assert signature is None
    assert input_example_user_msg == utils.FAILED_INPUT_EXAMPLE_PREFIX_TEXT + "exceptiontext"
    assert (
        utils.FAILED_MODEL_SIGNATURE_PREFIX_TEXT + "could not sample" in model_signature_user_msg
        and "exceptiontext" in model_signature_user_msg
    )


def test_if_model_signature_inference_fails(successful_input_example_info):
    def throws(_):
        raise Exception("exceptiontext")

    (
        input_example,
        signature,
        input_example_user_msg,
        model_signature_user_msg,
    ) = utils._process_and_build_user_facing_messages(successful_input_example_info, throws, True)

    assert input_example == "data"
    assert signature is None
    assert input_example_user_msg is None
    assert utils.FAILED_MODEL_SIGNATURE_PREFIX_TEXT + "exceptiontext" in model_signature_user_msg


def test_happy_path_works(successful_input_example_info):
    (
        input_example,
        signature,
        input_example_user_msg,
        model_signature_user_msg,
    ) = utils._process_and_build_user_facing_messages(successful_input_example_info, some_fn, True)

    assert input_example == "data"
    assert signature == "signature"
    assert input_example_user_msg is None
    assert model_signature_user_msg is None


def test_avoids_inferring_if_not_needed(successful_input_example_info, monkeypatch):
    # We patch the infer_signature function to throw.
    # If it was invoked, model_signature_user_msg should be populated,
    #   as there was an exception when trying to infer.
    # If it was not invoked, there should be no exception, hence the message
    #   is empty.
    def throws(_1, _2):
        raise Exception()

    monkeypatch.setattr(
        "mlflow.utils.input_example_and_signature_utils.infer_signature",
        throws
    )

    (
        input_example,
        signature,
        input_example_user_msg,
        model_signature_user_msg,
    ) = utils._process_and_build_user_facing_messages(
        successful_input_example_info, some_fn, False
    )

    assert model_signature_user_msg is None